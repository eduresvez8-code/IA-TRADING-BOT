"""Persistencia del bot: aiosqlite (modo WAL) + Parquet.

Dos almacenes porque sirven a dos patrones de acceso distintos:
- SQLite: escrituras pequeñas y constantes (la vela que acaba de cerrar, una
  noticia, un trade) y consultas puntuales ("las últimas 200 velas").
- Parquet: lotes grandes e inmutables (1 año de histórico) que el backtester
  lee completos a un DataFrame de una sola vez.

Regla del repo: el modo WAL se activa aquí, en la inicialización, siempre.
Varios módulos async comparten esta BD; sin WAL, una escritura bloquea todas
las lecturas y aparecen errores `database is locked`.
"""

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pandas as pd

from src.core.models import Candle

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    open_time INTEGER NOT NULL,  -- epoch ms UTC, formato nativo de Binance
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    PRIMARY KEY (symbol, timeframe, open_time)
)
"""

# Log auditado de órdenes (Sprint 6). La PK es el client_order_id: re-guardar la
# misma orden tras un reintento idempotente actualiza la fila, no la duplica.
ORDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_order_id   TEXT    PRIMARY KEY,
    ts                INTEGER NOT NULL,  -- epoch ms UTC del envío
    symbol            TEXT    NOT NULL,
    side              TEXT    NOT NULL,  -- BUY | SELL
    position_side     TEXT    NOT NULL,  -- LONG | SHORT | BOTH
    type              TEXT    NOT NULL,  -- MARKET | STOP_MARKET | TAKE_PROFIT_MARKET
    quantity          REAL,
    price             REAL,              -- stop_price (protectoras) o avg fill (entrada)
    status            TEXT    NOT NULL,  -- NEW | FILLED | ...
    exchange_order_id TEXT,
    decision_reason   TEXT
)
"""

# Estado de sesión del orquestador (Sprint 7.2). Fila única (id=1): debe
# sobrevivir a reinicios en caliente para que el kill switch y la pérdida diaria
# no se reinicien (y vuelvan a permitir operar) tras una caída.
SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    peak_wallet       REAL    NOT NULL,
    day_start_wallet  REAL    NOT NULL,
    day               TEXT    NOT NULL,  -- fecha UTC ISO (YYYY-MM-DD)
    kill_switch       INTEGER NOT NULL   -- 0 | 1 (latch)
)
"""


class Storage:
    def __init__(self, db_path: str | Path, candles_dir: str | Path):
        self.db_path = Path(db_path)
        self.candles_dir = Path(candles_dir)
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> "Storage":
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(SCHEMA)
        await self._db.execute(ORDERS_SCHEMA)
        await self._db.execute(SESSION_SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ---------- SQLite: velas en vivo ----------

    async def save_candle(self, c: Candle) -> None:
        # INSERT OR REPLACE + PRIMARY KEY compuesta = idempotente: si el
        # websocket re-emite una vela tras una reconexión, no se duplica.
        await self._db.execute(
            "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (c.symbol, c.timeframe, int(c.open_time.timestamp() * 1000),
             c.open, c.high, c.low, c.close, c.volume),
        )
        await self._db.commit()

    async def get_candles(self, symbol: str, timeframe: str,
                          limit: int = 200) -> list[Candle]:
        """Las últimas `limit` velas, en orden cronológico ascendente."""
        cur = await self._db.execute(
            "SELECT open_time, open, high, low, close, volume FROM candles"
            " WHERE symbol = ? AND timeframe = ?"
            " ORDER BY open_time DESC LIMIT ?",
            (symbol, timeframe, limit),
        )
        rows = await cur.fetchall()
        return [
            Candle(
                symbol=symbol, timeframe=timeframe,
                open_time=datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc),
                open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
            )
            for r in reversed(rows)
        ]

    # ---------- SQLite: log auditado de órdenes ----------

    async def save_order(
        self, *, client_order_id: str, ts_ms: int, symbol: str, side: str,
        position_side: str, type: str, quantity: float | None, price: float | None,
        status: str, exchange_order_id: str | None, decision_reason: str,
    ) -> None:
        # INSERT OR REPLACE + PK = idempotente: un reintento con el mismo
        # client_order_id actualiza el estado de la orden, no crea otra fila.
        await self._db.execute(
            "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (client_order_id, ts_ms, symbol, side, position_side, type,
             quantity, price, status, exchange_order_id, decision_reason),
        )
        await self._db.commit()

    async def get_orders(self, limit: int = 200) -> list[dict]:
        """Las últimas `limit` órdenes registradas, de la más reciente a la más antigua."""
        cur = await self._db.execute(
            "SELECT client_order_id, ts, symbol, side, position_side, type,"
            " quantity, price, status, exchange_order_id, decision_reason"
            " FROM orders ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        cols = [c[0] for c in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ---------- SQLite: estado de sesión (sobrevive a reinicios) ----------

    async def save_session_state(self, *, peak_wallet: float, day_start_wallet: float,
                                 day: str, kill_switch: bool) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO session_state VALUES (1, ?, ?, ?, ?)",
            (peak_wallet, day_start_wallet, day, int(kill_switch)),
        )
        await self._db.commit()

    async def load_session_state(self) -> dict | None:
        """Estado de sesión guardado, o None si es el primer arranque."""
        cur = await self._db.execute(
            "SELECT peak_wallet, day_start_wallet, day, kill_switch"
            " FROM session_state WHERE id = 1"
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {"peak_wallet": row[0], "day_start_wallet": row[1],
                "day": row[2], "kill_switch": bool(row[3])}

    # ---------- Parquet: histórico para backtesting ----------

    def parquet_path(self, symbol: str, timeframe: str) -> Path:
        return self.candles_dir / f"{symbol}_{timeframe}.parquet"

    def save_history_parquet(self, df: pd.DataFrame, symbol: str,
                             timeframe: str) -> Path:
        self.candles_dir.mkdir(parents=True, exist_ok=True)
        path = self.parquet_path(symbol, timeframe)
        df.to_parquet(path, index=False)
        return path

    def load_history_parquet(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return pd.read_parquet(self.parquet_path(symbol, timeframe))
