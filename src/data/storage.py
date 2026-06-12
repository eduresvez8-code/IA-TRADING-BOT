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
