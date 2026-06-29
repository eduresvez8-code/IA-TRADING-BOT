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

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pandas as pd

from src.core.models import Candle, NewsItem

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

# Corpus histórico de noticias y sus scores (Sprint C). La PK (hash de URL) hace
# idempotente acumular el corpus en ejecuciones sucesivas del free tier.
NEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id        TEXT    PRIMARY KEY,
    ts        INTEGER NOT NULL,  -- published_at en epoch ms UTC (alineación a velas)
    title     TEXT    NOT NULL,
    source    TEXT,
    url       TEXT,
    summary   TEXT
)
"""

SCORES_SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment_scores (
    news_id      TEXT    PRIMARY KEY,
    ts           INTEGER NOT NULL,  -- = published_at de la noticia (NO analyzed_at)
    score        REAL    NOT NULL,
    confidence   REAL    NOT NULL,
    high_impact  INTEGER NOT NULL,
    symbol_scope TEXT    NOT NULL,  -- JSON, ej. ["BTC","ETH"] o ["*"]
    rationale    TEXT
)
"""

# Serie temporal de equity (dashboard). Una fila por ciclo del orquestador: sin
# esto NO hay curva de capital (session_state es una sola fila). La PK por ts hace
# idempotente re-escribir el mismo instante. `positions` es un JSON con la foto de
# las piernas abiertas (símbolo/lado/qty/entrada/uPnL) en ese ciclo.
EQUITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts        INTEGER PRIMARY KEY,  -- epoch ms UTC del ciclo
    wallet    REAL    NOT NULL,     -- colateral SIN PnL no realizado
    equity    REAL    NOT NULL,     -- wallet + uPnL (margin balance)
    upnl      REAL    NOT NULL,     -- PnL no realizado total
    positions TEXT    NOT NULL      -- JSON: [{symbol, side, qty, entry_price, upnl}]
)
"""

# Log de decisiones de la confluencia (dashboard "¿por qué (no) operó?"). Persiste
# TODAS las decisiones, incluidos los HOLD, que antes no dejaban rastro. PK
# (symbol, ts): una decisión por símbolo por instante. `source`: 'slow' (vela) o
# 'event' (Fast Path).
DECISIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    ts              INTEGER NOT NULL,  -- epoch ms UTC de la decisión
    symbol          TEXT    NOT NULL,
    action          TEXT    NOT NULL,  -- LONG | SHORT | HOLD
    reason          TEXT    NOT NULL,  -- regla de la matriz (auditoría)
    quant_score     REAL    NOT NULL,  -- régimen (Opción 2)
    sentiment_score REAL    NOT NULL,
    size_factor     REAL    NOT NULL,
    source          TEXT    NOT NULL,  -- slow | event
    PRIMARY KEY (symbol, ts)
)
"""


# PnL REALIZADO acumulado por símbolo (panel "P&L por símbolo" del dashboard). Una
# fila por símbolo, sobrescrita cada sondeo con el total de la SESIÓN en vivo (suma
# del income history REALIZED_PNL desde que arrancó el bot). `updated_ms` data la
# última actualización para diagnóstico.
REALIZED_SCHEMA = """
CREATE TABLE IF NOT EXISTS realized_pnl (
    symbol     TEXT    PRIMARY KEY,
    realized   REAL    NOT NULL,      -- PnL realizado acumulado de la sesión (USDT)
    updated_ms INTEGER NOT NULL       -- epoch ms UTC de la última actualización
)
"""

# Latido del lazo en vivo: fila única que el bot reescribe cada pocos segundos
# mientras su event loop está VIVO. El dashboard lo lee para decir online/offline con
# precisión de segundos — más fino que la antigüedad del último equity_snapshot (que
# solo se escribe al cerrar vela, cada 5m). Si el proceso muere o se congela (dormir
# el Mac), el latido deja de avanzar → offline.
HEARTBEAT_SCHEMA = """
CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    ts INTEGER NOT NULL                -- epoch ms UTC del último latido
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
        await self._db.execute(NEWS_SCHEMA)
        await self._db.execute(SCORES_SCHEMA)
        await self._db.execute(EQUITY_SCHEMA)
        await self._db.execute(DECISIONS_SCHEMA)
        await self._db.execute(REALIZED_SCHEMA)
        await self._db.execute(HEARTBEAT_SCHEMA)
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

    # ---------- SQLite: corpus histórico de noticias y scores ----------

    async def save_news(self, item: NewsItem) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO news VALUES (?, ?, ?, ?, ?, ?)",
            (item.id, int(item.published_at.timestamp() * 1000), item.title,
             item.source, item.url, item.summary),
        )
        await self._db.commit()

    async def get_news(self, *, since_ms: int | None = None,
                       until_ms: int | None = None) -> list[NewsItem]:
        """Noticias en [since_ms, until_ms], orden cronológico ascendente."""
        clauses, params = [], []
        if since_ms is not None:
            clauses.append("ts >= ?"); params.append(since_ms)
        if until_ms is not None:
            clauses.append("ts <= ?"); params.append(until_ms)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = await self._db.execute(
            f"SELECT id, ts, title, source, url, summary FROM news{where}"
            " ORDER BY ts ASC", params,
        )
        rows = await cur.fetchall()
        return [
            NewsItem(id=r[0],
                     published_at=datetime.fromtimestamp(r[1] / 1000, tz=timezone.utc),
                     title=r[2], source=r[3] or "", url=r[4] or "", summary=r[5] or "")
            for r in rows
        ]

    async def save_sentiment_score(self, score, *, ts_ms: int) -> None:
        # ts_ms es el published_at de la noticia (no analyzed_at): así el score
        # se alinea al instante en que la información estuvo disponible.
        await self._db.execute(
            "INSERT OR REPLACE INTO sentiment_scores VALUES (?, ?, ?, ?, ?, ?, ?)",
            (score.news_id, ts_ms, score.score, score.confidence,
             int(score.high_impact), json.dumps(score.symbol_scope), score.rationale),
        )
        await self._db.commit()

    async def get_sentiment_scores(self, *, since_ms: int | None = None,
                                   until_ms: int | None = None) -> list[dict]:
        """Scores en [since_ms, until_ms], orden ascendente. Cada uno como dict."""
        clauses, params = [], []
        if since_ms is not None:
            clauses.append("ts >= ?"); params.append(since_ms)
        if until_ms is not None:
            clauses.append("ts <= ?"); params.append(until_ms)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = await self._db.execute(
            "SELECT news_id, ts, score, confidence, high_impact, symbol_scope, rationale"
            f" FROM sentiment_scores{where} ORDER BY ts ASC", params,
        )
        rows = await cur.fetchall()
        return [
            {"news_id": r[0], "ts": r[1], "score": r[2], "confidence": r[3],
             "high_impact": bool(r[4]), "symbol_scope": json.loads(r[5]), "rationale": r[6]}
            for r in rows
        ]

    # ---------- SQLite: serie de equity (curva de capital) ----------

    async def save_equity_snapshot(
        self, *, ts_ms: int, wallet: float, equity: float, upnl: float,
        positions: list[dict],
    ) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO equity_snapshots VALUES (?, ?, ?, ?, ?)",
            (ts_ms, wallet, equity, upnl, json.dumps(positions)),
        )
        await self._db.commit()

    async def get_equity_snapshots(self, limit: int = 500) -> list[dict]:
        """Los últimos `limit` snapshots, en orden cronológico ASCENDENTE (curva)."""
        cur = await self._db.execute(
            "SELECT ts, wallet, equity, upnl, positions FROM equity_snapshots"
            " ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [
            {"ts": r[0], "wallet": r[1], "equity": r[2], "upnl": r[3],
             "positions": json.loads(r[4])}
            for r in reversed(rows)
        ]

    # ---------- SQLite: log de decisiones (¿por qué (no) operó?) ----------

    async def save_decision(
        self, *, ts_ms: int, symbol: str, action: str, reason: str,
        quant_score: float, sentiment_score: float, size_factor: float, source: str,
    ) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts_ms, symbol, action, reason, quant_score, sentiment_score,
             size_factor, source),
        )
        await self._db.commit()

    async def get_decisions(self, limit: int = 100) -> list[dict]:
        """Las últimas `limit` decisiones, de la más reciente a la más antigua."""
        cur = await self._db.execute(
            "SELECT ts, symbol, action, reason, quant_score, sentiment_score,"
            " size_factor, source FROM decisions ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        cols = [c[0] for c in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ---------- SQLite: PnL realizado por símbolo (panel del dashboard) ----------

    async def save_realized_pnl(self, *, realized: dict[str, float], ts_ms: int) -> None:
        """Sobrescribe (upsert) el PnL realizado acumulado de cada símbolo dado."""
        for symbol, value in realized.items():
            await self._db.execute(
                "INSERT OR REPLACE INTO realized_pnl VALUES (?, ?, ?)",
                (symbol, float(value), ts_ms),
            )
        await self._db.commit()

    async def save_heartbeat(self, ts_ms: int) -> None:
        """Reescribe el latido (fila única) con el epoch ms actual."""
        await self._db.execute(
            "INSERT OR REPLACE INTO heartbeat (id, ts) VALUES (1, ?)", (ts_ms,))
        await self._db.commit()

    async def get_realized_pnl(self) -> list[dict]:
        """PnL realizado por símbolo (todas las filas), de mayor a menor."""
        cur = await self._db.execute(
            "SELECT symbol, realized, updated_ms FROM realized_pnl"
            " ORDER BY realized DESC",
        )
        cols = [c[0] for c in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

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
