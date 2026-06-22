"""Lectura READ-ONLY de la SQLite del bot y ensamblado del snapshot del dashboard.

Diseño deliberado: el dashboard es un PROCESO APARTE que abre la base en modo
`ro` (URI `file:...?mode=ro`), así ni un bug podría escribir. No usa la clase
`Storage` (async, dueña de la escritura) sino `sqlite3` síncrono: el dashboard no
tiene event loop de trading, así que el I/O bloqueante aquí es inofensivo (la
regla async/aiosqlite protege el pipeline EN VIVO, no este visor).

`build_snapshot` es casi puro: recibe `settings`, el reloj (`now`) y `testnet`
inyectables, y devuelve un dict JSON-serializable. Toda la lógica derivada
(drawdown, PnL del día, liveness) se calcula aquí para que el frontend sea tonto.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _interval_seconds(timeframe: str) -> int:
    """Segundos de un timeframe estilo Binance ('5m','1h','1d'). Parser local para
    NO acoplar el dashboard al SDK del exchange (se mantiene stdlib-only)."""
    unit = timeframe[-1].lower()
    n = int(timeframe[:-1])
    return n * {"m": 60, "h": 3600, "d": 86400}[unit]


def _connect(db_path: str | Path) -> sqlite3.Connection | None:
    """Conexión READ-ONLY, o None si la BD aún no existe (bot nunca arrancado)."""
    p = Path(db_path)
    if not p.exists():
        return None
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """Ejecuta una consulta y devuelve filas como dicts. Tolera tablas ausentes
    (una BD recién creada puede no tener todas) → lista vacía."""
    try:
        cur = conn.execute(sql, params)
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in cur.fetchall()]


def _mode_label(event_enabled: bool, sentiment_enabled: bool) -> str:
    if event_enabled and sentiment_enabled:
        return "Híbrido (Opción 2): noticias + régimen"
    if sentiment_enabled:
        return "Slow Path: noticias confirmadas por régimen"
    if event_enabled:
        return "Fast Path: originación por shocks"
    return "Gates OFF — sin originación por noticias"


def build_snapshot(
    settings, *, now: datetime | None = None, testnet: bool | None = None,
) -> dict:
    """Ensambla TODO lo que el frontend necesita en una sola lectura.

    Args:
        settings: configuración cargada (db_path, dashboard.*, market.*, gates).
        now:      reloj inyectable (determinista en tests). Por defecto, ahora UTC.
        testnet:  bandera informativa para el badge (no se consultan secretos aquí).
    """
    now = now or datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    d = settings.dashboard

    meta = {
        "now_ms": now_ms,
        "symbols": list(settings.market.symbols),
        "timeframe": settings.market.timeframe,
        "htf_timeframe": settings.market.htf_timeframe,
        "mode": _mode_label(settings.event.enabled, settings.sentiment.enabled),
        "event_enabled": settings.event.enabled,
        "sentiment_enabled": settings.sentiment.enabled,
        "testnet": testnet,
        "refresh_seconds": d.refresh_seconds,
        "last_update_ms": None,
        "age_seconds": None,
        "stale": True,
        "has_data": False,
    }
    empty = {
        "meta": meta,
        "kpis": {"wallet": None, "equity": None, "upnl": None, "peak_wallet": None,
                 "day_start_wallet": None, "drawdown_pct": None, "day_pnl": None,
                 "day_pnl_pct": None, "kill_switch": False, "open_positions": 0},
        "equity": [], "positions": [], "decisions": [], "orders": [], "news": [],
    }

    conn = _connect(settings.storage.db_path)
    if conn is None:
        return empty
    try:
        equity = _rows(
            conn,
            "SELECT ts, wallet, equity, upnl FROM equity_snapshots"
            " ORDER BY ts DESC LIMIT ?", (d.equity_points,),
        )[::-1]  # a ascendente para la curva
        decisions = _rows(
            conn,
            "SELECT ts, symbol, action, reason, quant_score, sentiment_score,"
            " size_factor, source FROM decisions ORDER BY ts DESC LIMIT ?",
            (d.decisions_rows,),
        )
        orders = _rows(
            conn,
            "SELECT client_order_id, ts, symbol, side, position_side, type,"
            " quantity, price, status, decision_reason FROM orders"
            " ORDER BY ts DESC LIMIT ?", (d.orders_rows,),
        )
        news = _rows(
            conn,
            "SELECT n.id AS news_id, n.ts AS ts, n.title AS title, n.source AS source,"
            " n.url AS url, s.score AS score, s.confidence AS confidence,"
            " s.high_impact AS high_impact, s.symbol_scope AS symbol_scope,"
            " s.rationale AS rationale"
            " FROM news n LEFT JOIN sentiment_scores s ON s.news_id = n.id"
            " ORDER BY n.ts DESC LIMIT ?", (d.news_rows,),
        )
        session = _rows(
            conn, "SELECT peak_wallet, day_start_wallet, day, kill_switch"
            " FROM session_state WHERE id = 1",
        )
        latest = _rows(
            conn, "SELECT ts, wallet, equity, upnl, positions FROM equity_snapshots"
            " ORDER BY ts DESC LIMIT 1",
        )
    finally:
        conn.close()

    # Normaliza tipos del JSON guardado.
    for n in news:
        n["high_impact"] = bool(n["high_impact"]) if n["high_impact"] is not None else None
        if n.get("symbol_scope"):
            try:
                n["symbol_scope"] = json.loads(n["symbol_scope"])
            except (TypeError, json.JSONDecodeError):
                n["symbol_scope"] = []

    sess = session[0] if session else None
    kill = bool(sess["kill_switch"]) if sess else False
    peak = sess["peak_wallet"] if sess else None
    day_start = sess["day_start_wallet"] if sess else None

    if latest:
        last = latest[0]
        meta["last_update_ms"] = last["ts"]
        meta["age_seconds"] = max(0.0, (now_ms - last["ts"]) / 1000.0)
        stale_limit = d.stale_after_intervals * _interval_seconds(settings.market.timeframe)
        meta["stale"] = meta["age_seconds"] > stale_limit
        meta["has_data"] = True

        wallet, eq, upnl = last["wallet"], last["equity"], last["upnl"]
        positions = json.loads(last["positions"]) if last["positions"] else []
        drawdown = ((peak - eq) / peak * 100.0) if (peak and peak > 0) else 0.0
        drawdown = max(0.0, drawdown)
        day_pnl = (eq - day_start) if day_start is not None else None
        day_pnl_pct = (day_pnl / day_start * 100.0) if (day_start and day_start > 0) else None
        kpis = {
            "wallet": wallet, "equity": eq, "upnl": upnl, "peak_wallet": peak,
            "day_start_wallet": day_start, "drawdown_pct": drawdown,
            "day_pnl": day_pnl, "day_pnl_pct": day_pnl_pct,
            "kill_switch": kill, "open_positions": len(positions),
        }
    else:
        positions = []
        kpis = dict(empty["kpis"], kill_switch=kill, peak_wallet=peak,
                    day_start_wallet=day_start)

    return {
        "meta": meta, "kpis": kpis, "equity": equity, "positions": positions,
        "decisions": decisions, "orders": orders, "news": news,
    }
