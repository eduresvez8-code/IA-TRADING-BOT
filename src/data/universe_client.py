"""Cliente del universo de perps de Binance Futuros (público, sin API key).

Para el momentum cross-sectional necesitamos el histórico diario de MUCHOS
activos (no uno). Este módulo lista los perpetuos USDT en TRADING y baja sus
velas diarias desde fapi.binance.com — el mismo patrón $0 que el resto.

Funciones async puras con cliente httpx inyectable (testeable sin red).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd

FAPI = "https://fapi.binance.com"
_DAY_MS = 86_400_000
KLINE_PAGE = 1500  # máximo de velas por página; 1100 días diarios caben en una


async def fetch_perp_symbols(client: httpx.AsyncClient | None = None) -> list[str]:
    """Lista de perpetuos USDT en estado TRADING (el universo del ranking)."""
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        info = (await client.get(FAPI + "/fapi/v1/exchangeInfo")).json()
    finally:
        if own:
            await client.aclose()
    return [
        s["symbol"] for s in info.get("symbols", [])
        if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
    ]


async def fetch_daily_klines(
    symbol: str, days: int, *, client: httpx.AsyncClient | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """Velas diarias de `symbol` para los últimos `days` días.

    `days ≤ 1500` cabe en una sola página (no paginamos). Devuelve
    [open_time (datetime UTC), close (float), quote_volume (float)].
    """
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=30.0)
    end_ms = end_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * _DAY_MS
    try:
        resp = await client.get(FAPI + "/fapi/v1/klines", params={
            "symbol": symbol, "interval": "1d",
            "startTime": start_ms, "limit": KLINE_PAGE})
        rows = resp.json()
    finally:
        if own:
            await client.aclose()
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["open_time", "close", "quote_volume"])
    # fila kline: [openTime, o, h, l, c, vol, closeTime, quoteVol, ...]
    df = pd.DataFrame({
        "open_time": pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True),
        "close": [float(r[4]) for r in rows],
        "quote_volume": [float(r[7]) for r in rows],
    })
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
