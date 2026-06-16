"""Cliente de datos NO-precio de Binance Futuros (público, sin API key).

Tras descartar la TA de solo-precio (sin edge), exploramos señales de
POSICIONAMIENTO del mercado, todas de `fapi.binance.com` (público, $0 — el mismo
patrón que Fear & Greed):

  - Funding rate: lo que los longs pagan a los shorts (o viceversa) cada 8h en un
    perpetuo. Funding muy positivo = longs apalancados pagando por mantenerse →
    posible exceso alcista (hipótesis contraria: precede a correcciones).
  - Basis / premium index: prima del perpetuo sobre el índice spot, (mark-index)/
    index. Prima alta = demanda de apalancamiento largo.

Funciones async puras con cliente httpx INYECTABLE (testeables sin red, igual que
`src/sentiment/fear_greed.py`). Paginan con un cursor de startTime hasta cubrir
`days` días, exactamente como `binance_client.download_history`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pandas as pd

FAPI = "https://fapi.binance.com"

_DAY_MS = 86_400_000
_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}

FUNDING_PAGE = 1000   # máx. de filas por página en /fapi/v1/fundingRate
KLINE_PAGE = 1500     # máx. de velas por página en /fapi/v1/premiumIndexKlines
PAGE_PAUSE = 0.25     # seg entre páginas: gentil con el rate limit


def interval_to_ms(interval: str) -> int:
    """'1h' → 3_600_000. El cursor de paginación de klines avanza en estas unidades."""
    unit = interval[-1]
    if unit not in _UNIT_MS or not interval[:-1].isdigit():
        raise ValueError(f"intervalo no soportado: {interval!r}")
    return int(interval[:-1]) * _UNIT_MS[unit]


async def _paginate(client, path, base_params, *, advance, page_limit, days,
                    end_ms, pause):
    """Bucle de paginación común: acumula filas avanzando un cursor de startTime.

    `advance(last_row)` devuelve el siguiente cursor a partir de la última fila de
    la página. Para si una página viene vacía o más corta que el límite (= se
    alcanzó el presente).
    """
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=30.0)
    end_ms = end_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = end_ms - days * _DAY_MS
    rows: list = []
    try:
        while cursor < end_ms:
            resp = await client.get(FAPI + path, params={**base_params,
                                                         "startTime": cursor,
                                                         "limit": page_limit})
            page = resp.json()
            if not page:
                break
            rows.extend(page)
            cursor = advance(page[-1])
            if len(page) < page_limit:
                break
            if pause:
                await asyncio.sleep(pause)
    finally:
        if own:
            await client.aclose()
    return rows


async def fetch_funding_rate_history(
    symbol: str, days: int, *, client: httpx.AsyncClient | None = None,
    end_ms: int | None = None, pause: float = PAGE_PAUSE,
) -> pd.DataFrame:
    """Histórico de funding rate (cada 8h) de un perpetuo. Sin API key.

    Returns:
        DataFrame [funding_time (datetime UTC), funding_rate (float)], ascendente.
    """
    rows = await _paginate(
        client, "/fapi/v1/fundingRate", {"symbol": symbol},
        advance=lambda r: int(r["fundingTime"]) + 1,
        page_limit=FUNDING_PAGE, days=days, end_ms=end_ms, pause=pause)
    if not rows:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    df = pd.DataFrame({
        "funding_time": pd.to_datetime(
            [int(r["fundingTime"]) for r in rows], unit="ms", utc=True),
        "funding_rate": [float(r["fundingRate"]) for r in rows],
    })
    return df.drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)


async def fetch_premium_index_klines(
    symbol: str, interval: str, days: int, *,
    client: httpx.AsyncClient | None = None, end_ms: int | None = None,
    pause: float = PAGE_PAUSE,
) -> pd.DataFrame:
    """Histórico del premium index (basis) en velas OHLC. Sin API key.

    El premium index ≈ (mark - index)/index: la prima del perpetuo sobre el spot.
    Tomamos su cierre por vela como el valor del basis en ese instante.

    Returns:
        DataFrame [open_time, premium_open, premium_high, premium_low,
        premium_close], ascendente.
    """
    step = interval_to_ms(interval)
    rows = await _paginate(
        client, "/fapi/v1/premiumIndexKlines", {"symbol": symbol, "interval": interval},
        advance=lambda r: int(r[0]) + step,
        page_limit=KLINE_PAGE, days=days, end_ms=end_ms, pause=pause)
    if not rows:
        return pd.DataFrame(columns=["open_time", "premium_open", "premium_high",
                                     "premium_low", "premium_close"])
    df = pd.DataFrame({
        "open_time": pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True),
        "premium_open": [float(r[1]) for r in rows],
        "premium_high": [float(r[2]) for r in rows],
        "premium_low": [float(r[3]) for r in rows],
        "premium_close": [float(r[4]) for r in rows],
    })
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
