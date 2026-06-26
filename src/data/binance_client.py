"""Cliente de datos de Binance: histórico REST paginado + stream websocket.

Patrón de paper trading: los DATOS de mercado vienen siempre de la API
pública de mainnet (precios reales, sin API key), mientras que las ÓRDENES
(Sprint 6) irán a la testnet. La testnet se resetea periódicamente y su
liquidez es ficticia: backtestear con sus datos sería estudiar un mercado
que no existe.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from functools import partial
from typing import Awaitable, Callable

import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from src.core.models import Candle

logger = logging.getLogger(__name__)

PAGE_LIMIT = 1000     # máximo de velas por petición en /api/v3/klines
REQUEST_PAUSE = 0.25  # seg entre páginas: ~8 weight/s, lejos del límite por minuto

# Código de negocio de Binance para "Timestamp for this request was N ms ahead of
# the server time": el reloj local va adelantado respecto al servidor (típico al
# despertar el Mac, antes de que macOS re-sincronice por NTP). No es un umbral de
# trading sino una constante del PROTOCOLO de la API, en la misma línea que los
# códigos -4059/-4084 que ya viven en binance_futures.py. Por eso es un literal con
# nombre y no un parámetro de config.
CLOCK_AHEAD_CODE = -1021

_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
_DAY_MS = 86_400_000


async def resync_clock_offset(client) -> int:
    """Recalibra `client.timestamp_offset` contra el reloj del servidor de Binance.

    python-binance firma cada request con `int(time.time()*1000) + timestamp_offset`.
    Si el reloj local va `Δ` ms adelantado, Binance rechaza con -1021. El offset que
    lo corrige es `server - local = -Δ`: al sumarlo, el timestamp enviado vuelve a
    caer dentro de la ventana de recepción. Devuelve el offset aplicado (para log).
    """
    server = await client.get_server_time()
    server_ms = int(server["serverTime"])
    local_ms = int(time.time() * 1000)
    offset = server_ms - local_ms
    client.timestamp_offset = offset
    return offset


def interval_to_ms(timeframe: str) -> int:
    """'5m' → 300_000. El cursor de paginación avanza en estas unidades."""
    unit = timeframe[-1]
    if unit not in _UNIT_MS or not timeframe[:-1].isdigit():
        raise ValueError(f"timeframe no soportado: {timeframe!r}")
    return int(timeframe[:-1]) * _UNIT_MS[unit]


async def retry_with_backoff(call: Callable[[], Awaitable], *,
                             max_retries: int = 6, base_delay: float = 1.0,
                             client=None, clock_resync_retries: int = 1,
                             sleep=asyncio.sleep):
    """Ejecuta `call` reintentando solo ante HTTP 429 (rate limit) y 418 (ban).

    La espera crece exponencialmente (1s, 2s, 4s, ...) y respeta el header
    Retry-After si Binance lo envía: insistir antes de ese plazo alarga el
    castigo. Cualquier otro error se propaga de inmediato — reintentar un 400
    (parámetros inválidos) solo ocultaría un bug.

    Caso especial `-1021` (reloj local adelantado, típico al despertar el Mac): NO
    se reintenta a ciegas (el reloj seguiría adelantado → fallo infinito que congela
    el bot). Si se inyecta `client`, se recalibra su `timestamp_offset` contra el
    servidor (`resync_clock_offset`) y se reintenta hasta `clock_resync_retries`
    veces. Si persiste tras el reajuste, se propaga. `clock_resync_retries` es un
    contador de mecánica de reintento, hermano de `max_retries`/`base_delay`: vive
    como default de la capa de datos, no es un umbral de trading.
    """
    attempt = 0
    clock_resyncs = 0
    while True:
        try:
            return await call()
        except BinanceAPIException as e:
            if (e.code == CLOCK_AHEAD_CODE and client is not None
                    and clock_resyncs < clock_resync_retries):
                offset = await resync_clock_offset(client)
                logger.warning("reloj desincronizado (-1021), ajustando offset a "
                               "%d ms; reintento %d", offset, clock_resyncs + 1)
                clock_resyncs += 1
                continue
            if e.status_code not in (429, 418) or attempt >= max_retries:
                raise
            headers = getattr(e.response, "headers", None) or {}
            retry_after = float(headers.get("Retry-After", 0))
            delay = max(base_delay * 2 ** attempt, retry_after)
            logger.warning("HTTP %s de Binance; reintento %d en %.1fs",
                           e.status_code, attempt + 1, delay)
            await sleep(delay)
            attempt += 1


# ---------- parsers: formato Binance → contrato Candle ----------

def rest_kline_to_candle(row: list, symbol: str, timeframe: str) -> Candle:
    """Fila de /api/v3/klines: [open_time, o, h, l, c, vol, close_time, ...]"""
    return Candle(
        symbol=symbol, timeframe=timeframe,
        open_time=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
        open=float(row[1]), high=float(row[2]), low=float(row[3]),
        close=float(row[4]), volume=float(row[5]), closed=True,
    )


def ws_msg_to_candle(msg: dict) -> Candle:
    """Mensaje kline del websocket; k.x indica si la vela ya cerró."""
    k = msg["k"]
    return Candle(
        symbol=k["s"], timeframe=k["i"],
        open_time=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
        open=float(k["o"]), high=float(k["h"]), low=float(k["l"]),
        close=float(k["c"]), volume=float(k["v"]), closed=bool(k["x"]),
    )


# ---------- histórico REST con paginación ----------

async def download_history(client, symbol: str, timeframe: str, days: int, *,
                           end_ms: int | None = None, pause: float = REQUEST_PAUSE,
                           page_limit: int = PAGE_LIMIT) -> pd.DataFrame:
    """Descarga `days` días de velas avanzando un cursor de startTime.

    El cursor salta a la vela siguiente a la última recibida; una página
    vacía o parcial significa que llegamos al presente. Se descartan las
    velas cuyo close_time es futuro (la vela aún en formación).
    """
    step = interval_to_ms(timeframe)
    end_ms = end_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = end_ms - days * _DAY_MS
    rows: list[list] = []

    while cursor < end_ms:
        fetch = partial(client.get_klines, symbol=symbol, interval=timeframe,
                        startTime=cursor, limit=page_limit)
        page = await retry_with_backoff(fetch)
        if not page:
            break
        rows.extend(page)
        cursor = page[-1][0] + step
        if len(page) < page_limit:
            break
        if pause:
            await asyncio.sleep(pause)

    rows = [r for r in rows if r[6] <= end_ms]  # fuera la vela sin cerrar
    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low",
                                     "close", "volume"])
    df = pd.DataFrame(
        [[int(r[0])] + [float(x) for x in r[1:6]] for r in rows],
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return (df.drop_duplicates("open_time")
              .sort_values("open_time")
              .reset_index(drop=True))


# ---------- stream websocket con reconexión ----------

async def stream_candles(client: AsyncClient, symbol: str, timeframe: str,
                         on_candle: Callable[[Candle], Awaitable[None]], *,
                         closed_only: bool = True,
                         reconnect_delay: float = 5.0) -> None:
    """Llama `on_candle(candle)` por cada vela del stream, reconectando solo.

    Aquí solo se garantiza que el flujo de datos se recupera; qué hacer con
    las posiciones durante el hueco lo decide el circuit breaker del Risk
    Manager (Sprint 5).
    """
    bsm = BinanceSocketManager(client)
    while True:
        try:
            async with bsm.kline_socket(symbol, interval=timeframe) as stream:
                logger.info("stream conectado: %s %s", symbol, timeframe)
                while True:
                    msg = await stream.recv()
                    if not isinstance(msg, dict) or msg.get("e") == "error":
                        raise ConnectionError(f"mensaje de error del stream: {msg}")
                    candle = ws_msg_to_candle(msg)
                    if closed_only and not candle.closed:
                        continue
                    await on_candle(candle)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("stream %s %s caído; reconexión en %.0fs",
                           symbol, timeframe, reconnect_delay, exc_info=True)
            await asyncio.sleep(reconnect_delay)


async def stream_mark_price(client: AsyncClient, symbol: str,
                            on_tick: Callable[[str, datetime, float], None], *,
                            reconnect_delay: float = 5.0) -> None:
    """Stream `<symbol>@markPrice@1s` (Futuros USD-M): plano de datos del Fast Path.

    Llama `on_tick(symbol, ts, mark_price)` por cada actualización (≈1/seg). El
    callback es SÍNCRONO a propósito: el push al deque del orquestador no tiene
    `await`, así corre atómico frente a la lectura de `_price_impulse_bps` (ambos
    en el mismo event loop). Reconecta solo, igual que `stream_candles`; la capa
    externa (`_supervise`) lo reinicia si la corrutina cae por completo.

    Mensaje markPriceUpdate: `p` = mark price (str), `E` = event time (ms epoch).
    """
    bsm = BinanceSocketManager(client)
    while True:
        try:
            async with bsm.symbol_mark_price_socket(symbol, fast=True) as stream:
                logger.info("stream markPrice conectado: %s", symbol)
                while True:
                    msg = await stream.recv()
                    if not isinstance(msg, dict) or msg.get("e") == "error":
                        raise ConnectionError(f"mensaje de error del stream: {msg}")
                    ts = datetime.fromtimestamp(msg["E"] / 1000.0, tz=timezone.utc)
                    on_tick(symbol, ts, float(msg["p"]))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("stream markPrice %s caído; reconexión en %.0fs",
                           symbol, reconnect_delay, exc_info=True)
            await asyncio.sleep(reconnect_delay)
