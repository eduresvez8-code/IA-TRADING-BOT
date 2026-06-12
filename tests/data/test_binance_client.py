"""Tests del cliente Binance: backoff, paginación y parsers — sin red.

La lógica de reintentos y cursor se testea inyectando fakes: un `sleep` que
solo registra las esperas y un cliente que devuelve páginas preparadas.
"""

from types import SimpleNamespace

import pytest
from binance.exceptions import BinanceAPIException

from src.data.binance_client import (
    download_history,
    interval_to_ms,
    rest_kline_to_candle,
    retry_with_backoff,
    ws_msg_to_candle,
)

STEP = 60_000  # 1m en ms


def make_api_exc(status_code: int, retry_after: float | None = None):
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    response = SimpleNamespace(headers=headers, text="")
    return BinanceAPIException(response, status_code,
                               '{"code": -1003, "msg": "too many requests"}')


# ---------- retry_with_backoff ----------

async def test_backoff_exponencial_hasta_exito():
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise make_api_exc(429)
        return "ok"

    assert await retry_with_backoff(flaky, sleep=fake_sleep) == "ok"
    assert sleeps == [1.0, 2.0, 4.0]  # 1 * 2^intento


async def test_backoff_respeta_retry_after():
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise make_api_exc(418, retry_after=30)
        return "ok"

    await retry_with_backoff(flaky, sleep=fake_sleep)
    assert sleeps == [30.0]  # el header manda sobre el backoff calculado


async def test_errores_no_transitorios_se_propagan_sin_reintento():
    attempts = 0

    async def bad_request():
        nonlocal attempts
        attempts += 1
        raise make_api_exc(400)

    with pytest.raises(BinanceAPIException):
        await retry_with_backoff(bad_request, sleep=lambda s: None)
    assert attempts == 1


async def test_se_rinde_tras_max_retries():
    async def fake_sleep(s):
        pass

    async def always_limited():
        raise make_api_exc(429)

    with pytest.raises(BinanceAPIException):
        await retry_with_backoff(always_limited, max_retries=2, sleep=fake_sleep)


# ---------- download_history ----------

def kline_row(open_ms: int, close_ms: int) -> list:
    # Formato de /api/v3/klines (los precios llegan como strings)
    return [open_ms, "100.0", "110.0", "90.0", "105.0", "12.5", close_ms]


class FakeClient:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    async def get_klines(self, **params):
        self.calls.append(params)
        return self.pages.pop(0) if self.pages else []


async def test_paginacion_avanza_cursor_y_filtra_vela_abierta():
    end_ms = 86_400_000  # días=1 → el cursor inicial queda en 0
    page1 = [kline_row(i * STEP, (i + 1) * STEP - 1) for i in range(3)]
    page2 = [kline_row(3 * STEP, 4 * STEP - 1),
             kline_row(4 * STEP, 5 * STEP - 1),
             kline_row(5 * STEP, end_ms + 10_000)]  # vela aún sin cerrar
    fake = FakeClient([page1, page2])

    df = await download_history(fake, "BTCUSDT", "1m", days=1,
                                end_ms=end_ms, pause=0, page_limit=3)

    assert fake.calls[0]["startTime"] == 0
    # el cursor salta a la vela siguiente a la última de la página 1
    assert fake.calls[1]["startTime"] == page1[-1][0] + STEP
    assert len(fake.calls) == 3  # la página vacía final corta el bucle
    assert len(df) == 5  # la vela sin cerrar quedó fuera
    assert df["open_time"].is_monotonic_increasing
    assert df["open_time"].dt.tz is not None


async def test_paginas_solapadas_no_duplican():
    end_ms = 86_400_000
    page1 = [kline_row(0, STEP - 1), kline_row(STEP, 2 * STEP - 1)]
    fake = FakeClient([page1, page1])  # Binance re-envía lo mismo

    df = await download_history(fake, "BTCUSDT", "1m", days=1,
                                end_ms=end_ms, pause=0, page_limit=2)
    assert len(df) == 2


# ---------- parsers ----------

def test_interval_to_ms():
    assert interval_to_ms("5m") == 300_000
    assert interval_to_ms("1h") == 3_600_000
    with pytest.raises(ValueError):
        interval_to_ms("5x")


def test_rest_kline_a_candle():
    c = rest_kline_to_candle(kline_row(STEP, 2 * STEP - 1), "BTCUSDT", "1m")
    assert (c.open, c.high, c.low, c.close) == (100.0, 110.0, 90.0, 105.0)
    assert c.open_time.tzinfo is not None
    assert c.closed is True


def test_ws_msg_a_candle():
    msg = {"e": "kline", "k": {
        "s": "BTCUSDT", "i": "1m", "t": STEP,
        "o": "100.0", "h": "110.0", "l": "90.0", "c": "105.0",
        "v": "12.5", "x": False,
    }}
    c = ws_msg_to_candle(msg)
    assert c.closed is False  # vela en formación: el quant engine la ignora
    assert c.volume == 12.5
