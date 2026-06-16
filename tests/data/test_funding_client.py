"""Tests del cliente de funding/basis — sin red (cliente fake con páginas en cola)."""

import pytest

from src.data.funding_client import (
    fetch_funding_rate_history,
    fetch_premium_index_klines,
    interval_to_ms,
)


class FakeResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class FakeClient:
    """Devuelve páginas en cola; cuando se agotan, devuelve [] (fin de histórico)."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    async def get(self, url, params=None):
        self.calls.append((url, params))
        return FakeResp(self._pages.pop(0) if self._pages else [])


def test_interval_to_ms():
    assert interval_to_ms("1h") == 3_600_000
    assert interval_to_ms("5m") == 300_000
    with pytest.raises(ValueError):
        interval_to_ms("1w")


async def test_funding_parsea_y_ordena():
    page = [
        {"symbol": "BTCUSDT", "fundingTime": 1700000000000, "fundingRate": "0.0001"},
        {"symbol": "BTCUSDT", "fundingTime": 1700028800000, "fundingRate": "-0.0002"},
    ]
    df = await fetch_funding_rate_history(
        "BTCUSDT", days=1, client=FakeClient([page]), end_ms=1700100000000, pause=0)
    assert list(df.columns) == ["funding_time", "funding_rate"]
    assert len(df) == 2
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert df["funding_time"].dt.tz is not None  # tz-aware UTC


async def test_funding_pagina_hasta_agotar():
    # Página llena (límite) seguida de uno corto → para tras la segunda.
    full = [{"symbol": "BTCUSDT", "fundingTime": 1700000000000 + i * 1000,
             "fundingRate": "0.0001"} for i in range(1000)]
    tail = [{"symbol": "BTCUSDT", "fundingTime": 1700000000000 + 1000 * 1000,
             "fundingRate": "0.0001"}]
    client = FakeClient([full, tail])
    df = await fetch_funding_rate_history(
        "BTCUSDT", days=30, client=client, end_ms=1800000000000, pause=0)
    assert len(df) == 1001          # ambas páginas consumidas
    assert df["funding_time"].is_monotonic_increasing


async def test_funding_vacio_devuelve_df_con_columnas():
    df = await fetch_funding_rate_history(
        "BTCUSDT", days=1, client=FakeClient([[]]), end_ms=1700100000000, pause=0)
    assert df.empty and list(df.columns) == ["funding_time", "funding_rate"]


async def test_premium_klines_parsea_ohlc():
    # fila kline: [openTime, open, high, low, close, volume, closeTime, ...]
    page = [
        [1700000000000, "0.0001", "0.0003", "0.0000", "0.0002", "0", 1700003599999],
        [1700003600000, "0.0002", "0.0004", "0.0001", "0.0003", "0", 1700007199999],
    ]
    df = await fetch_premium_index_klines(
        "BTCUSDT", "1h", days=1, client=FakeClient([page]), end_ms=1700100000000, pause=0)
    assert "premium_close" in df.columns and len(df) == 2
    assert df["premium_close"].iloc[1] == pytest.approx(0.0003)
    assert df["open_time"].dt.tz is not None
