"""Tests del cliente del universo — sin red (cliente fake)."""

import pytest

from src.data.universe_client import fetch_daily_klines, fetch_perp_symbols


class FakeResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, data):
        self._data = data
        self.calls = []

    async def get(self, url, params=None):
        self.calls.append((url, params))
        return FakeResp(self._data)


async def test_perp_symbols_filtra_usdt_perpetuos_trading():
    info = {"symbols": [
        {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "quoteAsset": "USDT"},
        {"symbol": "ETHUSDC", "contractType": "PERPETUAL", "status": "TRADING", "quoteAsset": "USDC"},  # no USDT
        {"symbol": "ADAUSDT", "contractType": "PERPETUAL", "status": "SETTLING", "quoteAsset": "USDT"},  # no TRADING
        {"symbol": "BTCUSDT_240927", "contractType": "CURRENT_QUARTER", "status": "TRADING", "quoteAsset": "USDT"},  # no perp
        {"symbol": "SOLUSDT", "contractType": "PERPETUAL", "status": "TRADING", "quoteAsset": "USDT"},
    ]}
    syms = await fetch_perp_symbols(client=FakeClient(info))
    assert syms == ["BTCUSDT", "SOLUSDT"]


async def test_daily_klines_parsea_close_y_quote_volume():
    # fila: [openTime, o, h, l, c, vol, closeTime, quoteVol, ...]
    rows = [
        [1700000000000, "1", "2", "0.5", "1.5", "100", 1700086399999, "150"],
        [1700086400000, "1.5", "3", "1", "2.0", "200", 1700172799999, "400"],
    ]
    df = await fetch_daily_klines("BTCUSDT", days=2, client=FakeClient(rows),
                                  end_ms=1700200000000)
    assert list(df.columns) == ["open_time", "close", "quote_volume"]
    assert df["close"].iloc[1] == pytest.approx(2.0)
    assert df["quote_volume"].iloc[1] == pytest.approx(400.0)
    assert df["open_time"].dt.tz is not None


async def test_daily_klines_vacio_devuelve_columnas():
    df = await fetch_daily_klines("ZZZUSDT", days=2, client=FakeClient([]),
                                  end_ms=1700200000000)
    assert df.empty and list(df.columns) == ["open_time", "close", "quote_volume"]
