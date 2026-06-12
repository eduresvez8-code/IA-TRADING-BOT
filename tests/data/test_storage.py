"""Tests de Storage: WAL activo, roundtrips e idempotencia."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.core.models import Candle
from src.data.storage import Storage

NOW = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)


def make_candle(minutes_offset: int = 0, close: float = 105.0) -> Candle:
    return Candle(
        symbol="BTCUSDT", timeframe="5m",
        open_time=NOW + timedelta(minutes=minutes_offset),
        open=100.0, high=110.0, low=95.0, close=close, volume=1.5,
    )


@pytest.fixture
async def storage(tmp_path):
    s = await Storage(tmp_path / "test.db", tmp_path / "candles").init()
    yield s
    await s.close()


async def test_wal_activado(storage):
    # Directriz del plan: WAL es obligatorio para concurrencia entre módulos.
    cur = await storage._db.execute("PRAGMA journal_mode")
    (mode,) = await cur.fetchone()
    assert mode.lower() == "wal"


async def test_roundtrip_velas_en_orden_cronologico(storage):
    for i in (10, 0, 5):  # guardadas en desorden a propósito
        await storage.save_candle(make_candle(minutes_offset=i))
    out = await storage.get_candles("BTCUSDT", "5m")
    assert len(out) == 3
    assert [c.open_time for c in out] == sorted(c.open_time for c in out)
    assert out[0].open_time.tzinfo is not None


async def test_guardar_misma_vela_dos_veces_no_duplica(storage):
    await storage.save_candle(make_candle(close=105.0))
    await storage.save_candle(make_candle(close=106.0))  # re-emisión corregida
    out = await storage.get_candles("BTCUSDT", "5m")
    assert len(out) == 1
    assert out[0].close == 106.0  # la última versión gana


async def test_limit_devuelve_las_mas_recientes(storage):
    for i in range(5):
        await storage.save_candle(make_candle(minutes_offset=i * 5))
    out = await storage.get_candles("BTCUSDT", "5m", limit=2)
    assert len(out) == 2
    assert out[-1].open_time == NOW + timedelta(minutes=20)


def test_parquet_roundtrip(tmp_path):
    s = Storage(tmp_path / "test.db", tmp_path / "candles")
    df = pd.DataFrame({
        "open_time": pd.to_datetime([NOW, NOW + timedelta(minutes=5)], utc=True),
        "open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
        "close": [1.2, 2.2], "volume": [10.0, 20.0],
    })
    s.save_history_parquet(df, "BTCUSDT", "5m")
    loaded = s.load_history_parquet("BTCUSDT", "5m")
    pd.testing.assert_frame_equal(loaded, df)
