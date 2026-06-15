"""Tests de Storage: WAL activo, roundtrips e idempotencia."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.core.models import Candle, NewsItem, SentimentScore
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


async def test_orders_roundtrip_y_orden_temporal(storage):
    for i, status in enumerate(["FILLED", "NEW", "NEW"]):
        await storage.save_order(
            client_order_id=f"c{i}", ts_ms=1000 + i, symbol="BTCUSDT",
            side="BUY", position_side="LONG", type="MARKET", quantity=1.0,
            price=100.0, status=status, exchange_order_id=str(i), decision_reason="t")
    out = await storage.get_orders()
    assert len(out) == 3
    assert out[0]["client_order_id"] == "c2"  # el más reciente primero (ts DESC)
    assert out[0]["position_side"] == "LONG"


async def test_order_idempotente_por_client_order_id(storage):
    await storage.save_order(
        client_order_id="x", ts_ms=1, symbol="BTCUSDT", side="BUY",
        position_side="LONG", type="MARKET", quantity=1.0, price=100.0,
        status="NEW", exchange_order_id="1", decision_reason="t")
    # Mismo client_order_id (reintento idempotente): actualiza, no duplica.
    await storage.save_order(
        client_order_id="x", ts_ms=1, symbol="BTCUSDT", side="BUY",
        position_side="LONG", type="MARKET", quantity=1.0, price=100.0,
        status="FILLED", exchange_order_id="1", decision_reason="t")
    out = await storage.get_orders()
    assert len(out) == 1 and out[0]["status"] == "FILLED"


async def test_news_roundtrip_y_rango_temporal(storage):
    for i in range(3):
        await storage.save_news(NewsItem(
            id=f"n{i}", title=f"Bitcoin news {i}", source="cp", url=f"u{i}",
            published_at=NOW + timedelta(hours=i), summary=""))
    # rango: solo las dos primeras
    until = int((NOW + timedelta(hours=1)).timestamp() * 1000)
    out = await storage.get_news(until_ms=until)
    assert [n.id for n in out] == ["n0", "n1"]          # orden cronológico
    assert out[0].published_at.tzinfo is not None


async def test_news_idempotente_por_hash(storage):
    item = NewsItem(id="x", title="t", source="cp", url="u", published_at=NOW, summary="")
    await storage.save_news(item)
    await storage.save_news(item)  # misma noticia desde otro feed
    assert len(await storage.get_news()) == 1


async def test_sentiment_scores_roundtrip(storage):
    sc = SentimentScore(news_id="n1", symbol_scope=["BTC", "ETH"], score=-0.8,
                        confidence=0.9, high_impact=True, rationale="hack",
                        analyzed_at=NOW)
    ts = int(NOW.timestamp() * 1000)
    await storage.save_sentiment_score(sc, ts_ms=ts)
    out = await storage.get_sentiment_scores()
    assert len(out) == 1
    assert out[0]["score"] == -0.8 and out[0]["symbol_scope"] == ["BTC", "ETH"]
    assert out[0]["high_impact"] is True and out[0]["ts"] == ts


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
