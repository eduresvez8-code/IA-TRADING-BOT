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


async def test_equity_snapshots_roundtrip_ascendente(storage):
    # Guardadas en desorden; la curva debe salir cronológica ASCENDENTE.
    for ts in (3000, 1000, 2000):
        await storage.save_equity_snapshot(
            ts_ms=ts, wallet=1000.0 + ts, equity=1000.0 + ts, upnl=0.0,
            positions=[{"symbol": "BTCUSDT", "side": "LONG", "qty": 0.1,
                        "entry_price": 50000.0, "upnl": 5.0}])
    out = await storage.get_equity_snapshots()
    assert [r["ts"] for r in out] == [1000, 2000, 3000]
    assert out[0]["positions"][0]["symbol"] == "BTCUSDT"  # JSON deserializado


async def test_equity_snapshot_idempotente_por_ts(storage):
    await storage.save_equity_snapshot(ts_ms=1, wallet=100.0, equity=100.0,
                                       upnl=0.0, positions=[])
    await storage.save_equity_snapshot(ts_ms=1, wallet=200.0, equity=205.0,
                                       upnl=5.0, positions=[])  # mismo ts → reemplaza
    out = await storage.get_equity_snapshots()
    assert len(out) == 1 and out[0]["wallet"] == 200.0


async def test_equity_limit_devuelve_los_mas_recientes(storage):
    for i in range(5):
        await storage.save_equity_snapshot(ts_ms=i, wallet=float(i), equity=float(i),
                                           upnl=0.0, positions=[])
    out = await storage.get_equity_snapshots(limit=2)
    assert [r["ts"] for r in out] == [3, 4]  # los 2 últimos, ascendente


async def test_decisions_roundtrip_y_orden_temporal(storage):
    for i, act in enumerate(["HOLD", "LONG", "HOLD"]):
        await storage.save_decision(
            ts_ms=1000 + i, symbol="BTCUSDT", action=act, reason="no_news_origination",
            quant_score=0.3, sentiment_score=0.0, size_factor=0.0, source="slow")
    out = await storage.get_decisions()
    assert len(out) == 3
    assert out[0]["ts"] == 1002 and out[0]["action"] == "HOLD"  # más reciente primero
    assert out[0]["source"] == "slow"


async def test_decisions_idempotente_por_symbol_ts(storage):
    await storage.save_decision(ts_ms=1, symbol="BTCUSDT", action="HOLD",
                                reason="r1", quant_score=0.0, sentiment_score=0.0,
                                size_factor=0.0, source="slow")
    await storage.save_decision(ts_ms=1, symbol="BTCUSDT", action="LONG",
                                reason="regime_confirms", quant_score=0.8,
                                sentiment_score=0.6, size_factor=1.0, source="slow")
    out = await storage.get_decisions()
    assert len(out) == 1 and out[0]["action"] == "LONG"  # (symbol, ts) reemplaza


async def test_realized_pnl_roundtrip_y_orden(storage):
    await storage.save_realized_pnl(
        realized={"BTCUSDT": -5.0, "ETHUSDT": 12.5, "SOLUSDT": 0.0}, ts_ms=1000)
    out = await storage.get_realized_pnl()
    assert len(out) == 3
    assert out[0]["symbol"] == "ETHUSDT" and out[0]["realized"] == 12.5  # mayor primero
    assert out[-1]["symbol"] == "BTCUSDT"                                 # menor último
    assert out[0]["updated_ms"] == 1000


async def test_realized_pnl_upsert_por_simbolo(storage):
    await storage.save_realized_pnl(realized={"BTCUSDT": -5.0}, ts_ms=1)
    await storage.save_realized_pnl(realized={"BTCUSDT": 7.0}, ts_ms=2)  # sobrescribe
    out = await storage.get_realized_pnl()
    assert len(out) == 1 and out[0]["realized"] == 7.0 and out[0]["updated_ms"] == 2


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
