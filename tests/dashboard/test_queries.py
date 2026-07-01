"""Tests de build_snapshot: estructura, métricas derivadas, liveness y BD ausente.

El dashboard lee en modo `ro` lo que el engine escribe. Aquí escribimos vía
Storage (la ruta real de escritura), cerramos, y leemos vía build_snapshot, para
verificar que ambos lados están alineados (esquema + derivaciones).
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import load_settings
from src.data.storage import Storage
from src.dashboard.queries import build_snapshot

NOW = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)


def _settings(tmp_path):
    s = load_settings().model_copy(deep=True)
    s.storage.db_path = str(tmp_path / "trading.db")
    return s


@pytest.fixture
async def populated(tmp_path):
    """Una BD con un poco de todo, devuelta junto a su settings apuntando a ella."""
    settings = _settings(tmp_path)
    st = await Storage(settings.storage.db_path, tmp_path / "candles").init()
    # Curva de equity: sube de 10000 a 10120; pico 10150 vía session_state.
    base = int(NOW.timestamp() * 1000)
    for i, eq in enumerate((10000.0, 10080.0, 10120.0)):
        await st.save_equity_snapshot(
            ts_ms=base - (2 - i) * 300_000,  # cada 5 min, el último en NOW
            wallet=eq - 20.0, equity=eq, upnl=20.0,
            positions=[{"symbol": "BTCUSDT", "side": "LONG", "qty": 0.1,
                        "entry_price": 50000.0, "upnl": 20.0}])
    await st.save_session_state(peak_wallet=10150.0, day_start_wallet=10000.0,
                                day=NOW.date().isoformat(), kill_switch=False)
    await st.save_decision(ts_ms=base, symbol="BTCUSDT", action="LONG",
                           reason="regime_confirms", quant_score=0.8,
                           sentiment_score=0.6, size_factor=1.0, source="slow")
    await st.save_order(client_order_id="c1", ts_ms=base, symbol="BTCUSDT",
                        side="BUY", position_side="LONG", type="MARKET",
                        quantity=0.1, price=50000.0, status="FILLED",
                        exchange_order_id="1", decision_reason="regime_confirms")
    # PnL realizado por símbolo: BTC perdió, ETH ganó (sin posición abierta).
    await st.save_realized_pnl(realized={"BTCUSDT": -8.0, "ETHUSDT": 15.0}, ts_ms=base)
    await st.close()
    return settings


async def test_snapshot_estructura_y_kpis(populated):
    snap = build_snapshot(populated, now=NOW, testnet=True)
    assert snap["meta"]["has_data"] is True
    assert snap["meta"]["testnet"] is True
    k = snap["kpis"]
    assert k["equity"] == 10120.0
    assert k["wallet"] == 10100.0
    assert k["upnl"] == 20.0
    assert k["peak_wallet"] == 10150.0
    # drawdown = (10150 - 10120)/10150*100 ≈ 0.2956
    assert k["drawdown_pct"] == pytest.approx((10150 - 10120) / 10150 * 100)
    # PnL del día = equity - day_start = 120
    assert k["day_pnl"] == pytest.approx(120.0)
    assert k["day_pnl_pct"] == pytest.approx(1.2)
    assert k["open_positions"] == 1
    assert k["kill_switch"] is False


async def test_snapshot_curva_ascendente_y_paneles(populated):
    snap = build_snapshot(populated, now=NOW, testnet=True)
    eq = snap["equity"]
    assert [p["equity"] for p in eq] == [10000.0, 10080.0, 10120.0]  # ascendente
    assert snap["positions"][0]["symbol"] == "BTCUSDT"
    assert snap["decisions"][0]["reason"] == "regime_confirms"
    assert snap["orders"][0]["client_order_id"] == "c1"


async def test_pnl_por_simbolo_fusiona_realizado_y_no_realizado(populated):
    snap = build_snapshot(populated, now=NOW, testnet=True)
    by = {r["symbol"]: r for r in snap["pnl_by_symbol"]}
    # BTC: realizado -8 + no realizado (uPnL posición) +20 = total 12.
    assert by["BTCUSDT"]["realized"] == pytest.approx(-8.0)
    assert by["BTCUSDT"]["unrealized"] == pytest.approx(20.0)
    assert by["BTCUSDT"]["total"] == pytest.approx(12.0)
    # ETH: realizado +15, sin posición → total 15.
    assert by["ETHUSDT"]["total"] == pytest.approx(15.0)
    # Todos los símbolos del universo aparecen (incluso planos en 0).
    assert set(by) >= set(populated.market.symbols)
    # Orden: por total descendente (ETH 15 antes que BTC 12).
    syms = [r["symbol"] for r in snap["pnl_by_symbol"]]
    assert syms.index("ETHUSDT") < syms.index("BTCUSDT")


async def test_pnl_por_simbolo_excluye_fuera_del_universo(tmp_path):
    # Un símbolo retirado del universo (p.ej. DOGE) con PnL viejo en la tabla NO debe
    # aparecer en el panel: se filtra al universo activo sin borrar el historial.
    settings = _settings(tmp_path)
    st = await Storage(settings.storage.db_path, tmp_path / "candles").init()
    base = int(NOW.timestamp() * 1000)
    await st.save_equity_snapshot(ts_ms=base, wallet=100.0, equity=100.0, upnl=0.0,
                                  positions=[])
    await st.save_realized_pnl(
        realized={"BTCUSDT": 5.0, "DOGEUSDT": 99.0}, ts_ms=base)  # DOGE ya no en universo
    await st.close()
    snap = build_snapshot(settings, now=NOW, testnet=True)
    syms = {r["symbol"] for r in snap["pnl_by_symbol"]}
    assert "DOGEUSDT" not in syms
    assert syms == set(settings.market.symbols)


async def test_modo_refleja_quant_apagado(populated):
    # El badge "quant OFF" vive en la rama Slow Path puro; aislamos ese modo
    # apagando el Fast Path para probar el label del quant.
    populated.event.enabled = False
    populated.confluence.quant_regime_enabled = False
    snap = build_snapshot(populated, now=NOW, testnet=True)
    assert "quant OFF" in snap["meta"]["mode"]


async def test_liveness_fresco_vs_obsoleto(populated):
    # El umbral de obsolescencia escala con el timeframe base (ahora 1h): stale si
    # la antigüedad supera stale_after_intervals(3) × 3600s = 10800s (3h).
    fresh = build_snapshot(populated, now=NOW + timedelta(seconds=5000), testnet=True)
    assert fresh["meta"]["stale"] is False        # 5000s < 3×3600s
    stale = build_snapshot(populated, now=NOW + timedelta(seconds=12000), testnet=True)
    assert stale["meta"]["stale"] is True          # 12000s > 10800s


async def test_online_por_latido_fresco_vs_viejo(tmp_path):
    # El online/offline lo decide el LATIDO del bot, no la antigüedad del snapshot.
    settings = _settings(tmp_path)
    st = await Storage(settings.storage.db_path, tmp_path / "candles").init()
    base = int(NOW.timestamp() * 1000)
    await st.save_equity_snapshot(ts_ms=base, wallet=100.0, equity=100.0, upnl=0.0,
                                  positions=[])
    await st.save_heartbeat(base)                      # latido en NOW
    await st.close()
    fresh = build_snapshot(settings, now=NOW + timedelta(seconds=10), testnet=True)
    assert fresh["meta"]["online"] is True             # 10s < timeout (45s)
    stale = build_snapshot(settings, now=NOW + timedelta(seconds=120), testnet=True)
    assert stale["meta"]["online"] is False            # 120s > timeout → offline


async def test_sin_latido_es_offline(populated):
    # Una BD con datos pero sin latido (bot nunca arrancó con heartbeat) → offline.
    snap = build_snapshot(populated, now=NOW, testnet=True)
    assert snap["meta"]["online"] is False
    assert snap["meta"]["heartbeat_ms"] is None


async def test_modo_repo_es_slow_path(populated):
    # settings.yaml del repo (2026-07-01): sentiment=True, event=False → Slow Path.
    # Fija el estado SHIPPED: si alguien cambia un gate sin querer, este test lo caza.
    snap = build_snapshot(populated, now=NOW, testnet=True)
    assert "Slow Path" in snap["meta"]["mode"]
    assert snap["meta"]["event_enabled"] is False
    assert snap["meta"]["sentiment_enabled"] is True


async def test_modo_gates_off_cuando_ambos_desactivados(populated):
    populated.event.enabled = False
    populated.sentiment.enabled = False
    snap = build_snapshot(populated, now=NOW, testnet=True)
    assert "Gates OFF" in snap["meta"]["mode"]


async def test_modo_hibrido_con_ambos_gates(populated):
    populated.event.enabled = True
    populated.sentiment.enabled = True
    snap = build_snapshot(populated, now=NOW, testnet=True)
    assert "Híbrido" in snap["meta"]["mode"]


def test_bd_ausente_devuelve_snapshot_vacio(tmp_path):
    settings = _settings(tmp_path)  # nunca se crea la BD
    snap = build_snapshot(settings, now=NOW, testnet=None)
    assert snap["meta"]["has_data"] is False
    assert snap["meta"]["stale"] is True
    assert snap["equity"] == [] and snap["kpis"]["equity"] is None
    assert snap["kpis"]["open_positions"] == 0
