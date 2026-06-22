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


async def test_liveness_fresco_vs_obsoleto(populated):
    fresh = build_snapshot(populated, now=NOW + timedelta(seconds=100), testnet=True)
    assert fresh["meta"]["stale"] is False        # 100s < 3×300s
    stale = build_snapshot(populated, now=NOW + timedelta(seconds=2000), testnet=True)
    assert stale["meta"]["stale"] is True          # 2000s > 900s


async def test_modo_gates_off_por_defecto(populated):
    snap = build_snapshot(populated, now=NOW, testnet=True)
    # settings.yaml del repo: event/sentiment enabled = false → sin originación.
    assert "Gates OFF" in snap["meta"]["mode"]
    assert snap["meta"]["event_enabled"] is False


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
