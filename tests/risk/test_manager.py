"""Tests del Risk Manager (contexto acciones cash).

Verifican los INVARIANTES de la política de riesgo, no números mágicos:
    - todo veto de estado dispara antes de dimensionar,
    - el sizing respeta el presupuesto de riesgo y el techo de cash,
    - las acciones se truncan (jamás se redondean arriba),
    - el kill switch latcha hasta reset() manual.
"""

from datetime import datetime, timezone

import pytest

from src.core.config import load_settings
from src.core.models import Action, Decision, Side
from src.risk.manager import PortfolioState, RiskAssessment, RiskManager

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _decision(action=Action.LONG, size_factor=1.0):
    return Decision(symbol="SPY", action=action, score=0.8,
                    size_factor=size_factor, reason="test", timestamp=NOW)


def _state(**overrides):
    base = dict(equity=10_000.0, cash_available=10_000.0, peak_equity=10_000.0,
                day_start_equity=10_000.0, open_positions=0, halted=False)
    base.update(overrides)
    return PortfolioState(**base)


@pytest.fixture
def rm():
    cfg = load_settings()
    return RiskManager(cfg)


# ---------- vetos de estado ----------

def test_aprueba_en_condiciones_normales(rm):
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=_state())
    assert res.approved and res.order is not None
    assert res.order.side == Side.BUY


def test_kill_switch_por_drawdown_latcha(rm):
    # 15% de drawdown (config real) → veto que persiste aunque el equity se recupere.
    bad = _state(equity=8_400.0, peak_equity=10_000.0)   # DD 16%
    assert rm.assess(_decision(), price=100.0, atr=2.0, state=bad).reason == "kill_switch_drawdown"
    ok = _state()                                        # cuenta recuperada
    assert rm.assess(_decision(), price=100.0, atr=2.0, state=ok).reason == "kill_switch_drawdown"
    rm.reset()
    assert rm.assess(_decision(), price=100.0, atr=2.0, state=ok).approved


def test_halted_veta(rm):
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=_state(halted=True))
    assert not res.approved and res.reason == "halted"


def test_perdida_diaria_veta(rm):
    # 2% de pérdida diaria (config real).
    bad = _state(equity=9_790.0, peak_equity=10_000.0, day_start_equity=10_000.0)
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=bad)
    assert not res.approved and res.reason == "daily_loss_limit"


def test_tope_de_posiciones_veta(rm):
    full = _state(open_positions=rm.cfg.risk.max_open_positions)
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=full)
    assert not res.approved and res.reason == "max_positions"


def test_hold_no_genera_orden(rm):
    res = rm.assess(_decision(action=Action.HOLD), price=100.0, atr=2.0, state=_state())
    assert not res.approved and res.reason == "hold"


def test_short_vetado_en_cuenta_cash(rm):
    res = rm.assess(_decision(action=Action.SHORT), price=100.0, atr=2.0, state=_state())
    assert not res.approved and res.reason == "short_not_allowed"


def test_atr_invalido_veta(rm):
    res = rm.assess(_decision(), price=100.0, atr=0.0, state=_state())
    assert not res.approved and res.reason == "invalid_sizing_inputs"


def test_stop_bajo_cero_veta(rm):
    # Precio 1.0 con ATR 2.0 y multiplicador 2 → stop en -3: sin sentido.
    res = rm.assess(_decision(), price=1.0, atr=2.0, state=_state())
    assert not res.approved and res.reason == "stop_below_zero"


# ---------- sizing ----------

def test_sizing_respeta_presupuesto_de_riesgo(rm):
    # equity 10k, riesgo 0.5% = $50. ATR 2, mult 2 → stop_distance 4 → qty cruda
    # 12.5 → floor 12 acciones. Riesgo real = 12·4 = $48 ≤ $50 (truncar protege).
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=_state())
    assert res.approved
    o = res.order
    assert o.quantity == 12.0
    stop_distance = o.entry_price - o.stop_loss
    assert stop_distance == pytest.approx(4.0)
    riesgo_real = o.quantity * stop_distance
    presupuesto = 10_000.0 * rm.cfg.risk.risk_per_trade_pct / 100.0
    assert riesgo_real <= presupuesto


def test_size_factor_reduce_la_cantidad(rm):
    full = rm.assess(_decision(size_factor=1.0), price=100.0, atr=2.0, state=_state())
    half = rm.assess(_decision(size_factor=0.5), price=100.0, atr=2.0, state=_state())
    assert half.order.quantity <= full.order.quantity / 2 + 1  # floor puede recortar 1


def test_sin_apalancamiento_el_notional_no_excede_el_cash(rm):
    # Cash de solo $500 a precio 100 → como mucho 5 acciones, aunque el
    # presupuesto de riesgo pidiera más.
    poor = _state(cash_available=500.0)
    res = rm.assess(_decision(), price=100.0, atr=0.5, state=poor)
    assert res.approved
    assert res.order.quantity * res.order.entry_price <= 500.0


def test_cantidad_cero_tras_floor_veta(rm):
    # Cash $50 a precio 100 → 0.5 acciones → floor 0 → rechazo.
    broke = _state(cash_available=50.0)
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=broke)
    assert not res.approved and res.reason == "below_min_qty"


def test_let_winners_run_sin_take_profit(rm):
    # Config real: let_winners_run=true → la orden no lleva techo de ganancia.
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=_state())
    assert res.approved and res.order.take_profit is None


def test_take_profit_fijo_si_se_desactiva_let_winners(rm):
    rm.cfg.risk.let_winners_run = False
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=_state())
    assert res.approved
    # TP = entrada + RR × distancia = 100 + 2×4 = 108.
    assert res.order.take_profit == pytest.approx(108.0)


def test_resultado_es_riskassessment(rm):
    res = rm.assess(_decision(), price=100.0, atr=2.0, state=_state())
    assert isinstance(res, RiskAssessment)
