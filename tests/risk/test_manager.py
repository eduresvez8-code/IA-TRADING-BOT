"""Tests de escenario del Risk Manager — uno por regla de riesgo (PLAN_MAESTRO §4).

Cubren: stop-loss obligatorio, sizing por volatilidad y sus reductores, techo de
no-apalancamiento, y cada veto (max posiciones, pérdida diaria, kill switch por
drawdown con su latch, feed obsoleto, halt). Cierra con la cadena completa
señal → confluencia → orden.
"""

from datetime import datetime, timezone

import pytest

from src.core.config import load_settings
from src.core.models import Action, Decision, Side, SentimentScore, Signal
from src.decision.confluence import decide
from src.risk.manager import PortfolioState, RiskManager

NOW = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
CFG = load_settings()


def make_decision(action: Action, size_factor: float = 1.0,
                  symbol: str = "BTCUSDT") -> Decision:
    return Decision(
        symbol=symbol, action=action, quant_score=0.8, sentiment_score=0.4,
        size_factor=size_factor, reason="test", timestamp=NOW,
    )


def healthy_state(**overrides) -> PortfolioState:
    base = dict(
        equity=10_000.0, peak_equity=10_000.0, day_start_equity=10_000.0,
        open_positions=0, feed_age_seconds=0.0, halted=False,
    )
    base.update(overrides)
    return PortfolioState(**base)


# ---------- Aprobación, stop-loss obligatorio y geometría de la orden ----------

def test_long_aprobado_construye_orden_con_sl_y_tp():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state())
    assert a.approved is True
    o = a.order
    assert o is not None
    assert o.side == Side.BUY
    # SL obligatorio y del lado que protege; TP por encima a RR×distancia.
    assert o.stop_loss == pytest.approx(1000.0 - 1.5 * 50.0)   # 925
    assert o.take_profit == pytest.approx(1000.0 + 2.0 * 1.5 * 50.0)  # 1150
    assert o.stop_loss < o.entry_price < o.take_profit


def test_short_aprobado_invierte_la_geometria():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.SHORT), price=1000.0, atr=50.0,
                  state=healthy_state())
    assert a.approved is True
    o = a.order
    assert o.side == Side.SELL
    assert o.stop_loss == pytest.approx(1075.0)
    assert o.take_profit == pytest.approx(850.0)
    assert o.take_profit < o.entry_price < o.stop_loss


# ---------------------------- Position sizing ----------------------------

def test_sizing_formula_riesgo_constante():
    # qty = (equity × risk% × size_factor) / stop_distance.
    # 10000 × 1% × 1.0 / (1.5×50) = 100 / 75 = 1.3333…
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG, size_factor=1.0), price=1000.0,
                  atr=50.0, state=healthy_state())
    assert a.order.quantity == pytest.approx(100.0 / 75.0)


def test_size_factor_reduce_la_cantidad():
    rm = RiskManager(CFG)
    full = rm.assess(make_decision(Action.LONG, size_factor=1.0), price=1000.0,
                     atr=50.0, state=healthy_state()).order.quantity
    half = rm.assess(make_decision(Action.LONG, size_factor=0.5), price=1000.0,
                     atr=50.0, state=healthy_state()).order.quantity
    assert half == pytest.approx(full * 0.5)


def test_baja_confianza_reduce_la_cantidad():
    rm = RiskManager(CFG)
    base = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                     state=healthy_state(), confidence=0.9).order.quantity
    low = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                    state=healthy_state(), confidence=0.2).order.quantity
    assert low == pytest.approx(base * CFG.risk.low_confidence_size_factor)


def test_techo_sin_apalancamiento():
    # Stop muy ajustado (ATR pequeño) dispararía una qty enorme; el notional
    # nunca debe exceder la equity → qty se topa en equity/price.
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=1.0,
                  state=healthy_state())
    assert a.order.quantity == pytest.approx(10_000.0 / 1000.0)  # 10
    assert a.order.quantity * a.order.entry_price <= 10_000.0 + 1e-9


def test_atr_cero_no_opera():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=0.0,
                  state=healthy_state())
    assert a.approved is False
    assert a.reason == "invalid_sizing_inputs"


# ------------------------------ Vetos ------------------------------

def test_hold_no_genera_orden():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.HOLD), price=1000.0, atr=50.0,
                  state=healthy_state())
    assert a.approved is False
    assert a.reason == "hold"
    assert a.order is None


def test_veto_max_posiciones():
    rm = RiskManager(CFG)
    state = healthy_state(open_positions=CFG.risk.max_open_positions)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0, state=state)
    assert a.approved is False
    assert a.reason == "max_positions"


def test_veto_perdida_diaria():
    rm = RiskManager(CFG)
    # 3.5% de pérdida en el día supera el límite del 3%.
    state = healthy_state(equity=9_650.0)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0, state=state)
    assert a.approved is False
    assert a.reason == "daily_loss_limit"


def test_veto_feed_obsoleto():
    rm = RiskManager(CFG)
    state = healthy_state(feed_age_seconds=CFG.risk.stale_feed_seconds + 1)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0, state=state)
    assert a.approved is False
    assert a.reason == "stale_feed"


def test_veto_halt_por_reconciliacion():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(halted=True))
    assert a.approved is False
    assert a.reason == "halted"


def test_kill_switch_drawdown_latcha_hasta_reset():
    rm = RiskManager(CFG)
    # 11% de drawdown desde el pico → dispara el kill switch.
    breached = healthy_state(equity=8_900.0, peak_equity=10_000.0)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0, state=breached)
    assert a.approved is False and a.reason == "kill_switch_drawdown"
    assert rm.kill_switch_active is True

    # Aunque la equity se recupere, el latch sigue bloqueando: NO se rearma solo.
    recovered = healthy_state(equity=9_999.0, peak_equity=10_000.0)
    a2 = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0, state=recovered)
    assert a2.approved is False and a2.reason == "kill_switch_drawdown"

    # Solo el rearme manual lo desbloquea.
    rm.reset()
    a3 = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0, state=recovered)
    assert a3.approved is True


# --------------------- Cadena completa señal → orden ---------------------

def test_pipeline_confluencia_a_orden():
    sig = Signal(symbol="BTCUSDT", score=0.8, strategy="ema_cross_rsi", timestamp=NOW)
    sent = SentimentScore(news_id="n1", symbol_scope=["BTCUSDT"], score=0.5,
                          confidence=0.8, high_impact=False, analyzed_at=NOW)
    decision = decide(sig, sent, CFG)
    assert decision.action == Action.LONG and decision.size_factor == 1.0

    rm = RiskManager(CFG)
    a = rm.assess(decision, price=1000.0, atr=50.0, state=healthy_state(),
                  confidence=sent.confidence)
    assert a.approved is True
    assert a.order.decision_reason == "sentiment_confirms"
    assert a.order.stop_loss < a.order.entry_price
