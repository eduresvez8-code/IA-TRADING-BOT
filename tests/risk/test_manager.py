"""Tests de escenario del Risk Manager — uno por regla de riesgo (PLAN_MAESTRO §4).

Venue: Binance Spot, long-only. Cubren: stop-loss obligatorio, sizing por
volatilidad y sus reductores, los dos techos (saldo libre + exposición agregada),
microestructura (stepSize/tickSize/minNotional), y cada veto. Cierra con la
cadena completa señal → confluencia → orden.
"""

from datetime import datetime, timezone

import pytest

from src.core.config import load_settings
from src.core.models import Action, Decision, Side, SentimentScore, Signal, SymbolFilters
from src.decision.confluence import decide
from src.risk.manager import PortfolioState, RiskManager

NOW = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
CFG = load_settings()

# Filtros "sin fricción": paso/tick finísimos y sin mínimos → aíslan la
# matemática de sizing del ajuste de microestructura.
FINE = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.00000001",
                     min_qty="0", min_notional="0")


def make_decision(action: Action, size_factor: float = 1.0,
                  symbol: str = "BTCUSDT") -> Decision:
    quant = 0.8 if action != Action.SHORT else -0.8
    return Decision(
        symbol=symbol, action=action, quant_score=quant, sentiment_score=0.4,
        size_factor=size_factor, reason="test", timestamp=NOW,
    )


def healthy_state(**overrides) -> PortfolioState:
    base = dict(
        equity=10_000.0, free_balance=10_000.0, committed_notional=0.0,
        peak_equity=10_000.0, day_start_equity=10_000.0,
        open_positions=0, feed_age_seconds=0.0, halted=False,
    )
    base.update(overrides)
    return PortfolioState(**base)


# ---------- Aprobación, stop-loss obligatorio y geometría de la orden ----------

def test_long_aprobado_construye_orden_con_sl_y_tp():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    assert a.approved is True
    o = a.order
    assert o.side == Side.BUY
    assert o.stop_loss == pytest.approx(925.0)              # 1000 - 1.5*50
    assert o.take_profit == pytest.approx(1150.0)           # 1000 + 2*1.5*50
    assert o.stop_loss < o.entry_price < o.take_profit


def test_short_open_vetado_en_spot():
    # Red de seguridad: aunque llegue una Decision SHORT, no se abre en Spot.
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.SHORT), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    assert a.approved is False
    assert a.reason == "short_not_allowed_spot"
    assert a.order is None


# ---------------------------- Position sizing ----------------------------

def test_sizing_formula_riesgo_constante():
    # qty = (equity × risk% × size_factor) / stop_distance = 100 / 75.
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    assert a.order.quantity == pytest.approx(100.0 / 75.0)


def test_size_factor_reduce_la_cantidad():
    rm = RiskManager(CFG)
    full = rm.assess(make_decision(Action.LONG, 1.0), price=1000.0, atr=50.0,
                     state=healthy_state(), filters=FINE).order.quantity
    half = rm.assess(make_decision(Action.LONG, 0.5), price=1000.0, atr=50.0,
                     state=healthy_state(), filters=FINE).order.quantity
    assert half == pytest.approx(full * 0.5)


def test_baja_confianza_reduce_la_cantidad():
    rm = RiskManager(CFG)
    base = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                     state=healthy_state(), filters=FINE, confidence=0.9).order.quantity
    low = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                    state=healthy_state(), filters=FINE, confidence=0.2).order.quantity
    assert low == pytest.approx(base * CFG.risk.low_confidence_size_factor)


def test_atr_cero_no_opera():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=0.0,
                  state=healthy_state(), filters=FINE)
    assert a.approved is False and a.reason == "invalid_sizing_inputs"


# ------------- Techos: saldo libre y exposición agregada (Spot) -------------

def test_techo_exposicion_con_colchon():
    # ATR diminuto → la cantidad por riesgo es enorme; el nocional se topa en el
    # 95% de la equity (no el 100%): queda 5% de colchón para fees/slippage.
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=1.0,
                  state=healthy_state(), filters=FINE)
    assert a.order.quantity * a.order.entry_price == pytest.approx(9_500.0)


def test_dinero_fantasma_se_topa_en_saldo_libre():
    # Equity 10k pero solo 2k libres y 8k comprometidos. El código viejo habría
    # permitido hasta equity/price=10 (10k de nocional) → INSUFFICIENT_BALANCE.
    rm = RiskManager(CFG)
    state = healthy_state(free_balance=2_000.0, committed_notional=8_000.0, open_positions=2)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=1.0,
                  state=state, filters=FINE)
    assert a.approved is True
    # Política: 0.95×10000 − 8000 = 1500 es el binding (más restrictivo que free).
    assert a.order.quantity * a.order.entry_price == pytest.approx(1_500.0)
    assert a.order.quantity * a.order.entry_price <= state.free_balance


def test_veto_exposicion_agregada():
    rm = RiskManager(CFG)
    state = healthy_state(committed_notional=9_600.0, open_positions=2)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is False and a.reason == "portfolio_exposure"


def test_veto_saldo_libre_insuficiente():
    # Con 4 USDT libres no cabe ni la orden mínima de Binance (minNotional=5).
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.00000001",
                         min_qty="0", min_notional="5")
    state = healthy_state(free_balance=4.0)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=filt)
    assert a.approved is False and a.reason == "insufficient_free_balance"


# ----------------------- Microestructura (filtros) -----------------------

def test_qty_truncada_a_step():
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.001",
                         min_qty="0", min_notional="0")
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=filt)
    # 1.33333… truncado (no redondeado) al paso 0.001 → 1.333.
    assert a.order.quantity == pytest.approx(1.333, abs=1e-9)


def test_sl_tp_redondeados_a_tick():
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.5", step_size="0.00000001",
                         min_qty="0", min_notional="0")
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=33.34,
                  state=healthy_state(), filters=filt)
    # SL/TP crudos (949.99 / 1100.02) cuantizados al tick de 0.5.
    assert a.order.stop_loss == pytest.approx(950.0)
    assert a.order.take_profit == pytest.approx(1100.0)


def test_below_min_notional_rechaza_no_infla():
    # Baja confianza encoge el nocional bajo el mínimo → se RECHAZA (no se sube).
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.00000001",
                         min_qty="0", min_notional="500")
    a = rm.assess(make_decision(Action.LONG, size_factor=0.5), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=filt, confidence=0.2)
    assert a.approved is False and a.reason == "below_min_notional"


def test_below_min_qty_rechaza():
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="1",
                         min_qty="2", min_notional="5")
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=filt)
    # 1.333 → floor a paso 1 → 1.0, menor que minQty 2.
    assert a.approved is False and a.reason == "below_min_qty"


def test_stop_que_redondea_a_la_entrada_se_rechaza():
    # Tick gigante (200): el SL crudo 925 redondea a 1000 = entrada → sin protección.
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="200", step_size="0.00000001",
                         min_qty="0", min_notional="0")
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=filt)
    assert a.approved is False and a.reason == "stop_rounds_to_entry"


# ------------------------------ Vetos de estado ------------------------------

def test_hold_no_genera_orden():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.HOLD), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    assert a.approved is False and a.reason == "hold" and a.order is None


def test_veto_max_posiciones():
    rm = RiskManager(CFG)
    state = healthy_state(open_positions=CFG.risk.max_open_positions)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is False and a.reason == "max_positions"


def test_veto_perdida_diaria():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(equity=9_650.0), filters=FINE)
    assert a.approved is False and a.reason == "daily_loss_limit"


def test_veto_feed_obsoleto():
    rm = RiskManager(CFG)
    state = healthy_state(feed_age_seconds=CFG.risk.stale_feed_seconds + 1)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is False and a.reason == "stale_feed"


def test_veto_halt_por_reconciliacion():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(halted=True), filters=FINE)
    assert a.approved is False and a.reason == "halted"


def test_kill_switch_drawdown_latcha_hasta_reset():
    rm = RiskManager(CFG)
    breached = healthy_state(equity=8_900.0, free_balance=8_900.0)  # 11% drawdown
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=breached, filters=FINE)
    assert a.approved is False and a.reason == "kill_switch_drawdown"
    assert rm.kill_switch_active is True

    recovered = healthy_state(equity=9_999.0, free_balance=9_999.0)
    a2 = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                   state=recovered, filters=FINE)
    assert a2.approved is False and a2.reason == "kill_switch_drawdown"

    rm.reset()
    a3 = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                   state=recovered, filters=FINE)
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
                  filters=FINE, confidence=sent.confidence)
    assert a.approved is True
    assert a.order.decision_reason == "sentiment_confirms"
    assert a.order.stop_loss < a.order.entry_price
