"""Tests de escenario del Risk Manager — uno por regla de riesgo (PLAN_MAESTRO §4).

Venue: Binance Futuros USD-M. Cubren: stop-loss obligatorio, simetría LONG/SHORT,
sizing por volatilidad y sus reductores, los dos techos de MARGEN (available_balance
+ margen agregado del wallet), microestructura (stepSize/tickSize/minNotional),
leverage en la orden, y cada veto. Cierra con la cadena señal → confluencia → orden.
"""

from datetime import datetime, timezone

import pytest

from src.core.config import load_settings
from src.core.models import (
    Action,
    Decision,
    PositionSide,
    Side,
    SentimentScore,
    Signal,
    SymbolFilters,
)
from src.risk.manager import PortfolioState, RiskManager

NOW = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
# El repo en vivo tiene el quant apagado (news_only); este test ejercita el pipeline
# confluencia→orden con el régimen confirmando, que requiere el quant encendido.
CFG = load_settings().model_copy(deep=True)
CFG.confluence.quant_regime_enabled = True
# El repo en vivo corre con let_winners_run=true (sin techo fijo, 2026-07-07); este
# archivo fija false para seguir probando a fondo la rama de cálculo de TP fijo
# (sigue siendo código real y necesario). La rama let_winners_run=True tiene sus
# propios tests dedicados más abajo.
CFG.risk.let_winners_run = False
L = CFG.risk.max_leverage  # 3

# Filtros "sin fricción": paso/tick finísimos y sin mínimos → aíslan el sizing.
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
        wallet_balance=10_000.0, available_balance=10_000.0, committed_margin=0.0,
        peak_wallet_balance=10_000.0, day_start_wallet_balance=10_000.0,
        open_positions=0, feed_age_seconds=0.0, halted=False,
    )
    base.update(overrides)
    return PortfolioState(**base)


# ---------- Aprobación, stop-loss obligatorio, simetría y leverage ----------

def test_long_aprobado_construye_orden_con_sl_y_tp():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    assert a.approved is True
    o = a.order
    assert o.side == Side.BUY
    assert o.position_side == PositionSide.LONG
    assert o.leverage == L
    assert o.stop_loss == pytest.approx(925.0)      # 1000 - 1.5*50
    assert o.take_profit == pytest.approx(1150.0)    # 1000 + 2*1.5*50
    assert o.stop_loss < o.entry_price < o.take_profit


def test_short_aprobado_simetrico():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.SHORT), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    assert a.approved is True
    o = a.order
    assert o.side == Side.SELL
    assert o.position_side == PositionSide.SHORT
    assert o.stop_loss == pytest.approx(1075.0)
    assert o.take_profit == pytest.approx(850.0)


def test_cap_direccional_veta_long_correlacionado_de_mas():
    # Con max_same_direction_positions longs ya abiertos, un long más se veta por
    # concentración por correlación — aunque queden ranuras en max_open_positions.
    cap = CFG.risk.max_same_direction_positions
    rm = RiskManager(CFG)
    state = healthy_state(open_positions=cap, long_positions=cap, short_positions=0)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is False
    assert a.reason == "max_same_direction"


def test_cap_direccional_no_estorba_la_direccion_opuesta():
    # El cap es POR dirección: con el cupo de longs lleno, un short sigue pasando.
    cap = CFG.risk.max_same_direction_positions
    rm = RiskManager(CFG)
    state = healthy_state(open_positions=cap, long_positions=cap, short_positions=0)
    a = rm.assess(make_decision(Action.SHORT), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is True
    assert a.order.position_side == PositionSide.SHORT
    assert a.order.take_profit < a.order.entry_price < a.order.stop_loss


def test_orden_lleva_el_leverage_del_bot():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    assert a.order.leverage == CFG.risk.max_leverage


# -------------------- let_winners_run: sin take-profit fijo --------------------
# 2026-07-07: config real del repo (settings.yaml). El stop sigue acotando la
# pérdida; la ganancia queda sin techo (sale por FLIP o time-stop, no por TP).

def _cfg_let_winners_run():
    c = CFG.model_copy(deep=True)
    c.risk.let_winners_run = True
    return c


def test_let_winners_run_deja_take_profit_en_none():
    rm = RiskManager(_cfg_let_winners_run())
    long_ = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                      state=healthy_state(), filters=FINE)
    short_ = rm.assess(make_decision(Action.SHORT), price=1000.0, atr=50.0,
                       state=healthy_state(), filters=FINE)
    assert long_.approved is True and long_.order.take_profit is None
    assert short_.approved is True and short_.order.take_profit is None
    # El stop de pérdida NO se toca: sigue siendo el único techo de riesgo.
    assert long_.order.stop_loss == pytest.approx(925.0)   # 1000 - 1.5*50
    assert short_.order.stop_loss == pytest.approx(1075.0)  # 1000 + 1.5*50


def test_let_winners_run_no_cambia_el_sizing():
    # El riesgo en $ depende SOLO de la distancia al stop, nunca del take-profit:
    # la cantidad debe ser IDÉNTICA con o sin techo fijo (misma entrada/ATR/riesgo).
    rm_fijo = RiskManager(CFG)                    # let_winners_run=False en este archivo
    rm_libre = RiskManager(_cfg_let_winners_run())
    qty_fijo = rm_fijo.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                              state=healthy_state(), filters=FINE).order.quantity
    qty_libre = rm_libre.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                                state=healthy_state(), filters=FINE).order.quantity
    assert qty_libre == pytest.approx(qty_fijo)


# ---------------------------- Position sizing ----------------------------

def test_sizing_formula_riesgo_constante():
    # qty = (wallet × risk% × size_factor) / stop_distance = 100 / 75.
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
    # Confianza en la banda media [min_confidence_to_trade, low_confidence_threshold)
    # opera pero a tamaño reducido. 0.6 cae en esa banda con la config del repo.
    rm = RiskManager(CFG)
    base = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                     state=healthy_state(), filters=FINE, confidence=0.9).order.quantity
    low = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                    state=healthy_state(), filters=FINE, confidence=0.6).order.quantity
    assert low == pytest.approx(base * CFG.risk.low_confidence_size_factor)


def test_confianza_bajo_el_piso_veta_el_trade():
    # Por debajo de min_confidence_to_trade la noticia es demasiado incierta: no se
    # opera (veto duro), no solo se reduce el tamaño.
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE, confidence=0.3)
    assert a.approved is False and a.reason == "low_confidence"


def test_atr_cero_no_opera():
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=0.0,
                  state=healthy_state(), filters=FINE)
    assert a.approved is False and a.reason == "invalid_sizing_inputs"


# ------------- Techos de margen: available_balance y margen agregado -------------

def test_techo_margen_agregado_con_colchon():
    # ATR diminuto → qty por riesgo enorme; el margen inicial se topa en el 85%
    # del wallet (deja 15% de colchón para PnL/liquidación).
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=1.0,
                  state=healthy_state(), filters=FINE)
    margin = a.order.quantity * a.order.entry_price / a.order.leverage
    assert margin == pytest.approx(8_500.0)   # 0.85 × 10000


def test_available_balance_es_el_techo_fisico():
    # Margen libre escaso (1k) pese a wallet grande (10k): el techo físico manda.
    rm = RiskManager(CFG)
    state = healthy_state(available_balance=1_000.0, committed_margin=2_000.0, open_positions=2)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=1.0,
                  state=state, filters=FINE)
    assert a.approved is True
    margin = a.order.quantity * a.order.entry_price / a.order.leverage
    assert margin == pytest.approx(1_000.0)   # = available_balance
    assert margin <= state.available_balance + 1e-9


def test_veto_margen_agregado():
    rm = RiskManager(CFG)
    # 8600 comprometido > 85% del wallet (8500) → sin sitio para más margen.
    state = healthy_state(committed_margin=8_600.0, available_balance=1_400.0, open_positions=2)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is False and a.reason == "portfolio_margin"


def test_veto_margen_insuficiente():
    # Con 1 USDT de margen libre no cabe ni la orden mínima (minNotional=5).
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.00000001",
                         min_qty="0", min_notional="5")
    state = healthy_state(available_balance=1.0)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=filt)
    assert a.approved is False and a.reason == "insufficient_margin"


# ----------------------- Microestructura (filtros) -----------------------

def test_qty_truncada_a_step():
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.001",
                         min_qty="0", min_notional="0")
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=filt)
    assert a.order.quantity == pytest.approx(1.333, abs=1e-9)


def test_sl_tp_redondeados_a_tick():
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.5", step_size="0.00000001",
                         min_qty="0", min_notional="0")
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=33.34,
                  state=healthy_state(), filters=filt)
    assert a.order.stop_loss == pytest.approx(950.0)
    assert a.order.take_profit == pytest.approx(1100.0)


def test_below_min_notional_rechaza_no_infla():
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.00000001",
                         min_qty="0", min_notional="500")
    a = rm.assess(make_decision(Action.LONG, size_factor=0.5), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=filt, confidence=0.6)
    assert a.approved is False and a.reason == "below_min_notional"


def test_below_min_qty_rechaza():
    rm = RiskManager(CFG)
    filt = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="1",
                         min_qty="2", min_notional="5")
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=filt)
    assert a.approved is False and a.reason == "below_min_qty"


def test_stop_que_redondea_a_la_entrada_se_rechaza():
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
                  state=healthy_state(wallet_balance=9_650.0), filters=FINE)
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
    breached = healthy_state(wallet_balance=8_900.0, available_balance=8_900.0)  # 11% DD
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=breached, filters=FINE)
    assert a.approved is False and a.reason == "kill_switch_drawdown"
    assert rm.kill_switch_active is True

    recovered = healthy_state(wallet_balance=9_999.0, available_balance=9_999.0)
    a2 = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                   state=recovered, filters=FINE)
    assert a2.approved is False and a2.reason == "kill_switch_drawdown"

    rm.reset()
    a3 = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                   state=recovered, filters=FINE)
    assert a3.approved is True


# --------------------- Modo event (Fase 2.4): sizing diferenciado ---------------------

def test_event_mode_stop_mas_ancho_da_qty_menor():
    # Con ATR y riesgo fijos, stop_mult mayor → stop_distance mayor → qty menor.
    # qty_slow  = (10000 × 0.01 × 1.0) / (1.5 × 50) = 100/75 ≈ 1.333
    # qty_event = (10000 × 0.005 × 1.0) / (2.5 × 50) = 50/125 = 0.4
    rm = RiskManager(CFG)
    slow = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                     state=healthy_state(), filters=FINE, mode="slow")
    event = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                      state=healthy_state(), filters=FINE, mode="event")
    assert slow.approved and event.approved
    assert event.order.quantity < slow.order.quantity
    assert event.order.stop_loss == pytest.approx(1000.0 - 2.5 * 50)  # 875


def test_event_mode_risk_pct_menor_reduce_tamanio():
    # Con el mismo stop, el presupuesto base menor de evento (0.5% vs 1%) reduce qty.
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE, mode="event")
    # risk_amount = 10000 × 0.005 × 1.0 = 50; stop = 2.5×50=125; qty = 0.4
    assert a.approved
    assert a.order.quantity == pytest.approx(50.0 / 125.0)


def test_event_mode_vol_expansion_recorta():
    # vol_ratio = atr/baseline = 100/40 = 2.5 > cap=2.0 → vol_damp = 2.0/2.5 = 0.8
    # qty = (10000 × 0.005 × 1.0 × 0.8) / (2.5 × 100) = 40/250 = 0.16
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=100.0,
                  state=healthy_state(), filters=FINE,
                  mode="event", atr_baseline=40.0)
    assert a.approved
    expected_qty = (10_000 * 0.005 * 1.0 * (2.0 / 2.5)) / (2.5 * 100)
    assert a.order.quantity == pytest.approx(expected_qty)


def test_event_mode_vol_bajo_cap_no_recorta():
    # vol_ratio = atr/baseline = 50/40 = 1.25 < cap=2.0 → vol_damp = 1.0 (sin recorte)
    # qty = (10000 × 0.005 × 1.0 × 1.0) / (2.5 × 50) = 50/125 = 0.4
    rm = RiskManager(CFG)
    sin_recorte = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                            state=healthy_state(), filters=FINE, mode="event")
    con_baseline_baja = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                                  state=healthy_state(), filters=FINE,
                                  mode="event", atr_baseline=40.0)
    assert sin_recorte.approved and con_baseline_baja.approved
    # baseline=40, vol_ratio=1.25<cap → vol_damp=1.0: misma qty que sin baseline
    assert con_baseline_baja.order.quantity == pytest.approx(sin_recorte.order.quantity)


def test_event_mode_atr_baseline_none_sin_recorte():
    # Sin baseline → vol_damp=1.0: el resultado es idéntico al caso con baseline ≥ atr.
    rm = RiskManager(CFG)
    a_none = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                       state=healthy_state(), filters=FINE,
                       mode="event", atr_baseline=None)
    a_cero = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                       state=healthy_state(), filters=FINE,
                       mode="event", atr_baseline=0.0)
    assert a_none.approved and a_cero.approved
    # Ambos deben dar la misma qty (vol_damp=1.0 en los dos casos)
    assert a_none.order.quantity == pytest.approx(a_cero.order.quantity)


def test_slow_mode_inalterado_regresion():
    # Los 460 tests del Slow Path dependen de que mode="slow" sea equivalente al
    # comportamiento anterior (sin modificar ninguno de sus parámetros).
    rm = RiskManager(CFG)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=healthy_state(), filters=FINE)
    # Misma fórmula: qty = (10000 × 0.01 × 1.0) / (1.5 × 50) = 100/75
    assert a.order.quantity == pytest.approx(100.0 / 75.0)
    assert a.order.stop_loss == pytest.approx(925.0)  # 1000 - 1.5×50


# --------------------- Cadena completa señal → orden (SHORT) ---------------------

def test_pipeline_confluencia_a_orden_short():
    # Hack/FUD: quant bajista + sentimiento negativo fuerte → SHORT pleno.
    sig = Signal(symbol="BTCUSDT", score=-0.8, strategy="ema_cross_rsi", timestamp=NOW)
    sent = SentimentScore(news_id="n1", symbol_scope=["BTCUSDT"], score=-0.7,
                          confidence=0.8, high_impact=False, analyzed_at=NOW)
    from src.decision.confluence import decide
    decision = decide(sig, sent, CFG)
    assert decision.action == Action.SHORT and decision.size_factor == 1.0

    rm = RiskManager(CFG)
    a = rm.assess(decision, price=1000.0, atr=50.0, state=healthy_state(),
                  filters=FINE, confidence=sent.confidence)
    assert a.approved is True
    assert a.order.side == Side.SELL
    assert a.order.stop_loss > a.order.entry_price
    assert a.order.decision_reason == "regime_confirms"


# --- Auditoría 2026-07: umbral de feed obsoleto escalado al timeframe ---

def test_stale_feed_usa_umbral_escalado_si_esta_estampado():
    # Con vela base de 1h, la edad NORMAL del feed entre velas es de hasta 3600s.
    # El orquestador estampa stale_after_seconds = max(stale_feed_seconds,
    # intervals × intervalo); el RM debe comparar contra ESE umbral, no contra el
    # absoluto (30s), o vetaría cualquier trade de evento a mitad de vela.
    rm = RiskManager(CFG)
    state = healthy_state(
        feed_age_seconds=1800.0,            # 30 min: normal a mitad de vela 1h
        stale_after_seconds=7200.0,         # 2 × 3600s (intervals × intervalo)
    )
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is True


def test_stale_feed_veta_si_supera_el_umbral_escalado():
    rm = RiskManager(CFG)
    state = healthy_state(feed_age_seconds=7300.0, stale_after_seconds=7200.0)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is False and a.reason == "stale_feed"


def test_stale_feed_sin_estampar_conserva_el_umbral_absoluto():
    # Compatibilidad: un snapshot sin stale_after_seconds (None) se compara contra
    # risk.stale_feed_seconds, exactamente como antes de la auditoría.
    rm = RiskManager(CFG)
    state = healthy_state(feed_age_seconds=CFG.risk.stale_feed_seconds + 1)
    a = rm.assess(make_decision(Action.LONG), price=1000.0, atr=50.0,
                  state=state, filters=FINE)
    assert a.approved is False and a.reason == "stale_feed"
