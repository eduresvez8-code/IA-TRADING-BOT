"""Risk Manager: el único módulo con poder de veto sobre TODA orden.

PLAN_MAESTRO §4. El peligro #1 de un bot casero no es una mala estrategia: es
un bug operando sin control. Por eso ningún módulo llama al executor
directamente — toda Decision pasa primero por aquí, y aquí puede morir.

Responsabilidades:
    1. Vetos (límites duros y circuit breakers), en orden de gravedad.
    2. Position sizing por volatilidad (riesgo en dinero constante).
    3. Construcción de la Order con stop-loss OBLIGATORIO y take-profit por RR.

Diseño: el Risk Manager es un EVALUADOR sobre un snapshot del estado de la
cartera (`PortfolioState`), no el dueño de ese estado. La persistencia de
equity/peak/día la lleva el orquestador (Sprint 6). El único estado interno es
el kill switch, que LATCHA: una vez salta, requiere reset() manual — un kill
switch que se rearma solo no es un kill switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.core.config import Settings, load_settings
from src.core.models import Action, Decision, Order, Side


@dataclass
class PortfolioState:
    """Foto del estado de la cartera en el instante de evaluar una Decision."""

    equity: float            # equity realizada actual (USDT)
    peak_equity: float       # máximo histórico de equity (base del drawdown)
    day_start_equity: float  # equity al inicio del día UTC (base de pérdida diaria)
    open_positions: int      # posiciones abiertas ahora mismo
    feed_age_seconds: float = 0.0  # antigüedad del último precio (circuit breaker a)
    halted: bool = False     # parada manual / discrepancia de reconciliación (cb c)


@dataclass
class RiskAssessment:
    """Veredicto del Risk Manager: aprobar (con Order) o vetar (con motivo)."""

    approved: bool
    reason: str              # regla que disparó la decisión (auditoría)
    order: Order | None = None


class RiskManager:
    """Aplica límites de riesgo y, si todo pasa, dimensiona y arma la orden."""

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or load_settings()
        # Latch del kill switch: persiste entre llamadas hasta reset() manual.
        self.kill_switch_active = False

    def reset(self) -> None:
        """Rearmado manual del kill switch, tras revisión humana del drawdown."""
        self.kill_switch_active = False

    def assess(
        self,
        decision: Decision,
        *,
        price: float,
        atr: float,
        state: PortfolioState,
        confidence: float = 1.0,
    ) -> RiskAssessment:
        """Evalúa una Decision contra el estado de la cartera.

        Args:
            decision:   salida de la matriz de confluencia.
            price:      precio actual del activo (base del sizing y los stops).
            atr:        ATR(14) actual; fija la distancia al stop por volatilidad.
            state:      snapshot de la cartera (equity, posiciones, salud feed…).
            confidence: confianza del sentimiento [0,1]; baja → tamaño reducido.

        Returns:
            RiskAssessment.approved=True con una Order válida, o False con el
            motivo del veto.
        """
        r = self.cfg.risk

        # ---- Vetos, del más grave al menos grave (define la precedencia) ----

        # Kill switch por drawdown: latcha y bloquea hasta reset() manual. Se
        # evalúa primero porque, una vez disparado, nada más debería operar.
        if state.peak_equity > 0:
            drawdown = (state.peak_equity - state.equity) / state.peak_equity
            if drawdown >= r.max_drawdown_pct / 100.0:
                self.kill_switch_active = True
        if self.kill_switch_active:
            return RiskAssessment(False, "kill_switch_drawdown")

        # Discrepancia de reconciliación / parada manual (circuit breaker c).
        if state.halted:
            return RiskAssessment(False, "halted")

        # Feed de precios obsoleto (circuit breaker a): no abrir a ciegas.
        if state.feed_age_seconds > r.stale_feed_seconds:
            return RiskAssessment(False, "stale_feed")

        # Pérdida diaria: detiene nuevas entradas hasta el siguiente día UTC
        # (el orquestador resetea day_start_equity en el cambio de día).
        if state.day_start_equity > 0:
            daily_loss = (state.day_start_equity - state.equity) / state.day_start_equity
            if daily_loss >= r.max_daily_loss_pct / 100.0:
                return RiskAssessment(False, "daily_loss_limit")

        # Tope de posiciones simultáneas.
        if state.open_positions >= r.max_open_positions:
            return RiskAssessment(False, "max_positions")

        # ---- Nada que ejecutar si la confluencia dijo HOLD ----
        if decision.action == Action.HOLD:
            return RiskAssessment(False, "hold")

        # ---- Sizing por volatilidad ----
        stop_distance = r.atr_stop_multiplier * atr
        if stop_distance <= 0 or price <= 0:
            # ATR cero/negativo o precio inválido: no se puede colocar un stop
            # con sentido → no operar (mejor que dividir por cero).
            return RiskAssessment(False, "invalid_sizing_inputs")

        # qty = (equity × riesgo% × size_factor) / distancia_al_stop.
        # El riesgo en DINERO es constante; la cantidad se ajusta sola a la
        # volatilidad (más ATR ⇒ stop más lejos ⇒ menos cantidad).
        risk_amount = state.equity * (r.risk_per_trade_pct / 100.0) * decision.size_factor
        if confidence < r.low_confidence_threshold:
            risk_amount *= r.low_confidence_size_factor
        qty = risk_amount / stop_distance

        # Sin apalancamiento: el notional de la orden no excede la equity.
        qty = min(qty, state.equity / price)
        if qty <= 0:
            return RiskAssessment(False, "size_zero")

        # ---- Construcción de la orden (SL obligatorio, TP por RR) ----
        if decision.action == Action.LONG:
            side = Side.BUY
            stop = price - stop_distance
            tp = price + r.take_profit_rr * stop_distance
        else:  # SHORT (espejo)
            side = Side.SELL
            stop = price + stop_distance
            tp = price - r.take_profit_rr * stop_distance

        order = Order(
            symbol=decision.symbol,
            side=side,
            quantity=qty,
            entry_price=price,
            stop_loss=stop,
            take_profit=tp if tp > 0 else None,
            decision_reason=decision.reason,
            created_at=datetime.now(timezone.utc),
        )
        return RiskAssessment(True, "approved", order)
