"""Risk Manager: el único módulo con poder de veto sobre TODA orden.

PLAN_MAESTRO §4. El peligro #1 de un bot casero no es una mala estrategia: es
un bug operando sin control. Por eso ningún módulo llama al executor
directamente — toda Decision pasa primero por aquí, y aquí puede morir.

Venue: **Binance Spot, long-only**. Implicaciones que moldean el diseño:
    - No hay apalancamiento: comprometido + nuevo nunca puede exceder el capital.
      Por eso el techo físico se calcula sobre el SALDO LIBRE (free_balance), no
      sobre la equity total (que incluye lo ya inmovilizado en posiciones).
    - No se ABREN cortos: una Decision SHORT se veta (red de seguridad; la
      confluencia ya debería haberla convertido en HOLD).
    - El Risk Manager es el ÚLTIMO filtro antes del executor: ajusta la orden a
      los filtros de microestructura (LOT_SIZE/PRICE_FILTER/MIN_NOTIONAL) para
      que Binance no la rechace.

Diseño: evaluador sobre un snapshot del estado (`PortfolioState`); no es dueño
del estado. La persistencia (equity/peak/día/posiciones) la lleva el orquestador
en Sprint 6. El único estado interno es el kill switch, que LATCHA: una vez
salta, requiere reset() manual.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from src.core.config import Settings, load_settings
from src.core.models import Action, Decision, Order, Side, SymbolFilters
from src.risk.filters import floor_to_step, round_to_tick


@dataclass
class PortfolioState:
    """Foto de la cartera en el instante de evaluar una Decision (Spot)."""

    equity: float            # E: equity TOTAL (USDT libre + valor MtM abierto)
    free_balance: float      # F: USDT libre AHORA (techo físico de la orden)
    committed_notional: float  # C: suma de nocionales MtM de posiciones abiertas
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
        filters: SymbolFilters,
        confidence: float = 1.0,
    ) -> RiskAssessment:
        """Evalúa una Decision contra el estado de la cartera y la microestructura.

        Args:
            decision:   salida de la matriz de confluencia.
            price:      precio actual (base del sizing y de los stops).
            atr:        ATR(14) actual; fija la distancia al stop por volatilidad.
            state:      snapshot de la cartera (equity, free, comprometido…).
            filters:    restricciones del par (tick/step/min) de exchangeInfo.
            confidence: confianza del sentimiento [0,1]; baja → tamaño reducido.

        Returns:
            RiskAssessment.approved=True con una Order válida, o False con el
            motivo del veto.
        """
        r = self.cfg.risk

        # ===== Vetos de estado (del más grave al menos grave) =====

        # Kill switch por drawdown: latcha y bloquea hasta reset() manual.
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

        # Pérdida diaria: detiene nuevas entradas hasta el siguiente día UTC.
        if state.day_start_equity > 0:
            daily_loss = (state.day_start_equity - state.equity) / state.day_start_equity
            if daily_loss >= r.max_daily_loss_pct / 100.0:
                return RiskAssessment(False, "daily_loss_limit")

        # Tope de posiciones simultáneas.
        if state.open_positions >= r.max_open_positions:
            return RiskAssessment(False, "max_positions")

        # ===== Naturaleza de la decisión =====

        if decision.action == Action.HOLD:
            return RiskAssessment(False, "hold")

        # Spot: jamás ABRIR un corto. Red de seguridad — la confluencia ya
        # debería haberlo convertido en HOLD ("short_disabled_spot").
        if decision.action == Action.SHORT:
            return RiskAssessment(False, "short_not_allowed_spot")

        # ===== Pipeline de construcción de la orden (LONG / BUY) =====

        # (1) Precio de entrada (orden a mercado: el fill esperado es el precio).
        entry = price
        if entry <= 0 or atr <= 0:
            # ATR/precio inválidos: no se puede colocar un stop con sentido.
            return RiskAssessment(False, "invalid_sizing_inputs")

        # (2-3) SL/TP crudos por ATR, ajustados al tickSize del par.
        stop_distance_raw = r.atr_stop_multiplier * atr
        stop_loss = float(round_to_tick(entry - stop_distance_raw, filters.tick_size))
        take_profit = float(
            round_to_tick(entry + r.take_profit_rr * stop_distance_raw, filters.tick_size)
        )

        # (4) Distancia REAL al stop, recalculada desde el SL ya redondeado: así
        #     el sizing corresponde al stop que de verdad se coloca.
        stop_distance = entry - stop_loss
        if stop_distance <= 0:
            # El tick es ≥ que la distancia al stop: redondeó hasta/sobre la
            # entrada y dejaría de proteger. No se puede operar este par así.
            return RiskAssessment(False, "stop_rounds_to_entry")

        # (5) Cantidad por RIESGO (sobre equity, para mantener el 1% constante).
        risk_amount = state.equity * (r.risk_per_trade_pct / 100.0) * decision.size_factor
        if confidence < r.low_confidence_threshold:
            risk_amount *= r.low_confidence_size_factor
        qty_risk = risk_amount / stop_distance

        # (6) Techos en NOCIONAL: físico (saldo libre) y de política (exposición
        #     agregada). λ = max_portfolio_exposure_pct; el (1-λ) es el colchón.
        lam = r.max_portfolio_exposure_pct / 100.0
        cap_free = lam * state.free_balance                          # físico
        cap_policy = lam * state.equity - state.committed_notional   # agregado
        if cap_policy <= 0:
            # Ya estamos al/por encima del tope de exposición de la cartera.
            return RiskAssessment(False, "portfolio_exposure")
        notional_cap = min(cap_free, cap_policy)
        if notional_cap < filters.min_notional:
            # Ni la orden mínima de Binance cabe en el cash/política disponible.
            return RiskAssessment(False, "insufficient_free_balance")
        qty = min(qty_risk, notional_cap / entry)

        # (7) Truncar al stepSize (LOT_SIZE) — floor con Decimal, nunca arriba.
        qty_dec = floor_to_step(qty, filters.step_size)

        # (8-9) Validación de microestructura. Si la orden cae bajo el mínimo, se
        #       RECHAZA — jamás se infla, eso violaría el riesgo y el saldo libre.
        if qty_dec <= 0 or qty_dec < filters.min_qty:
            return RiskAssessment(False, "below_min_qty")
        notional = qty_dec * Decimal(str(entry))
        if notional < filters.min_notional:
            return RiskAssessment(False, "below_min_notional")

        # (10) Construir la orden (siempre BUY en Spot long-only).
        order = Order(
            symbol=decision.symbol,
            side=Side.BUY,
            quantity=float(qty_dec),
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit if take_profit > 0 else None,
            decision_reason=decision.reason,
            created_at=datetime.now(timezone.utc),
        )
        return RiskAssessment(True, "approved", order)
