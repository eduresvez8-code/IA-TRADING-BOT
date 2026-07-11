"""Risk Manager: el único módulo con poder de veto sobre TODA orden.

El peligro #1 de un bot casero no es una mala estrategia: es un bug operando
sin control. Por eso ningún módulo llamará a un broker directamente — toda
Decision pasa primero por aquí, y aquí puede morir.

Contexto: **acciones EE.UU., cuenta cash** (pivote 2026-07-11). Implicaciones
frente a la versión de futuros cripto:
    - SIN apalancamiento: el notional de una compra no puede exceder el cash
      disponible. No hay margen, no hay funding, no hay liquidación.
    - SIN cortos por defecto (`risk.allow_short=false`): shortear acciones
      exige cuenta de margen + costo de préstamo, no modelable con datos gratis.
    - Acciones ENTERAS salvo `allow_fractional_shares=true`: la cantidad se
      trunca hacia abajo (floor) — truncar jamás infla el riesgo; redondear
      hacia arriba sí lo haría.

Lo que NO cambió (filosofía universal, probada en 4 meses de investigación):
    - Sizing por riesgo fijo: qty = (equity × riesgo%) / distancia_al_stop.
      El riesgo en dinero es constante aunque la volatilidad cambie.
    - Circuit breakers: pérdida diaria (pausa hasta mañana) y drawdown máximo
      (kill switch que LATCHA: una vez salta, requiere reset() manual).
    - Stop-loss obligatorio: una orden sin SL no existe en este sistema.

Diseño: evaluador puro sobre un snapshot (`PortfolioState`); no es dueño del
estado. El único estado interno es el kill switch, que latcha.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from src.core.config import Settings, load_settings
from src.core.models import Action, Decision, Order, Side


@dataclass
class PortfolioState:
    """Foto de la cuenta al evaluar una Decision."""

    equity: float                 # valor total de la cuenta (cash + posiciones a mercado)
    cash_available: float         # efectivo libre para comprar (techo físico, sin margen)
    peak_equity: float            # máximo histórico del equity (base del drawdown)
    day_start_equity: float       # equity al inicio del día (base de la pérdida diaria)
    open_positions: int           # posiciones abiertas ahora mismo
    halted: bool = False          # parada manual / estado inconsistente


@dataclass
class RiskAssessment:
    """Veredicto del Risk Manager: aprobar (con Order) o vetar (con motivo)."""

    approved: bool
    reason: str                   # regla que disparó la decisión (auditoría)
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
    ) -> RiskAssessment:
        """Evalúa una Decision contra el estado de la cuenta.

        Args:
            decision: salida de la estrategia (LONG/SHORT/HOLD + size_factor).
            price:    precio actual (base del sizing y de los stops).
            atr:      ATR actual; fija la distancia al stop.
            state:    snapshot de la cuenta.

        Returns:
            RiskAssessment.approved=True con una Order válida, o False con el
            motivo del veto.
        """
        r = self.cfg.risk

        # ===== Vetos de estado (del más grave al menos grave) =====

        # Kill switch por drawdown sobre el equity: latcha hasta reset() manual.
        if state.peak_equity > 0:
            drawdown = (state.peak_equity - state.equity) / state.peak_equity
            if drawdown >= r.max_drawdown_pct / 100.0:
                self.kill_switch_active = True
        if self.kill_switch_active:
            return RiskAssessment(False, "kill_switch_drawdown")

        # Parada manual / estado inconsistente.
        if state.halted:
            return RiskAssessment(False, "halted")

        # Pérdida diaria: detiene entradas hasta el día siguiente.
        if state.day_start_equity > 0:
            daily_loss = (
                (state.day_start_equity - state.equity) / state.day_start_equity
            )
            if daily_loss >= r.max_daily_loss_pct / 100.0:
                return RiskAssessment(False, "daily_loss_limit")

        # Tope de posiciones simultáneas.
        if state.open_positions >= r.max_open_positions:
            return RiskAssessment(False, "max_positions")

        # ===== Naturaleza de la decisión =====

        if decision.action == Action.HOLD:
            return RiskAssessment(False, "hold")
        is_long = decision.action == Action.LONG
        if not is_long and not r.allow_short:
            # Cuenta cash: los cortos están prohibidos por política.
            return RiskAssessment(False, "short_not_allowed")

        # ===== Pipeline de construcción de la orden =====

        # (1) Entradas de sizing válidas (orden a mercado: fill esperado = price).
        entry = price
        if entry <= 0 or atr <= 0 or math.isnan(atr):
            return RiskAssessment(False, "invalid_sizing_inputs")

        # (2) Stop por ATR en el lado perdedor; TP solo si hay techo fijo.
        direction = 1.0 if is_long else -1.0
        stop_distance = r.atr_stop_multiplier * atr
        stop_loss = entry - direction * stop_distance
        if stop_loss <= 0:
            # Precio bajo + stop ancho: el SL caería a ≤0 (sin sentido en acciones).
            return RiskAssessment(False, "stop_below_zero")
        take_profit = (
            None if r.let_winners_run
            else entry + direction * r.take_profit_rr * stop_distance
        )

        # (3) Cantidad por RIESGO sobre el equity, escalada por la convicción.
        risk_amount = state.equity * (r.risk_per_trade_pct / 100.0) * decision.size_factor
        qty = risk_amount / stop_distance

        # (4) Techo físico SIN apalancamiento: el notional no excede el cash libre.
        if entry > 0:
            qty = min(qty, state.cash_available / entry)

        # (5) Acciones enteras salvo fraccionales: floor, jamás redondear arriba
        #     (inflaría el riesgo por encima del presupuesto).
        if not r.allow_fractional_shares:
            qty = float(math.floor(qty))
        if qty <= 0:
            return RiskAssessment(False, "below_min_qty")

        # (6) Construir la orden (BUY en LONG, SELL en SHORT).
        order = Order(
            symbol=decision.symbol,
            side=Side.BUY if is_long else Side.SELL,
            quantity=qty,
            entry_price=entry,
            stop_loss=stop_loss,
            # Defensivo: un TP no positivo por aritmética extraña tampoco protege.
            take_profit=take_profit if (take_profit is not None and take_profit > 0) else None,
            decision_reason=decision.reason,
            created_at=datetime.now(timezone.utc),
        )
        return RiskAssessment(True, "approved", order)
