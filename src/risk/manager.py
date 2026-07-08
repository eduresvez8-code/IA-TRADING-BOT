"""Risk Manager: el único módulo con poder de veto sobre TODA orden.

PLAN_MAESTRO §4. El peligro #1 de un bot casero no es una mala estrategia: es
un bug operando sin control. Por eso ningún módulo llama al executor
directamente — toda Decision pasa primero por aquí, y aquí puede morir.

Venue: **Binance Futuros USD-M** (testnet primero). Implicaciones de diseño:
    - Hay apalancamiento, pero el bot se AUTO-LIMITA a `max_leverage`. El
      apalancamiento NO cambia la cantidad (la fija el riesgo: 1% del wallet por
      el stop ATR); solo decide cuánto MARGEN inmoviliza el nocional
      (margen_inicial = nocional / leverage).
    - Los cortos son nativos y simétricos a los largos.
    - El "techo físico" de una apertura ya no es cash libre, sino que el
      **margen inicial requerido (nocional/L) ≤ available_balance**; y el margen
      agregado de la cartera ≤ `max_portfolio_margin_pct` del wallet_balance.
    - El Risk Manager es el ÚLTIMO filtro antes del executor: ajusta la orden a
      los filtros de microestructura (LOT_SIZE/PRICE_FILTER/MIN_NOTIONAL).

Saldos (terminología del exchange):
    - wallet_balance: colateral total de la cuenta de futuros, SIN PnL no
      realizado. Base del riesgo (1%), del drawdown (kill switch) y de la
      pérdida diaria.
    - available_balance: margen libre que el exchange reporta AHORA para abrir
      nuevas posiciones. Techo físico de la apertura.
    - committed_margin: margen inicial ya inmovilizado por las posiciones
      abiertas. Base del límite de margen agregado de la cartera.

Diseño: evaluador sobre un snapshot del estado (`PortfolioState`); no es dueño
del estado. La persistencia la lleva el orquestador en Sprint 6. El único estado
interno es el kill switch, que LATCHA: una vez salta, requiere reset() manual.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from src.core.config import Settings, load_settings
from src.core.models import Action, Decision, Order, PositionSide, Side, SymbolFilters
from src.risk.filters import floor_to_step, round_to_tick


@dataclass
class PortfolioState:
    """Foto de la cuenta de Futuros USD-M al evaluar una Decision."""

    wallet_balance: float    # colateral total SIN PnL no realizado (base de riesgo/DD)
    available_balance: float  # margen libre AHORA para abrir (techo físico)
    committed_margin: float  # margen inicial ya inmovilizado por lo abierto
    peak_wallet_balance: float    # máximo histórico del wallet (base del drawdown)
    day_start_wallet_balance: float  # wallet al inicio del día UTC (pérdida diaria)
    open_positions: int      # posiciones abiertas ahora mismo
    feed_age_seconds: float = 0.0  # antigüedad del último precio (circuit breaker a)
    halted: bool = False     # parada manual / discrepancia de reconciliación (cb c)
    long_positions: int = 0   # piernas LONG abiertas (cap de concentración direccional)
    short_positions: int = 0  # piernas SHORT abiertas (cap de concentración direccional)
    # Umbral de obsolescencia del feed ESCALADO AL TIMEFRAME, estampado por quien
    # posee el reloj (el orquestador: max(stale_feed_seconds, intervals×intervalo)).
    # Sin esto, el veto stale_feed comparaba contra el absoluto (30s) y, con vela
    # base de 1h, vetaba CUALQUIER trade de evento a mitad de vela (edad normal
    # entre velas ≈ 3600s > 30s): el Fast Path habría nacido muerto. None →
    # comportamiento legado (compara contra risk.stale_feed_seconds a secas).
    stale_after_seconds: float | None = None


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
        mode: Literal["slow", "event"] = "slow",
        atr_baseline: float | None = None,
    ) -> RiskAssessment:
        """Evalúa una Decision contra el estado de la cuenta y la microestructura.

        Args:
            decision:     salida de la confluencia (LONG/SHORT/HOLD).
            price:        precio actual (base del sizing y de los stops).
            atr:          ATR(14) actual; fija la distancia al stop.
            state:        snapshot de la cuenta de futuros.
            filters:      restricciones del par (tick/step/min) de exchangeInfo.
            confidence:   confianza del sentimiento [0,1]; baja → tamaño reducido.
            mode:         "slow" (Slow Path, por defecto) o "event" (Fast Path).
                          "event" usa parámetros propios (risk_pct y stop_mult más
                          conservadores) y aplica el amortiguador de expansión de
                          volatilidad (vol_damp).
            atr_baseline: mediana del ATR sobre vol_regime_lookback velas (línea
                          base del régimen). None → vol_damp=1.0 (sin recorte).

        Returns:
            RiskAssessment.approved=True con una Order válida, o False con el
            motivo del veto.
        """
        r = self.cfg.risk

        # ===== Vetos de estado (del más grave al menos grave) =====

        # Kill switch por drawdown sobre el WALLET: latcha hasta reset() manual.
        if state.peak_wallet_balance > 0:
            drawdown = (
                (state.peak_wallet_balance - state.wallet_balance)
                / state.peak_wallet_balance
            )
            if drawdown >= r.max_drawdown_pct / 100.0:
                self.kill_switch_active = True
        if self.kill_switch_active:
            return RiskAssessment(False, "kill_switch_drawdown")

        # Discrepancia de reconciliación / parada manual (circuit breaker c).
        if state.halted:
            return RiskAssessment(False, "halted")

        # Feed de precios obsoleto (circuit breaker a): no abrir a ciegas. El
        # umbral escala con el timeframe si el snapshot lo estampa (en vivo lo
        # hace el orquestador); si no, cae al absoluto de la config.
        stale_limit = (state.stale_after_seconds
                       if state.stale_after_seconds is not None
                       else r.stale_feed_seconds)
        if state.feed_age_seconds > stale_limit:
            return RiskAssessment(False, "stale_feed")

        # Pérdida diaria sobre el WALLET: detiene entradas hasta el día UTC siguiente.
        if state.day_start_wallet_balance > 0:
            daily_loss = (
                (state.day_start_wallet_balance - state.wallet_balance)
                / state.day_start_wallet_balance
            )
            if daily_loss >= r.max_daily_loss_pct / 100.0:
                return RiskAssessment(False, "daily_loss_limit")

        # Tope de posiciones simultáneas.
        if state.open_positions >= r.max_open_positions:
            return RiskAssessment(False, "max_positions")

        # ===== Naturaleza de la decisión =====

        if decision.action == Action.HOLD:
            return RiskAssessment(False, "hold")
        # En Futuros operamos LONG y SHORT de forma simétrica.
        is_long = decision.action == Action.LONG

        # Tope de concentración DIRECCIONAL: una noticia market-wide abriría, si no, los
        # 5 perps en el mismo sentido. Como la cripto está muy correlacionada, eso no son
        # 5 riesgos independientes sino ~1 apuesta direccional grande a beta. Capamos las
        # piernas simultáneas en la misma dirección (el símbolo a abrir aún no cuenta).
        same_dir = state.long_positions if is_long else state.short_positions
        if same_dir >= r.max_same_direction_positions:
            return RiskAssessment(False, "max_same_direction")

        # Veto por confianza del sentimiento: por debajo del piso, la noticia es
        # demasiado incierta para arriesgar capital (Claude dudando del titular).
        # Entre el piso y low_confidence_threshold el tamaño se reduce (paso 5); por
        # encima, tamaño pleno. El Fast Path ya filtra <event.min_confidence en
        # decide_event; este veto cubre el Slow Path (decide no mira la confianza).
        if confidence < r.min_confidence_to_trade:
            return RiskAssessment(False, "low_confidence")

        # ===== Pipeline de construcción de la orden =====

        # (1) Precio de entrada (orden a mercado: el fill esperado es el precio).
        entry = price
        if entry <= 0 or atr <= 0:
            return RiskAssessment(False, "invalid_sizing_inputs")

        # (1b) Selección de parámetros y amortiguador de volatilidad según el modo.
        #      mode="slow": parámetros del Slow Path, sin amortiguador.
        #      mode="event": parámetros propios (stop más ancho, riesgo menor) +
        #        vol_damp = min(1, cap/ratio). El stop ancho no sube el riesgo:
        #        qty = risk/stop → qty baja, riesgo en USD queda constante.
        if mode == "event":
            risk_pct = r.event_risk_per_trade_pct
            stop_mult = r.event_atr_stop_multiplier
            if atr_baseline is not None and atr_baseline > 0:
                vol_ratio = atr / atr_baseline
                vol_damp = min(1.0, r.vol_expansion_cap / vol_ratio)
            else:
                vol_damp = 1.0  # sin línea base: no recortamos (fallar-abierto al alza)
        else:
            risk_pct = r.risk_per_trade_pct
            stop_mult = r.atr_stop_multiplier
            vol_damp = 1.0

        # (2-3) SL/TP crudos por ATR (el stop va en el lado perdedor: bajo la
        #       entrada en LONG, sobre la entrada en SHORT) y ajuste al tickSize.
        stop_distance_raw = stop_mult * atr
        direction = 1.0 if is_long else -1.0
        stop_loss = float(
            round_to_tick(entry - direction * stop_distance_raw, filters.tick_size)
        )
        # "Dejar correr las ganancias": con let_winners_run, ningún techo fijo — el
        # stop (arriba) sigue acotando la pérdida, pero la salida ganadora queda solo
        # en manos del FLIP (señal revertida) o del time-stop. take_profit=None
        # atraviesa Order → translate.build_open_requests (ya soporta None: no arma
        # la orden TAKE_PROFIT_MARKET) → Executor (ya itera protectoras variables).
        take_profit = (
            None if r.let_winners_run else
            float(round_to_tick(
                entry + direction * r.take_profit_rr * stop_distance_raw,
                filters.tick_size,
            ))
        )

        # (4) Distancia REAL al stop, recalculada desde el SL ya redondeado, y
        #     verificación de que sigue del lado que protege.
        protective = (stop_loss < entry) if is_long else (stop_loss > entry)
        if not protective:
            # El tick es ≥ que la distancia: el SL redondeó hasta/sobre la entrada
            # y dejaría de proteger. No se puede operar este par así.
            return RiskAssessment(False, "stop_rounds_to_entry")
        stop_distance = abs(entry - stop_loss)

        # (5) Cantidad por RIESGO sobre el wallet. Los tres amortiguadores se apilan
        #     multiplicativamente: size_factor (Decision), low_confidence y vol_damp.
        #     vol_damp solo recorta (≤1.0), nunca amplifica: si el mercado está más
        #     tranquilo que la línea base, la posición NO se incrementa.
        risk_amount = (
            state.wallet_balance * (risk_pct / 100.0) * decision.size_factor
        )
        # Tramo de confianza media [min_confidence_to_trade, low_confidence_threshold):
        # opera pero a tamaño reducido. El veto duro (< min) ya salió antes; aquí solo
        # queda decidir pleno vs reducido.
        if confidence < r.low_confidence_threshold:
            risk_amount *= r.low_confidence_size_factor
        risk_amount *= vol_damp
        qty_risk = risk_amount / stop_distance

        # (6) Techos de MARGEN, expresados como tope de NOCIONAL (= margen × L).
        #     Físico:   margen_nuevo ≤ available_balance        → nocional ≤ avail·L
        #     Agregado: margen_comprometido + nuevo ≤ μ·wallet  → nocional ≤ (μ·wallet − comprometido)·L
        L = r.max_leverage
        margin_room = (
            state.wallet_balance * (r.max_portfolio_margin_pct / 100.0)
            - state.committed_margin
        )
        if margin_room <= 0:
            # La cartera ya está al/por encima del tope de margen agregado.
            return RiskAssessment(False, "portfolio_margin")
        cap_phys_notional = state.available_balance * L     # techo físico
        cap_policy_notional = margin_room * L               # techo agregado
        notional_cap = min(cap_phys_notional, cap_policy_notional)
        if notional_cap < filters.min_notional:
            # Ni la orden mínima de Binance cabe en el margen disponible.
            return RiskAssessment(False, "insufficient_margin")
        qty = min(qty_risk, notional_cap / entry)

        # (7) Truncar al stepSize (LOT_SIZE) — floor con Decimal, nunca arriba.
        qty_dec = floor_to_step(qty, filters.step_size)

        # (8-9) Validación de microestructura. Si la orden cae bajo el mínimo, se
        #       RECHAZA — jamás se infla (violaría el riesgo y el margen).
        if qty_dec <= 0 or qty_dec < filters.min_qty:
            return RiskAssessment(False, "below_min_qty")
        notional = qty_dec * Decimal(str(entry))
        if notional < filters.min_notional:
            return RiskAssessment(False, "below_min_notional")

        # (10) Construir la orden (BUY en LONG, SELL en SHORT) con su leverage y
        #      el cubo de hedge mode (positionSide) que el executor impondrá.
        order = Order(
            symbol=decision.symbol,
            side=Side.BUY if is_long else Side.SELL,
            quantity=float(qty_dec),
            entry_price=entry,
            stop_loss=stop_loss,
            # take_profit ya es None si let_winners_run; si no, el filtro >0 se
            # mantiene (defensivo: un valor no positivo por redondeo tampoco protege).
            take_profit=take_profit if (take_profit is not None and take_profit > 0) else None,
            leverage=L,
            position_side=PositionSide.LONG if is_long else PositionSide.SHORT,
            decision_reason=decision.reason,
            created_at=datetime.now(timezone.utc),
        )
        return RiskAssessment(True, "approved", order)
