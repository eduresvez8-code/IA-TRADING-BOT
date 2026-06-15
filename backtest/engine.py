"""Motor de backtesting: simula la estrategia barra a barra sobre histórico.

Principios de honestidad (anti sesgo de anticipación / look-ahead bias):
    1. La DECISIÓN se toma con el cierre de la vela t (score y ATR causales).
    2. La EJECUCIÓN (entrada/salida por señal) ocurre a la APERTURA de t+1.
       Nunca rellenamos una orden al mismo precio que usamos para decidir.
    3. Los stops/take-profit se vigilan con el high/low de las velas POSTERIORES
       a la entrada, jamás con la vela que generó la señal.

Costos: comisión y slippage se aplican en CADA lado (entrada y salida).

NOTA: el sizing y la colocación de stops son una versión MÍNIMA embebida.
La versión definitiva (límites duros, circuit breakers, kill switch) vive en
risk/manager.py a partir del Sprint 5. Aquí solo se simula para evaluar la
estrategia cuantitativa de forma realista.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.core.config import Settings, load_settings
from src.quant.strategy import compute_signal_series
from src.quant.indicators import atr

from backtest.metrics import BacktestMetrics, compute_metrics


@dataclass
class Trade:
    """Un trade cerrado, con todo lo necesario para auditarlo."""

    symbol: str
    side: str                # "LONG" | "SHORT"
    entry_time: datetime
    entry_price: float       # precio de relleno (ya incluye slippage)
    exit_time: datetime
    exit_price: float        # precio de relleno (ya incluye slippage)
    quantity: float
    gross_pnl: float         # PnL antes de comisiones
    commission: float        # comisión total (entrada + salida)
    net_pnl: float           # PnL tras comisiones (lo que mueve la equity)
    return_pct: float        # net_pnl / notional de entrada
    bars_held: int
    exit_reason: str         # "stop_loss" | "take_profit" | "signal" | "end_of_data"


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    initial_capital: float
    final_equity: float
    equity_curve: pd.Series          # marcada a mercado, indexada por tiempo
    trades: list[Trade] = field(default_factory=list)
    metrics: BacktestMetrics | None = None


class BacktestEngine:
    """Simula la estrategia ema_cross_rsi sobre un DataFrame de velas."""

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or load_settings()

    def run(self, df: pd.DataFrame, symbol: str, timeframe: str,
            *, decider=None) -> BacktestResult:
        """Simula la estrategia barra a barra.

        `decider(i, position_side, score, ts)` decide la acción de cada vela y
        devuelve ("enter", side, size_factor) | ("exit",) | None. Por defecto usa
        la estrategia por umbrales del Sprint 3 (comportamiento intacto); la ruta
        de confluencia (Sprint C.2) inyecta su propio decider con sentimiento.
        """
        bt = self.cfg.backtest
        atr_mult = self.cfg.risk.atr_stop_multiplier
        risk_pct = self.cfg.risk.risk_per_trade_pct
        comm = bt.commission_pct / 100.0     # de % a fracción
        base_slip = bt.slippage_pct / 100.0  # slippage fijo por lado
        slip_k = bt.slippage_atr_multiplier  # componente dinámico: k·ATR/precio

        df = df.reset_index(drop=True)
        n = len(df)

        # Indicadores precalculados de una sola pasada (causales → sin look-ahead).
        scores = compute_signal_series(df).to_numpy()
        atrs = atr(df, self.cfg.risk.atr_period).to_numpy()

        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        times = df["open_time"].tolist() if "open_time" in df.columns else list(range(n))

        cash = bt.initial_capital            # equity realizada
        position: dict | None = None         # None = plano
        pending: tuple | None = None         # acción a ejecutar en la apertura siguiente
        equity_list: list[float] = []
        bars_in_market = 0
        trades: list[Trade] = []

        def slip_at(bar_idx: int, price: float) -> float:
            """Slippage efectivo en una vela: fijo + componente por volatilidad.

            slip = base_slip + slip_k · ATR_vela / precio.  Con slip_k = 0 se
            reduce EXACTAMENTE al slippage fijo original (regresión protegida).
            El ATR se normaliza por el precio para que slip quede en fracción.
            """
            a = atrs[bar_idx]
            if price <= 0 or pd.isna(a):
                return base_slip
            return base_slip + slip_k * a / price

        def close_position(exit_price: float, bar_idx: int, reason: str) -> None:
            nonlocal cash, position
            p = position
            qty = p["qty"]
            s = slip_at(bar_idx, exit_price)
            if p["side"] == "LONG":
                fill = exit_price * (1 - s)             # vender: peor precio
                gross = qty * (fill - p["entry_price"])
            else:  # SHORT: cerrar = comprar
                fill = exit_price * (1 + s)
                gross = qty * (p["entry_price"] - fill)
            exit_comm = qty * fill * comm
            cash += gross - exit_comm
            net = gross - p["entry_commission"] - exit_comm
            notional = qty * p["entry_price"]
            trades.append(Trade(
                symbol=symbol, side=p["side"],
                entry_time=p["entry_time"], entry_price=p["entry_price"],
                exit_time=times[bar_idx], exit_price=fill, quantity=qty,
                gross_pnl=gross, commission=p["entry_commission"] + exit_comm,
                net_pnl=net, return_pct=(net / notional if notional else 0.0),
                bars_held=bar_idx - p["entry_bar"], exit_reason=reason,
            ))
            position = None

        if decider is None:
            # Estrategia por umbrales del Sprint 3 (comportamiento de referencia).
            def decider(i, position_side, score, ts):
                if position_side is None:
                    if score >= bt.entry_threshold:
                        return ("enter", "LONG", 1.0)
                    if score <= -bt.entry_threshold and bt.allow_short:
                        return ("enter", "SHORT", 1.0)
                    return None
                if position_side == "LONG" and score <= bt.exit_threshold:
                    return ("exit",)
                if position_side == "SHORT" and score >= -bt.exit_threshold:
                    return ("exit",)
                return None

        for i in range(n):
            # ---- 1. Ejecutar acción pendiente a la APERTURA de esta vela ----
            if pending is not None:
                kind = pending[0]
                if kind == "enter" and position is None:
                    side, atr_at_decision, size_factor = pending[1], pending[2], pending[3]
                    stop_px, tp_px = pending[4], pending[5]
                    s = slip_at(i, opens[i])
                    entry = opens[i] * (1 + s) if side == "LONG" else opens[i] * (1 - s)
                    if stop_px is None:
                        # Defecto (Sprint 3): stop a ATR de la decisión, TP por RR.
                        stop_distance = atr_mult * atr_at_decision
                        if side == "LONG":
                            stop = entry - stop_distance
                            tp = entry + bt.take_profit_rr * stop_distance
                        else:
                            stop = entry + stop_distance
                            tp = entry - bt.take_profit_rr * stop_distance
                    else:
                        # Stop/TP explícitos del decider: niveles de precio fijados con
                        # datos causales de la vela de decisión (canal de Donchian, etc.).
                        # tp_px puede ser None → salir solo por señal o por stop.
                        stop, tp = stop_px, tp_px
                        stop_distance = abs(entry - stop)
                    # El stop debe quedar en el lado correcto; si no, no se abre.
                    valid_stop = ((side == "LONG" and stop < entry)
                                  or (side == "SHORT" and stop > entry))
                    if stop_distance > 0 and valid_stop:
                        equity_now = cash  # plano: equity == cash
                        # El size_factor de la confluencia escala el riesgo del trade.
                        risk_amount = equity_now * risk_pct / 100.0 * size_factor
                        qty = risk_amount / stop_distance
                        # Sin apalancamiento: el notional no excede la equity.
                        qty = min(qty, equity_now / entry) if entry > 0 else 0.0
                        if qty > 0:
                            entry_comm = qty * entry * comm
                            cash -= entry_comm
                            position = {
                                "side": side, "entry_price": entry, "qty": qty,
                                "stop": stop, "tp": tp, "entry_time": times[i],
                                "entry_bar": i, "entry_commission": entry_comm,
                            }
                elif kind == "exit" and position is not None:
                    close_position(opens[i], i, "signal")
                pending = None

            # ---- 2. Vigilar stop/TP con ejecución en GAP ----
            # Si la vela ABRE ya cruzada respecto al nivel, el mercado nunca
            # cotizó ese nivel: el fill realista es el `open`, no el nivel.
            # En contra (stop) el open es peor → pesimista; a favor (TP) el open
            # es mejor → justo. El orden del elif preserva "stop antes que TP"
            # cuando ambos se tocan intrabar en la misma vela.
            if position is not None:
                stop = position["stop"]
                tp = position["tp"]
                # tp puede ser None (arquetipo de reversión: salida solo por señal).
                if position["side"] == "LONG":
                    if opens[i] <= stop:                  # gap EN CONTRA: abrió bajo el stop
                        close_position(opens[i], i, "stop_loss")
                    elif lows[i] <= stop:                 # tocado intrabar → al nivel
                        close_position(stop, i, "stop_loss")
                    elif tp is not None and opens[i] >= tp:   # gap A FAVOR: abrió sobre el TP
                        close_position(opens[i], i, "take_profit")
                    elif tp is not None and highs[i] >= tp:   # tocado intrabar → al nivel
                        close_position(tp, i, "take_profit")
                else:  # SHORT (espejo)
                    if opens[i] >= stop:                  # gap EN CONTRA: abrió sobre el stop
                        close_position(opens[i], i, "stop_loss")
                    elif highs[i] >= stop:
                        close_position(stop, i, "stop_loss")
                    elif tp is not None and opens[i] <= tp:   # gap A FAVOR: abrió bajo el TP
                        close_position(opens[i], i, "take_profit")
                    elif tp is not None and lows[i] <= tp:
                        close_position(tp, i, "take_profit")

            # ---- 3. Marcar a mercado: equity con PnL no realizado al cierre ----
            if position is not None:
                if position["side"] == "LONG":
                    unrealized = position["qty"] * (closes[i] - position["entry_price"])
                else:
                    unrealized = position["qty"] * (position["entry_price"] - closes[i])
                equity_list.append(cash + unrealized)
                bars_in_market += 1
            else:
                equity_list.append(cash)

            # ---- 4. Decidir acción para la vela SIGUIENTE (con cierre de t) ----
            if pd.isna(scores[i]) or pd.isna(atrs[i]):
                continue
            dec = decider(i, position["side"] if position else None, scores[i], times[i])
            if dec is None:
                continue
            if dec[0] == "enter" and position is None:
                # ("enter", side, size_factor[, stop_px, tp_px]). Sin stop/tp
                # explícitos → None → el motor usa el stop ATR + TP por RR.
                stop_px = dec[3] if len(dec) > 3 else None
                tp_px = dec[4] if len(dec) > 4 else None
                pending = ("enter", dec[1], atrs[i], dec[2], stop_px, tp_px)
            elif dec[0] == "exit" and position is not None:
                pending = ("exit",)

        # ---- Cierre forzado si quedó una posición abierta al final ----
        if position is not None:
            close_position(closes[n - 1], n - 1, "end_of_data")
            equity_list[-1] = cash  # la última equity refleja el cierre realizado

        idx = df["open_time"] if "open_time" in df.columns else pd.RangeIndex(n)
        equity_curve = pd.Series(equity_list, index=idx, name="equity")

        metrics = compute_metrics(
            equity_curve=equity_list,
            trade_pnls=[t.net_pnl for t in trades],
            bars_held=[t.bars_held for t in trades],
            bars_in_market=bars_in_market,
            timeframe=timeframe,
        )

        return BacktestResult(
            symbol=symbol, timeframe=timeframe,
            initial_capital=bt.initial_capital, final_equity=cash,
            equity_curve=equity_curve, trades=trades, metrics=metrics,
        )
