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

    def run(self, df: pd.DataFrame, symbol: str, timeframe: str) -> BacktestResult:
        bt = self.cfg.backtest
        atr_mult = self.cfg.risk.atr_stop_multiplier
        risk_pct = self.cfg.risk.risk_per_trade_pct
        comm = bt.commission_pct / 100.0     # de % a fracción
        slip = bt.slippage_pct / 100.0

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

        def close_position(exit_price: float, bar_idx: int, reason: str) -> None:
            nonlocal cash, position
            p = position
            qty = p["qty"]
            if p["side"] == "LONG":
                fill = exit_price * (1 - slip)          # vender: peor precio
                gross = qty * (fill - p["entry_price"])
            else:  # SHORT: cerrar = comprar
                fill = exit_price * (1 + slip)
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

        for i in range(n):
            # ---- 1. Ejecutar acción pendiente a la APERTURA de esta vela ----
            if pending is not None:
                kind = pending[0]
                if kind == "enter" and position is None:
                    side, atr_at_decision = pending[1], pending[2]
                    stop_distance = atr_mult * atr_at_decision
                    if stop_distance > 0:
                        if side == "LONG":
                            entry = opens[i] * (1 + slip)   # comprar: peor precio
                            stop = entry - stop_distance
                            tp = entry + bt.take_profit_rr * stop_distance
                        else:
                            entry = opens[i] * (1 - slip)   # vender en corto
                            stop = entry + stop_distance
                            tp = entry - bt.take_profit_rr * stop_distance
                        equity_now = cash  # plano: equity == cash
                        risk_amount = equity_now * risk_pct / 100.0
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

            # ---- 2. Vigilar stop/TP intrabar (con high/low de ESTA vela) ----
            if position is not None:
                if position["side"] == "LONG":
                    if lows[i] <= position["stop"]:          # stop primero (pesimista)
                        close_position(position["stop"], i, "stop_loss")
                    elif highs[i] >= position["tp"]:
                        close_position(position["tp"], i, "take_profit")
                else:  # SHORT
                    if highs[i] >= position["stop"]:
                        close_position(position["stop"], i, "stop_loss")
                    elif lows[i] <= position["tp"]:
                        close_position(position["tp"], i, "take_profit")

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
            score = scores[i]
            a = atrs[i]
            if pd.isna(score) or pd.isna(a):
                continue
            if position is None:
                if score >= bt.entry_threshold:
                    pending = ("enter", "LONG", a)
                elif score <= -bt.entry_threshold and bt.allow_short:
                    pending = ("enter", "SHORT", a)
            else:
                # Salida por debilidad/giro de la señal.
                if position["side"] == "LONG" and score <= bt.exit_threshold:
                    pending = ("exit",)
                elif position["side"] == "SHORT" and score >= -bt.exit_threshold:
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
