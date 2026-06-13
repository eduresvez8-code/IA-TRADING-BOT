"""Métricas de evaluación de un backtest: funciones puras.

Igual que src/quant/indicators.py, este módulo es 100% funcional: cada función
recibe arrays/listas y devuelve un número, sin estado ni I/O. Eso permite
testearlas con valores de referencia calculados a mano.

Convención de unidades:
    - retornos y drawdown se devuelven como FRACCIÓN (0.10 = 10%), no en %.
    - Sharpe/Sortino se devuelven ANUALIZADOS.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

# Minutos por año (365 días). Base para anualizar a partir del timeframe.
_MINUTES_PER_YEAR = 365 * 24 * 60


def bars_per_year(timeframe: str) -> float:
    """Cuántas velas de `timeframe` caben en un año.

    Anualizar Sharpe/Sortino requiere saber cuántas observaciones hay por año.
    timeframe es la notación de Binance: '1m', '5m', '15m', '1h', '4h', '1d'.

        '5m' → 525600 / 5   = 105120
        '1h' → 525600 / 60  = 8760
        '1d' → 525600 / 1440 = 365
    """
    m = re.fullmatch(r"(\d+)([mhd])", timeframe.strip())
    if not m:
        raise ValueError(f"timeframe no reconocido: {timeframe!r}")
    n, unit = int(m.group(1)), m.group(2)
    minutes = n * {"m": 1, "h": 60, "d": 1440}[unit]
    return _MINUTES_PER_YEAR / minutes


def total_return(equity_curve) -> float:
    """Retorno total: E_final / E_inicial - 1."""
    e = np.asarray(equity_curve, dtype=float)
    if len(e) < 2 or e[0] == 0:
        return 0.0
    return float(e[-1] / e[0] - 1.0)


def cagr(equity_curve, timeframe: str) -> float:
    """Retorno anualizado compuesto: (E_n/E_0)^(1/años) - 1.

    Si la equity final es ≤ 0 (ruina total) no tiene sentido la raíz: -100%.
    """
    e = np.asarray(equity_curve, dtype=float)
    if len(e) < 2 or e[0] <= 0:
        return 0.0
    years = (len(e) - 1) / bars_per_year(timeframe)
    if years <= 0:
        return 0.0
    if e[-1] <= 0:
        return -1.0
    # math.pow con floats nativos lanza OverflowError (en vez del inf silencioso
    # de numpy) si anualizamos un retorno enorme sobre un período minúsculo.
    try:
        return math.pow(float(e[-1]) / float(e[0]), 1.0 / years) - 1.0
    except OverflowError:
        return math.inf


def bar_returns(equity_curve) -> np.ndarray:
    """Retornos simples por barra: r_t = E_t/E_{t-1} - 1."""
    e = np.asarray(equity_curve, dtype=float)
    if len(e) < 2:
        return np.array([])
    return e[1:] / e[:-1] - 1.0


def sharpe_ratio(equity_curve, timeframe: str, risk_free: float = 0.0) -> float:
    """Sharpe anualizado: (mean(r) - rf) / std(r) * sqrt(bars_por_año).

    Mide retorno por unidad de riesgo TOTAL (volatilidad). std poblacional
    (ddof=0): en backtesting tratamos la serie como la población observada,
    no una muestra. Si no hay varianza (equity plana o constante), Sharpe es 0
    por convención — no hay riesgo, pero tampoco información.
    """
    r = bar_returns(equity_curve)
    if len(r) == 0:
        return 0.0
    sd = r.std(ddof=0)
    if sd == 0:
        return 0.0
    excess = r.mean() - risk_free
    return float(excess / sd * math.sqrt(bars_per_year(timeframe)))


def sortino_ratio(equity_curve, timeframe: str, risk_free: float = 0.0) -> float:
    """Sortino anualizado: como Sharpe pero con desviación SOLO a la baja.

    σ_down = sqrt(mean(min(r,0)^2)). Penaliza únicamente la volatilidad que
    duele (caídas); la volatilidad al alza no se castiga. Si nunca hubo
    retornos negativos, no hay downside → inf (lo reportamos como tal).
    """
    r = bar_returns(equity_curve)
    if len(r) == 0:
        return 0.0
    downside = np.minimum(r, 0.0)
    dd = math.sqrt(np.mean(downside ** 2))
    excess = r.mean() - risk_free
    if dd == 0:
        return math.inf if excess > 0 else 0.0
    return float(excess / dd * math.sqrt(bars_per_year(timeframe)))


def max_drawdown(equity_curve) -> float:
    """Máxima caída desde un pico, como fracción positiva.

    Para cada t: drawdown_t = (pico_hasta_t - E_t) / pico_hasta_t.
    Devuelve el peor (mayor) de todos. 0.25 = se perdió el 25% desde un máximo.
    """
    e = np.asarray(equity_curve, dtype=float)
    if len(e) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(e)
    # Evita /0 si la equity arrancara en 0 (no debería con capital inicial > 0).
    drawdowns = np.where(running_peak > 0, (running_peak - e) / running_peak, 0.0)
    return float(drawdowns.max())


def win_rate(trade_pnls) -> float:
    """Fracción de trades con PnL neto > 0."""
    p = np.asarray(trade_pnls, dtype=float)
    if len(p) == 0:
        return 0.0
    return float((p > 0).sum() / len(p))


def profit_factor(trade_pnls) -> float:
    """Ganancia bruta / pérdida bruta (en valor absoluto).

    >1 = rentable. inf si no hubo pérdidas (con al menos una ganancia).
    """
    p = np.asarray(trade_pnls, dtype=float)
    gross_profit = p[p > 0].sum()
    gross_loss = -p[p < 0].sum()
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


@dataclass
class BacktestMetrics:
    """Resumen cuantitativo de un backtest. Todo en fracciones, no en %."""

    n_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    exposure: float          # fracción de barras con posición abierta
    avg_win: float           # PnL medio de los trades ganadores (en moneda)
    avg_loss: float          # PnL medio de los trades perdedores (en moneda)
    expectancy: float        # PnL medio por trade (en moneda)
    avg_bars_held: float     # duración media de un trade, en velas


def compute_metrics(
    equity_curve,
    trade_pnls,
    bars_held,
    bars_in_market: int,
    timeframe: str,
) -> BacktestMetrics:
    """Compone todas las métricas a partir de la curva de equity y los trades.

    Args:
        equity_curve:  valor del capital marcado a mercado por barra.
        trade_pnls:    PnL neto (tras costos) de cada trade cerrado.
        bars_held:     nº de velas que duró cada trade (mismo orden que pnls).
        bars_in_market: total de velas con una posición abierta.
        timeframe:     '5m', '1h'… para anualizar.
    """
    pnls = np.asarray(trade_pnls, dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total_bars = max(len(np.asarray(equity_curve)), 1)

    return BacktestMetrics(
        n_trades=len(pnls),
        win_rate=win_rate(pnls),
        profit_factor=profit_factor(pnls),
        total_return=total_return(equity_curve),
        cagr=cagr(equity_curve, timeframe),
        sharpe=sharpe_ratio(equity_curve, timeframe),
        sortino=sortino_ratio(equity_curve, timeframe),
        max_drawdown=max_drawdown(equity_curve),
        exposure=bars_in_market / total_bars,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        expectancy=float(pnls.mean()) if len(pnls) else 0.0,
        avg_bars_held=float(np.mean(bars_held)) if len(bars_held) else 0.0,
    )
