"""Familia C — Reversión a VWAP intradía (microestructura, 5m).

Hipótesis: el precio revierte hacia el VWAP intradía anclado a medianoche UTC.
Los algos de ejecución institucional (TWAP/VWAP) anclan al VWAP del día → cuando
el precio se aleja demasiado, hay presión de reversión. A diferencia de la
Familia B (cointegración de pares), el VWAP RESETEA cada día → la señal NO sufre
el problema I(1) (random walk) que mató a los pares; vive en escala intradía.

VWAP intradía anclado:
    tp_t   = (high_t + low_t + close_t) / 3          (precio típico)
    VWAP_t = Σ_{día}(tp·vol) / Σ_{día}(vol)          (acumulado DENTRO del día UTC)
    dev_t  = (close_t − VWAP_t) / VWAP_t              (desviación relativa)
    z_t    = (dev_t − μ_rolling) / σ_rolling          (z-score con ventana rolling)

Robustez al bid-ask bounce (CRÍTICO):
    El IC inmediato (entrar en close_t) está contaminado por el rebote bid/ask:
    el close salta entre bid y ask → reversión MECÁNICA de 1 barra que NO es
    capturable (pagarías el spread). El probe de auditoría confirmó que el edge
    SOBREVIVE saltando la barra inmediata (entrar en close_{t+1}). Por eso la
    señal de trading usa `z.shift(1)`: se observa en t, se actúa en close_{t+1}.
    Esto es lookahead-free Y robusto al bounce a la vez.

Embudo de 2 etapas:
    ETAPA 1 — IC de Spearman(z_t, ret_{t+1 → t+1+h}) con corrección n_eff.
              h = vwap_forward_horizon (el horizonte de la reversión, no h=1:
              la reversión es multi-barra, gatearla en h=1 la subestima).
    ETAPA 2 — P&L de la máquina de estados con costos REALES (taker + slippage
              %+k·ATR), cierre forzado en el límite del día (el VWAP resetea).

Una sesión = un módulo. Reemplaza el stub run_family_volume de quant_matrix.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.metrics import max_drawdown, profit_factor
from src.core.config import QuantMatrixConfig

# Barras de 5m por año: 365 días × 24 h × 12 (5m) = 105120.
BARS_PER_YEAR_5M = 365 * 24 * 12


@dataclass
class VwapStats:
    """Resultado del simulador de reversión a VWAP para un activo."""
    symbol: str
    n_periods: int               # barras válidas (post-warmup del z-score)
    ic_spearman: float           # IC(z_t, ret_{+h}), esperado < 0 (reversión)
    ic_tstat: float              # t-stat corregido por autocorr (n_eff)
    n_trades: int
    net_return_ann_pct: float    # retorno neto anualizado sobre notional (%)
    sharpe: float                # anualizado, base-tiempo (incluye barras flat)
    max_drawdown: float          # MaxDD de la equity neta (fracción positiva)
    profit_factor: float         # PF de los retornos netos por trade
    pct_winning_trades: float
    avg_holding_bars: float      # duración media de un trade (barras de 5m)
    folds_same_sign: int
    n_folds: int
    passes_golden: bool


# ---------------------------------------------------------------------------
# Spearman sin scipy (scipy no está en el proyecto)
# ---------------------------------------------------------------------------

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=float)
        out[np.argsort(a, kind="stable")] = np.arange(1, n + 1, dtype=float)
        return out

    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


# ---------------------------------------------------------------------------
# Construcción de la señal
# ---------------------------------------------------------------------------

def daily_anchored_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, day_key: pd.Series,
) -> pd.Series:
    """VWAP intradía anclado: cumsum(tp·vol)/cumsum(vol) reiniciado cada día UTC."""
    tp = (high + low + close) / 3.0
    pv = tp * volume
    cum_pv = pv.groupby(day_key).cumsum()
    cum_v = volume.groupby(day_key).cumsum()
    return cum_pv / cum_v


def vwap_deviation_z(close: pd.Series, vwap: pd.Series, window: int) -> pd.Series:
    """Z-score de la desviación (close−VWAP)/VWAP sobre una ventana rolling.

    El z-score da un umbral ESTACIONARIO (z_entry fijo es comparable a lo largo
    del tiempo aunque la volatilidad de la desviación cambie). La ventana cruza
    días a propósito: estandariza "qué tan inusual es esta desviación" vs el
    régimen reciente.
    """
    dev = (close - vwap) / vwap
    mu = dev.rolling(window).mean()
    sd = dev.rolling(window).std()
    return (dev - mu) / sd.replace(0.0, np.nan)


def atr_over_price(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """ATR(period) / close → fracción usada por el slippage dinámico k·ATR/precio."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / close


# ---------------------------------------------------------------------------
# Etapa 1 — IC gate (bounce-robust: forward desde t+1)
# ---------------------------------------------------------------------------

def vwap_ic(dev_z: pd.Series, close: pd.Series, horizon: int) -> tuple[float, float]:
    """IC Spearman(z_t, ret_{t+1 → t+1+h}) con corrección n_eff.

    El retorno futuro arranca en t+1 (no en t) para excluir el bid-ask bounce de
    la barra inmediata: ése es el IC realmente OPERABLE (entras una barra tarde).
    """
    z = dev_z.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)
    if n <= horizon + 2:
        return 0.0, 0.0

    fwd = np.full(n, np.nan)
    entry = c[1 : n - horizon]          # close_{t+1}
    exitp = c[1 + horizon : n]          # close_{t+1+h}
    fwd[: len(entry)] = exitp / entry - 1.0

    valid = ~np.isnan(z) & ~np.isnan(fwd)
    z_v, fwd_v = z[valid], fwd[valid]
    nv = len(z_v)
    if nv < 30:
        return 0.0, 0.0

    ic = _spearman_r(z_v, fwd_v)
    rho1 = float(pd.Series(z_v).autocorr(lag=1))
    n_eff = max(nv * (1.0 - rho1) / (1.0 + rho1), 2.0) if abs(rho1) < 1.0 else 2.0
    denom = math.sqrt(max(1.0 - ic ** 2, 1e-9))
    t = ic * math.sqrt(n_eff) / denom
    return float(ic), float(t)


# ---------------------------------------------------------------------------
# Etapa 2 — máquina de estados (bar a bar)
# ---------------------------------------------------------------------------

def _simulate_bars(
    dev_z: pd.Series,
    close: pd.Series,
    atr_px: pd.Series,
    day_key: pd.Series,
    cfg: QuantMatrixConfig,
) -> tuple[np.ndarray, list[float], list[int]]:
    """FLAT / LONG / SHORT con entrada diferida una barra (bounce-robust).

    Señal = z.shift(1): se observa en t, se ENTRA/SALE en close_t (= una barra
    después del close que generó la señal). El precio cripto es continuo en el
    límite del día (24/7), así que el M2M cruza medianoche sin gap; aun así
    forzamos el cierre al cambiar de día porque el VWAP resetea y la desviación
    pierde su referencia.

    Returns:
        bar_returns: retorno NETO por barra (M2M − costos al cerrar). La equity
                     curve (cumsum) refleja costos → MaxDD/Sharpe honestos.
        trade_net:   retorno neto por trade cerrado (para PF y % ganadores).
        holdings:    nº de barras que duró cada trade (diagnóstico).
    """
    sig = dev_z.shift(1).to_numpy(dtype=float)   # actuar una barra tarde
    c = close.to_numpy(dtype=float)
    slip = (cfg.slippage_pct / 100.0) + cfg.slippage_atr_mult * atr_px.to_numpy(dtype=float)
    taker = cfg.taker_commission_pct / 100.0
    day = day_key.to_numpy()
    n = len(c)

    bar_returns = np.zeros(n, dtype=float)
    trade_net: list[float] = []
    holdings: list[int] = []

    pos = 0            # 0=flat, +1=long, −1=short
    entry_i = -1
    trade_gross = 0.0

    for i in range(1, n):
        # 1. Mark-to-market bruto (si hay posición desde la barra previa)
        if pos != 0:
            bar_returns[i] = pos * (c[i] / c[i - 1] - 1.0)
            trade_gross += bar_returns[i]

        s = sig[i]
        force = day[i] != day[i - 1]   # cambio de día UTC → VWAP reseteó

        # 2. Salida (después del M2M)
        if pos != 0:
            do_exit = force or (not math.isnan(s) and abs(s) < cfg.vwap_z_exit)
            if do_exit:
                cost = 2.0 * taker + _slip_at(slip, entry_i) + _slip_at(slip, i)
                trade_net.append(trade_gross - cost)
                holdings.append(i - entry_i)
                bar_returns[i] -= cost      # la equity neta incluye el costo una vez
                pos = 0

        # 3. Entrada (solo si FLAT y la señal supera el umbral)
        if pos == 0 and not math.isnan(s):
            if s > cfg.vwap_z_entry:
                pos = -1                    # precio sobre VWAP → revierte abajo → SHORT
                entry_i = i
                trade_gross = 0.0
            elif s < -cfg.vwap_z_entry:
                pos = 1                     # precio bajo VWAP → revierte arriba → LONG
                entry_i = i
                trade_gross = 0.0

    # Cierre forzado al final de la serie
    if pos != 0:
        last = n - 1
        cost = 2.0 * taker + _slip_at(slip, entry_i) + _slip_at(slip, last)
        trade_net.append(trade_gross - cost)
        holdings.append(last - entry_i)
        bar_returns[last] -= cost

    return bar_returns, trade_net, holdings


def _slip_at(slip: np.ndarray, i: int) -> float:
    """Slippage en la barra i; si es NaN (warmup del ATR) usa 0 (conservador-neutral)."""
    v = slip[i]
    return 0.0 if math.isnan(v) else float(v)


# ---------------------------------------------------------------------------
# Simulador principal
# ---------------------------------------------------------------------------

def simulate_vwap(
    df: pd.DataFrame,
    cfg: QuantMatrixConfig,
    *,
    symbol: str = "",
    n_folds: int = 4,
) -> VwapStats:
    """Simula la reversión a VWAP intradía sobre un DataFrame OHLCV de 5m.

    df debe traer columnas: open_time (datetime UTC), high, low, close, volume.
    """
    day_key = df["open_time"].dt.floor("D")
    vwap = daily_anchored_vwap(df["high"], df["low"], df["close"], df["volume"], day_key)
    dev_z = vwap_deviation_z(df["close"], vwap, cfg.vwap_z_window)
    atr_px = atr_over_price(df["high"], df["low"], df["close"], cfg.atr_period)

    # Etapa 1
    ic, ic_t = vwap_ic(dev_z, df["close"], cfg.vwap_forward_horizon)

    # Etapa 2
    bar_returns, trade_net, holdings = _simulate_bars(
        dev_z, df["close"], atr_px, day_key, cfg
    )

    valid_mask = ~np.isnan(dev_z.to_numpy(dtype=float))
    r = bar_returns[valid_mask]
    n = len(r)
    if n == 0:
        return VwapStats(
            symbol=symbol, n_periods=0, ic_spearman=ic, ic_tstat=ic_t,
            n_trades=0, net_return_ann_pct=0.0, sharpe=0.0, max_drawdown=0.0,
            profit_factor=0.0, pct_winning_trades=0.0, avg_holding_bars=0.0,
            folds_same_sign=0, n_folds=n_folds, passes_golden=False,
        )

    years = n / BARS_PER_YEAR_5M
    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(BARS_PER_YEAR_5M) if std_r > 0 else 0.0
    equity = 1.0 + np.cumsum(r)
    mdd = max_drawdown(equity)

    trade_arr = np.array(trade_net, dtype=float)
    n_trades = len(trade_arr)
    pf = profit_factor(trade_arr) if n_trades > 0 else 0.0
    pct_win = float((trade_arr > 0).mean()) * 100.0 if n_trades > 0 else 0.0
    avg_hold = float(np.mean(holdings)) if holdings else 0.0
    net_total = float(trade_arr.sum()) if n_trades > 0 else 0.0
    net_ann = (net_total / years) * 100.0 if years > 0 else 0.0

    folds = np.array_split(r, n_folds)
    fold_signs = [1 if f.mean() > 0 else -1 for f in folds]
    folds_same_sign = max(fold_signs.count(1), fold_signs.count(-1))

    passes = (
        ic < 0
        and abs(ic_t) >= cfg.golden_min_tstat
        and pf > cfg.golden_min_profit_factor
        and net_ann > 0
        and folds_same_sign == n_folds
    )

    return VwapStats(
        symbol=symbol, n_periods=n, ic_spearman=ic, ic_tstat=ic_t,
        n_trades=n_trades, net_return_ann_pct=net_ann, sharpe=sharpe,
        max_drawdown=mdd, profit_factor=pf, pct_winning_trades=pct_win,
        avg_holding_bars=avg_hold, folds_same_sign=folds_same_sign,
        n_folds=n_folds, passes_golden=passes,
    )
