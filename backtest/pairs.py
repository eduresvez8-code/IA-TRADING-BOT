"""Familia B — Cointegración de pares (rolling OLS + IC gate + P&L).

Hipótesis: dos activos crypto cointegrados → el z-score del spread (residuo de
la regresión de log-precios) revierte a la media → IC Spearman < 0 y
significativo (Etapa 1) → rentabilidad después de costos (Etapa 2).

Diseño de rolling OLS (vectorizado):
    β_t = Cov(log_A, log_B)[t-w : t] / Var(log_B)[t-w : t]
    shift(1) → β estimada con datos [t-w : t-1], aplicada al bar t (sin lookahead)
    spread_t = log_A_t − β_t · log_B_t

Gate de Etapa 1 — IC Spearman con corrección n_eff:
    IC = Spearman(z_t, Δspread_{t+1})    esperado < 0 (z alto → spread cae)
    n_eff = n × (1 − ρ₁) / (1 + ρ₁)     ρ₁ = autocorr(z, lag=1)
    t_corr = IC × √n_eff / √(1 − IC²)
El z-score de un spread cripto típico tiene ρ₁ ≈ 0.995 → n_eff ≈ 54 para n=25000.
Para |t_corr| ≥ 2.0 se necesita |IC| ≥ 0.27, mucho más que el IC típico de señales
quant (0.02-0.08). Es probable que ningún par cripto a 1h pase este gate honestamente.

Costos (por round-trip de 1 par):
    4 lados taker (2 entradas + 2 salidas), divididos entre (1 + |β|) sobre capital.
    Mismo taker_commission_pct que Familia E (sin mezclar con BacktestConfig legacy).

Una sesión = un módulo. Este módulo reemplaza el stub run_family_pairs en quant_matrix.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from backtest.metrics import max_drawdown, profit_factor
from src.core.config import QuantMatrixConfig

PERIODS_PER_YEAR = 365 * 24  # barras de 1h por año


@dataclass
class PairsStats:
    """Resultado del simulador de par para un par (A, B)."""
    pair: str
    n_periods: int              # barras válidas (después del warmup del rolling)
    ic_spearman: float          # IC(z, Δspread_{+1h}), esperado < 0
    ic_tstat: float             # t-stat corregido por autocorr del z-score
    n_trades: int               # round-trips completados (incluyendo forzado al final)
    net_return_ann_pct: float   # retorno neto anualizado sobre capital (%)
    sharpe: float               # Sharpe anualizado sobre bar-returns brutos
    max_drawdown: float         # MaxDD de la equity curve bruta (fracción positiva)
    profit_factor: float        # PF de trade_net_returns (neto de costos)
    pct_winning_trades: float   # % trades netos positivos
    folds_same_sign: int        # folds con bar-P&L positivo
    n_folds: int
    passes_golden: bool         # supera el gate de Etapa 1 + Etapa 2


# ---------------------------------------------------------------------------
# Spearman sin scipy (scipy no está en el proyecto)
# ---------------------------------------------------------------------------

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de Spearman vía ranking + corrcoef de numpy."""
    n = len(x)
    if n < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=float)
        out[np.argsort(a, kind="stable")] = np.arange(1, n + 1, dtype=float)
        return out

    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


# ---------------------------------------------------------------------------
# Etapa 1 — IC gate
# ---------------------------------------------------------------------------

def pairs_ic(z: pd.Series, spread: pd.Series) -> tuple[float, float]:
    """IC Spearman y t-stat corregido por autocorrelación del z-score.

    La corrección n_eff es FUNDAMENTAL para el spread de un par cripto: con
    ρ₁(z) ≈ 0.995 a 1h, n_eff ≈ 54 aunque tengamos 25 000 observaciones.
    El t-stat naive (sin corregir) es unas 20× demasiado grande.
    """
    future_ret = spread.diff(1).shift(-1)   # Δspread en el bar siguiente
    valid = z.notna() & future_ret.notna()
    z_v = z[valid].to_numpy(dtype=float)
    fr_v = future_ret[valid].to_numpy(dtype=float)
    n = int(valid.sum())
    if n < 30:
        return 0.0, 0.0

    ic = _spearman_r(z_v, fr_v)
    rho1 = float(pd.Series(z_v).autocorr(lag=1))
    if abs(rho1) < 1.0:
        n_eff = max(n * (1.0 - rho1) / (1.0 + rho1), 2.0)
    else:
        n_eff = 2.0
    denom = math.sqrt(max(1.0 - ic ** 2, 1e-9))
    t = ic * math.sqrt(n_eff) / denom
    return float(ic), float(t)


# ---------------------------------------------------------------------------
# Etapa 2 — simulador bar a bar
# ---------------------------------------------------------------------------

def _simulate_bars(
    z: pd.Series,
    spread: pd.Series,
    beta: pd.Series,
    cfg: QuantMatrixConfig,
) -> tuple[np.ndarray, list[float]]:
    """Máquina de estados: FLAT / LONG_SPREAD / SHORT_SPREAD.

    Returns:
        bar_returns_gross: retorno bruto por barra (0 cuando FLAT).
        trade_net_returns: retorno neto por trade cerrado (bruto − costos).

    bar_returns acumula el mark-to-market del spread; la suma de bar_returns
    sobre un trade cerrado iguala exactamente el gross del trade (comprobable
    en los tests). Los costos one-time (4 lados taker) se separan en
    trade_net_returns, consistente con el enfoque de la Familia E.
    """
    z_arr = z.to_numpy(dtype=float)
    spread_arr = spread.to_numpy(dtype=float)
    beta_arr = beta.to_numpy(dtype=float)
    n = len(z_arr)

    bar_returns = np.zeros(n, dtype=float)
    trade_net_returns: list[float] = []

    position = 0          # 0=flat, +1=long spread, −1=short spread
    entry_idx = -1
    entry_spread = 0.0
    cap_mult = 2.0        # se actualiza en cada entrada; aquí solo placeholder
    last_spread = math.nan

    for i in range(n):
        if math.isnan(z_arr[i]) or math.isnan(beta_arr[i]) or math.isnan(spread_arr[i]):
            last_spread = math.nan
            continue

        # 1. Mark-to-market (antes de la lógica de señal)
        if position != 0 and not math.isnan(last_spread) and i > entry_idx:
            bar_returns[i] = position * (spread_arr[i] - last_spread) / cap_mult

        last_spread = spread_arr[i]

        # 2. Señal de salida (después de M2M, antes de nueva entrada)
        if position != 0 and i > entry_idx and abs(z_arr[i]) < cfg.pairs_z_exit:
            gross = position * (spread_arr[i] - entry_spread) / cap_mult
            cost = (4.0 * cfg.taker_commission_pct / 100.0) / cap_mult
            trade_net_returns.append(gross - cost)
            position = 0

        # 3. Señal de entrada (solo cuando FLAT)
        if position == 0:
            if z_arr[i] < -cfg.pairs_z_entry:
                position = 1    # long spread (espera subida)
                entry_idx = i
                entry_spread = spread_arr[i]
                cap_mult = 1.0 + abs(beta_arr[i])
            elif z_arr[i] > cfg.pairs_z_entry:
                position = -1   # short spread (espera bajada)
                entry_idx = i
                entry_spread = spread_arr[i]
                cap_mult = 1.0 + abs(beta_arr[i])

    # Cierre forzado al final del período (para no dejar trade abierto)
    if position != 0 and not math.isnan(last_spread):
        gross = position * (last_spread - entry_spread) / cap_mult
        cost = (4.0 * cfg.taker_commission_pct / 100.0) / cap_mult
        trade_net_returns.append(gross - cost)

    return bar_returns, trade_net_returns


# ---------------------------------------------------------------------------
# Simulador principal
# ---------------------------------------------------------------------------

def simulate_pairs(
    log_a: pd.Series,
    log_b: pd.Series,
    cfg: QuantMatrixConfig,
    *,
    pair_name: str = "",
    n_folds: int = 4,
) -> PairsStats:
    """Simula el pairs trade entre dos series de log-precios (1h).

    Etapa 1: IC gate (barato). Etapa 2: P&L bar-a-bar (caro, siempre corre
    pero solo 'pasa' si también supera el gate de Etapa 1).
    """
    w = cfg.pairs_lookback_hours

    # Rolling OLS vectorizado: β = Cov(A,B) / Var(B), shift(1) evita lookahead
    cov_ab = log_a.rolling(w).cov(log_b)
    var_b = log_b.rolling(w).var()
    beta = (cov_ab / var_b).shift(1)   # beta_t estimada sobre [t-w:t-1]

    spread = log_a - beta * log_b
    spread_mean = spread.rolling(w).mean()
    spread_std = spread.rolling(w).std()
    z = (spread - spread_mean) / spread_std.replace(0.0, float("nan"))

    # Etapa 1
    ic, ic_t = pairs_ic(z, spread)

    # Etapa 2
    bar_returns, trade_net = _simulate_bars(z, spread, beta, cfg)

    # Solo contamos barras donde z tiene valor (post-warmup)
    valid_mask = ~np.isnan(z.to_numpy(dtype=float))
    r = bar_returns[valid_mask]
    n = len(r)

    if n == 0:
        return PairsStats(
            pair=pair_name, n_periods=0, ic_spearman=ic, ic_tstat=ic_t,
            n_trades=0, net_return_ann_pct=0.0, sharpe=0.0, max_drawdown=0.0,
            profit_factor=0.0, pct_winning_trades=0.0,
            folds_same_sign=0, n_folds=n_folds, passes_golden=False,
        )

    years = n / PERIODS_PER_YEAR
    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(PERIODS_PER_YEAR) if std_r > 0 else 0.0
    equity = 1.0 + np.cumsum(r)
    mdd = max_drawdown(equity)

    trade_arr = np.array(trade_net, dtype=float)
    n_trades = len(trade_arr)
    pf = profit_factor(trade_arr) if n_trades > 0 else 0.0
    pct_win = float((trade_arr > 0).mean()) * 100.0 if n_trades > 0 else 0.0

    # Retorno neto anualizado: suma de trade P&L neto sobre número de años
    net_total = float(trade_arr.sum()) if n_trades > 0 else 0.0
    net_ann = (net_total / years) * 100.0

    # Walk-forward: P&L de cada fold debe ser positivo
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

    return PairsStats(
        pair=pair_name, n_periods=n, ic_spearman=ic, ic_tstat=ic_t,
        n_trades=n_trades, net_return_ann_pct=net_ann, sharpe=sharpe,
        max_drawdown=mdd, profit_factor=pf, pct_winning_trades=pct_win,
        folds_same_sign=folds_same_sign, n_folds=n_folds, passes_golden=passes,
    )


# ---------------------------------------------------------------------------
# Runner — todos los C(n, 2) pares
# ---------------------------------------------------------------------------

def run_pairs_all(
    log_prices: pd.DataFrame,
    cfg: QuantMatrixConfig,
    *,
    n_folds: int = 4,
) -> list[PairsStats]:
    """Evalúa todos los C(n, 2) pares del DataFrame de log-precios alineados."""
    symbols = list(log_prices.columns)
    results: list[PairsStats] = []
    for a, b in combinations(symbols, 2):
        aligned = log_prices[[a, b]].dropna()
        if len(aligned) < 2 * cfg.pairs_lookback_hours:
            continue   # datos insuficientes para el rolling
        results.append(
            simulate_pairs(
                aligned[a], aligned[b], cfg,
                pair_name=f"{a}-{b}", n_folds=n_folds,
            )
        )
    return results
