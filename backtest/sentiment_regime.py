"""Tres métodos cuantitativos de sentimiento con el Fear & Greed como ancla de régimen.

NO es el overlay de tamaño del Sprint C (que ya falló). Aquí el F&G CONDICIONA el
comportamiento, a frecuencia DIARIA (estabilidad: nada de scalping):

  A. Regime-switching: ¿el retorno futuro difiere según el régimen de F&G
     (miedo extremo vs codicia extrema)? IC + medias condicionales con t-stat +
     estrategia long/flat por régimen (contrarian y momentum).
  B. Mean-reversion gated por sentimiento: ¿la reversión a la media funciona
     SOLO cuando el sentimiento es extremo? IC del señal MR en días extremos vs
     normales + estrategia MR gateada.
  C. Vol-scaling por sentimiento: reducir exposición en euforia (codicia extrema)
     para cortar el drawdown. Compara buy-and-hold vs exposición escalada.

Funciones puras sobre un DataFrame diario [close, fng, ret]. Reutilizan el IC de
backtest/edge.py y la matriz de métricas de backtest/metrics.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import SentimentRegimeConfig
from backtest.edge import corr_tstat, spearman_ic
from backtest.metrics import (
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    win_rate,
)

REGIME_ORDER = ["ExtFear", "Fear", "Neutral", "Greed", "ExtGreed"]


def build_daily(close: pd.Series, fng: pd.Series) -> pd.DataFrame:
    """Alinea precio diario y F&G por fecha; añade el retorno diario."""
    df = pd.DataFrame({"close": close, "fng": fng}).dropna().sort_index()
    df["ret"] = df["close"].pct_change()
    return df


def label_regime(value: float, sr: SentimentRegimeConfig) -> str:
    if value < sr.ext_fear_below:
        return "ExtFear"
    if value < sr.fear_below:
        return "Fear"
    if value < sr.greed_above:
        return "Neutral"
    if value < sr.ext_greed_above:
        return "Greed"
    return "ExtGreed"


def fwd_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


# ----------------------------- métricas de estrategia -----------------------------

@dataclass
class StrategyMetrics:
    name: str
    n_trades: int
    exposure: float            # fracción de días invertido
    total_return: float
    ann_sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    expectancy: float          # PnL medio por trade (fracción)
    is_edge: bool


def _trades_from_positions(pos: np.ndarray, net: np.ndarray) -> list[float]:
    """PnL de cada periodo contiguo invertido (compuesto)."""
    pnls: list[float] = []
    i, n = 0, len(pos)
    while i < n:
        if pos[i] > 0:
            comp, j = 1.0, i
            while j < n and pos[j] > 0:
                comp *= 1.0 + net[j]
                j += 1
            pnls.append(comp - 1.0)
            i = j
        else:
            i += 1
    return pnls


def backtest_long_flat(df: pd.DataFrame, in_market: pd.Series, *, one_way: float,
                       pf_min: float, name: str) -> StrategyMetrics:
    """Backtest diario long/flat: invertido cuando in_market, neto de costos."""
    ret = df["ret"].fillna(0.0).to_numpy()
    pos = in_market.shift(1).fillna(False).astype(float).to_numpy()  # entra al día siguiente
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))             # |Δposición| = costo
    net = pos * ret - turn * one_way
    eq = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    pnls = _trades_from_positions(pos, net)
    pf = profit_factor(pnls)
    tr = total_return(eq)
    return StrategyMetrics(
        name=name, n_trades=len(pnls), exposure=float((pos > 0).mean()),
        total_return=tr, ann_sharpe=sharpe_ratio(eq, "1d"),
        max_drawdown=max_drawdown(eq), win_rate=win_rate(pnls),
        profit_factor=pf, expectancy=(float(np.mean(pnls)) if pnls else 0.0),
        is_edge=(pf > pf_min and tr > 0))


# ----------------------------- Método A: regime-switching -----------------------------

@dataclass
class RegimeStat:
    regime: str
    n_days: int
    n_eff: int
    mean_fwd: float
    median_fwd: float
    win_rate: float
    t_vs_uncond: float


def regime_stats(df: pd.DataFrame, sr: SentimentRegimeConfig) -> list[RegimeStat]:
    h = sr.forward_days
    fwd = fwd_return(df["close"], h)
    reg = df["fng"].map(lambda v: label_regime(v, sr))
    valid = fwd.notna()
    u = fwd[valid]
    mu, sd = float(u.mean()), float(u.std(ddof=1))
    out = []
    for name in REGIME_ORDER:
        m = valid & (reg == name)
        v = fwd[m]
        n = len(v)
        n_eff = max(n // h, 1)
        t = ((float(v.mean()) - mu) / (sd / math.sqrt(n_eff))) if (n_eff > 1 and sd > 0) else 0.0
        out.append(RegimeStat(name, n, n_eff, float(v.mean()) if n else 0.0,
                              float(v.median()) if n else 0.0,
                              float((v > 0).mean()) if n else 0.0, t))
    return out


def regime_ic(df: pd.DataFrame, sr: SentimentRegimeConfig) -> tuple[float, float]:
    """IC del nivel de F&G vs retorno futuro (signo + = momentum, − = contrarian)."""
    h = sr.forward_days
    fwd = fwd_return(df["close"], h)
    m = fwd.notna()
    ic = spearman_ic(df["fng"][m], fwd[m])
    return ic, corr_tstat(ic, max(int(m.sum()) // h, 1))


# ----------------------------- Método B: MR gated por sentimiento -----------------------------

def mr_signal(close: pd.Series, lookback: int) -> pd.Series:
    """Señal de reversión: + cuando el precio CAYÓ en los últimos `lookback` días."""
    return -(close / close.shift(lookback) - 1.0)


@dataclass
class GatedICResult:
    subset: str
    n: int
    ic: float
    t: float


def mr_gated_ic(df: pd.DataFrame, sr: SentimentRegimeConfig) -> list[GatedICResult]:
    """IC del señal MR vs retorno futuro, en TODOS los días, EXTREMOS y NORMALES."""
    h = sr.forward_days
    sig = mr_signal(df["close"], sr.mr_lookback_days)
    fwd = fwd_return(df["close"], h)
    extreme = (df["fng"] - 50).abs() >= sr.extreme_abs_threshold
    out = []
    for label, mask in (("todos", fwd.notna() & sig.notna()),
                        ("extremo", fwd.notna() & sig.notna() & extreme),
                        ("normal", fwd.notna() & sig.notna() & ~extreme)):
        ic = spearman_ic(sig[mask], fwd[mask])
        out.append(GatedICResult(label, int(mask.sum()), ic,
                                 corr_tstat(ic, max(int(mask.sum()) // h, 1))))
    return out


# ----------------------------- Método C: vol-scaling por sentimiento -----------------------------

@dataclass
class ScalingResult:
    name: str
    total_return: float
    ann_sharpe: float
    max_drawdown: float


def _scaled_metrics(df: pd.DataFrame, scale: pd.Series, *, one_way: float, name: str) -> ScalingResult:
    ret = df["ret"].fillna(0.0).to_numpy()
    s = scale.shift(1).fillna(0.0).to_numpy()                 # exposición decidida ayer
    turn = np.abs(np.diff(np.concatenate([[0.0], s])))
    net = s * ret - turn * one_way
    eq = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    return ScalingResult(name, total_return(eq), sharpe_ratio(eq, "1d"), max_drawdown(eq))


def vol_scaling(df: pd.DataFrame, sr: SentimentRegimeConfig, *, one_way: float
                ) -> tuple[ScalingResult, ScalingResult]:
    """Buy-and-hold vs exposición escalada que baja en codicia extrema."""
    buy_hold = _scaled_metrics(df, pd.Series(1.0, index=df.index),
                               one_way=one_way, name="buy_hold")
    # escala 1 hasta greed_above; baja linealmente a vol_scale_min al llegar a 100.
    span = max(100 - sr.greed_above, 1)
    over = (df["fng"] - sr.greed_above).clip(lower=0) / span
    scale = 1.0 - (1.0 - sr.vol_scale_min) * over.clip(upper=1.0)
    scaled = _scaled_metrics(df, scale, one_way=one_way, name="fng_scaled")
    return buy_hold, scaled
