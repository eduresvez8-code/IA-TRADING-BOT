"""Edge test del factor de momentum CROSS-SECTIONAL (relative strength).

A diferencia del edge test de un solo activo (¿la señal predice el retorno futuro
de ESTE activo?), aquí preguntamos: en cada fecha de rebalanceo, ¿el RANKING del
factor entre todo el universo predice el ranking de retornos futuros? Es la
metodología estándar de factor research (Grinold-Kahn):

  1. Factor por activo y fecha: momentum = retorno de los últimos `lookback` días
     (opcionalmente saltando los últimos `skip` y/o dividido por su volatilidad).
  2. IC cross-sectional de una fecha = Spearman(ranking del factor, retorno futuro)
     ENTRE los activos de esa fecha.
  3. Serie temporal de ICs (una por fecha de rebalanceo, no solapadas si
     rebalance_days = forward_days). Agregado: IC medio y su t-stat (IC-IR):
        t = mean(IC) / std(IC) · sqrt(n_fechas)
     |t|≥2 ⇒ el factor premia al ganador de forma consistente en el tiempo.

🥇 REGLA DE ORO: además, el spread de retorno futuro entre el quintil top y el
bottom del factor (el long-short bruto semanal) debe superar el costo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import Settings, load_settings
from backtest.edge import spearman_ic
from backtest.funding_edge import round_trip_cost


def load_universe_panel(settings: Settings) -> pd.DataFrame:
    """Panel ancho de cierres diarios: índice = fecha, columnas = símbolo."""
    path = Path(settings.storage.universe_dir) / "daily.parquet"
    long = pd.read_parquet(path)
    return long.pivot(index="open_time", columns="symbol", values="close").sort_index()


def momentum_factor(
    close: pd.DataFrame, *, lookback: int, skip: int,
    vol_adjust: bool, vol_lookback: int,
) -> pd.DataFrame:
    """Factor momentum por activo y fecha: retorno de los últimos `lookback` días.

    `skip` excluye los días más recientes (la reversión de corto plazo puede
    contaminar el momentum). `vol_adjust` lo normaliza por la volatilidad
    realizada (premia tendencia suave sobre saltos ruidosos). Todo causal:
    en t solo se usan precios ≤ t.
    """
    ref = close.shift(skip)
    base = close.shift(skip + lookback)
    factor = ref / base - 1.0
    if vol_adjust:
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(vol_lookback).std().shift(skip)
        factor = factor / vol.replace(0.0, np.nan)
    return factor


def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Retorno futuro a `horizon` días por activo: close[t+h]/close[t]-1."""
    return close.shift(-horizon) / close - 1.0


@dataclass
class CrossSectionalResult:
    lookback: int
    vol_adjust: bool
    n_dates: int             # nº de fechas de rebalanceo con cross-section válida
    avg_universe: float      # tamaño medio de la cross-section por fecha
    mean_ic: float
    std_ic: float
    t_stat: float            # IC-IR × sqrt(n): significancia del IC medio
    ic_positive_rate: float  # fracción de fechas con IC>0
    mean_quantile_spread: float  # retorno medio top-quintil − bottom-quintil (semanal)
    cost: float
    beats_cost: bool         # |spread|>costo Y |t|≥2
    fold_mean_ics: list[float]
    fold_same_sign: int


def _quantile_spread(factor_row: pd.Series, fwd_row: pd.Series, n_quantiles: int) -> float:
    df = pd.DataFrame({"f": factor_row, "y": fwd_row}).dropna()
    if len(df) < n_quantiles * 2:
        return float("nan")
    try:
        buckets = pd.qcut(df["f"], n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    means = df["y"].groupby(buckets).mean()
    if len(means) < 2:
        return float("nan")
    return float(means.iloc[-1] - means.iloc[0])


def cross_sectional_ic(
    factor: pd.DataFrame, fwd: pd.DataFrame, *, lookback: int, vol_adjust: bool,
    rebalance_days: int, min_assets: int, n_quantiles: int, cost: float, n_folds: int = 4,
) -> CrossSectionalResult:
    dates = factor.index[::rebalance_days]
    ics: list[float] = []
    spreads: list[float] = []
    sizes: list[int] = []
    for t in dates:
        f, y = factor.loc[t], fwd.loc[t]
        valid = f.notna() & y.notna()
        if int(valid.sum()) < min_assets:
            continue
        fv, yv = f[valid], y[valid]
        ics.append(spearman_ic(fv, yv))
        sizes.append(int(valid.sum()))
        sp = _quantile_spread(fv, yv, n_quantiles)
        if not math.isnan(sp):
            spreads.append(sp)

    n = len(ics)
    arr = np.asarray(ics, dtype=float)
    mean_ic = float(arr.mean()) if n else 0.0
    std_ic = float(arr.std(ddof=1)) if n > 1 else 0.0
    t_stat = (mean_ic / std_ic * math.sqrt(n)) if std_ic > 0 else 0.0
    pos_rate = float((arr > 0).mean()) if n else 0.0
    mean_spread = float(np.mean(spreads)) if spreads else 0.0

    fold_means: list[float] = []
    if n >= n_folds * 3:
        size = n // n_folds
        for k in range(n_folds):
            lo, hi = k * size, ((k + 1) * size if k < n_folds - 1 else n)
            fold_means.append(float(arr[lo:hi].mean()))
    same = sum(1 for fm in fold_means if fm != 0 and (fm > 0) == (mean_ic > 0))

    # Retorno long-short COHERENTE con el signo del IC: si IC>0 (momentum) vas
    # largo el top y corto el bottom (= spread); si IC<0 (reversión) al revés
    # (= −spread). Solo es edge si ese retorno es POSITIVO y supera el costo —
    # un IC negativo con spread positivo (media inflada por la cola) NO es edge.
    coherent_ls = (1.0 if mean_ic >= 0 else -1.0) * mean_spread
    return CrossSectionalResult(
        lookback=lookback, vol_adjust=vol_adjust, n_dates=n,
        avg_universe=(float(np.mean(sizes)) if sizes else 0.0),
        mean_ic=mean_ic, std_ic=std_ic, t_stat=t_stat, ic_positive_rate=pos_rate,
        mean_quantile_spread=mean_spread, cost=cost,
        beats_cost=(coherent_ls > cost and abs(t_stat) >= 2.0),
        fold_mean_ics=fold_means, fold_same_sign=same,
    )


def analyze(
    close: pd.DataFrame, settings: Settings, *,
    lookback: int | None = None, skip: int | None = None, vol_adjust: bool | None = None,
) -> CrossSectionalResult:
    """Corre el edge test cross-sectional para una especificación de factor."""
    xs = settings.cross_sectional
    lb = lookback if lookback is not None else xs.momentum_lookback_days
    sk = skip if skip is not None else xs.momentum_skip_days
    va = xs.vol_adjust if vol_adjust is None else vol_adjust
    factor = momentum_factor(close, lookback=lb, skip=sk, vol_adjust=va,
                             vol_lookback=xs.vol_lookback_days)
    fwd = forward_return(close, xs.forward_days)
    return cross_sectional_ic(
        factor, fwd, lookback=lb, vol_adjust=va, rebalance_days=xs.rebalance_days,
        min_assets=xs.min_assets, n_quantiles=xs.n_quantiles, cost=round_trip_cost(settings))
