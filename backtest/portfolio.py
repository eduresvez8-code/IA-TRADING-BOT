"""Portafolio long-short de REVERSIÓN cross-sectional: ¿el IC se cobra como PnL?

El edge test dijo que existe reversión significativa (IC negativo), pero la cola
gorda hacía que el long-short ingenuo no la cobrara. Aquí construimos el
portafolio ROBUSTO A LA COLA que pediste y lo backtesteamos neto de costos:

  1. Filtro de liquidez: fuera el `liquidity_drop_pct` de menor volumen (las
     microcaps que forman las colas artificiales).
  2. Winsorización: los retornos de cada rebalanceo se recortan al percentil
     [w, 1-w] para que ningún activo (un 5x) domine una pierna.
  3. LONG el quintil de perdedores recientes, SHORT el de ganadores recientes.
  4. Pesos inversos a la volatilidad, con tope `max_weight` por activo.
  5. Neto de costos por turnover real, con walk-forward.

Reporta Sharpe, Max DD, Profit Factor y —clave— el aporte del lado LARGO vs el
SHORT por separado (si todo viene del corto, es el lado peligroso).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import Settings
from backtest.cross_sectional import forward_return, momentum_factor


def load_universe_fields(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, quote_volume) en formato ancho: índice fecha, columnas símbolo."""
    long = pd.read_parquet(Path(settings.storage.universe_dir) / "daily.parquet")
    close = long.pivot(index="open_time", columns="symbol", values="close").sort_index()
    qvol = long.pivot(index="open_time", columns="symbol", values="quote_volume").sort_index()
    return close, qvol


def inverse_vol_weights(vols: pd.Series, max_weight: float) -> pd.Series:
    """Pesos ∝ 1/volatilidad, normalizados a sumar 1 y con tope por activo.

    Inversa a la vol: las monedas más tranquilas pesan más (riesgo equilibrado).
    El tope evita que una sola domine; tras recortar, se renormaliza.
    """
    inv = 1.0 / vols.replace(0.0, np.nan)
    if inv.notna().sum() == 0:
        return pd.Series(1.0 / len(vols), index=vols.index)
    inv = inv.fillna(inv.median())
    w = inv / inv.sum()
    for _ in range(10):  # capping iterativo hasta respetar el tope
        over = w > max_weight
        if not over.any():
            break
        w[over] = max_weight
        free = ~over
        residual = 1.0 - w[over].sum()
        if free.any() and w[free].sum() > 0:
            w[free] = w[free] / w[free].sum() * residual
        else:
            break
    return w


@dataclass
class PortfolioResult:
    lookback: int
    n_periods: int
    avg_universe: float
    total_return: float
    ann_sharpe: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    avg_long_leg: float          # retorno semanal medio de la pierna LARGA (perdedores)
    avg_short_contrib: float     # aporte semanal medio de SHORTear ganadores (−ret corto)
    avg_cost: float
    fold_returns: list[float]    # retorno compuesto neto por tramo
    fold_pfs: list[float]
    folds_positive: int
    is_edge: bool


def _profit_factor(rets: np.ndarray) -> float:
    gains = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return float(gains / losses)


def backtest_reversal(
    close: pd.DataFrame, qvol: pd.DataFrame, settings: Settings, *, lookback: int,
) -> PortfolioResult:
    xs = settings.cross_sectional
    bt = settings.backtest
    one_way = (bt.commission_pct + bt.slippage_pct) / 100.0  # costo por lado (turnover)
    pf_min = settings.scan.edge_profit_factor_min

    factor = momentum_factor(close, lookback=lookback, skip=xs.momentum_skip_days,
                             vol_adjust=False, vol_lookback=xs.vol_lookback_days)
    fwd = forward_return(close, xs.forward_days)
    vol = close.pct_change().rolling(xs.vol_lookback_days).std()

    dates = factor.index[::xs.rebalance_days]
    net_rets: list[float] = []
    long_rets: list[float] = []
    short_contribs: list[float] = []
    costs: list[float] = []
    sizes: list[int] = []
    prev_w: pd.Series | None = None

    for t in dates:
        d = pd.DataFrame({"f": factor.loc[t], "y": fwd.loc[t],
                          "v": qvol.loc[t], "vol": vol.loc[t]}).dropna()
        if len(d) < xs.min_assets:
            continue
        # 1) filtro de liquidez: fuera el cuantil inferior de volumen
        d = d[d["v"] >= d["v"].quantile(xs.liquidity_drop_pct)]
        if len(d) < xs.min_assets:
            continue
        # 2) winsorización de los retornos forward
        if xs.winsorize_quantile > 0:
            lo, hi = d["y"].quantile(xs.winsorize_quantile), d["y"].quantile(1 - xs.winsorize_quantile)
            d["y"] = d["y"].clip(lo, hi)
        # 3) quintiles por momentum
        try:
            b = pd.qcut(d["f"], xs.n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        if pd.Series(b).nunique() < xs.n_quantiles:
            continue
        losers = d[b == 0]                       # Q1: perdedores → LONG
        winners = d[b == xs.n_quantiles - 1]     # Q5: ganadores → SHORT
        # 4) pesos inversos a la vol, con tope
        wl = inverse_vol_weights(losers["vol"], xs.max_weight)
        ws = inverse_vol_weights(winners["vol"], xs.max_weight)
        long_ret = float((wl * losers["y"]).sum())
        short_ret = float((ws * winners["y"]).sum())
        gross = long_ret - short_ret             # largo perdedores − corto ganadores

        # 5) turnover real → costo
        w_now = pd.Series(0.0, index=d.index)
        w_now.loc[losers.index] = wl
        w_now.loc[winners.index] = -ws
        if prev_w is None:
            turnover = float(w_now.abs().sum())
        else:
            idx = w_now.index.union(prev_w.index)
            turnover = float((w_now.reindex(idx, fill_value=0.0)
                              - prev_w.reindex(idx, fill_value=0.0)).abs().sum())
        cost = turnover * one_way
        prev_w = w_now

        net_rets.append(gross - cost)
        long_rets.append(long_ret)
        short_contribs.append(-short_ret)
        costs.append(cost)
        sizes.append(len(d))

    arr = np.asarray(net_rets, dtype=float)
    n = len(arr)
    if n == 0:
        return PortfolioResult(lookback, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               [], [], 0, False)
    equity = np.cumprod(1.0 + arr)
    total_return = float(equity[-1] - 1.0)
    sharpe = float(arr.mean() / arr.std(ddof=1) * math.sqrt(52)) if n > 1 and arr.std(ddof=1) > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak)) if n else 0.0
    pf = _profit_factor(arr)
    win_rate = float((arr > 0).mean())

    # walk-forward: 4 tramos
    fold_returns: list[float] = []
    fold_pfs: list[float] = []
    n_folds = 4
    if n >= n_folds * 3:
        size = n // n_folds
        for k in range(n_folds):
            lo, hi = k * size, ((k + 1) * size if k < n_folds - 1 else n)
            seg = arr[lo:hi]
            fold_returns.append(float(np.prod(1.0 + seg) - 1.0))
            fold_pfs.append(_profit_factor(seg))
    folds_positive = sum(1 for r in fold_returns if r > 0)
    is_edge = (pf > pf_min and total_return > 0
               and len(fold_returns) == n_folds and folds_positive == n_folds)

    return PortfolioResult(
        lookback=lookback, n_periods=n, avg_universe=float(np.mean(sizes)),
        total_return=total_return, ann_sharpe=sharpe, max_drawdown=max_dd,
        profit_factor=pf, win_rate=win_rate,
        avg_long_leg=float(np.mean(long_rets)), avg_short_contrib=float(np.mean(short_contribs)),
        avg_cost=float(np.mean(costs)), fold_returns=fold_returns, fold_pfs=fold_pfs,
        folds_positive=folds_positive, is_edge=is_edge)
