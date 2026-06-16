"""Edge test de señales NO-precio: ¿funding rate / basis predicen el precio futuro?

Reutiliza la maquinaria de IC de `backtest/edge.py` (Spearman/Pearson, t-stat con
muestra efectiva, monotonicidad por cuantiles). Solo cambia la ENTRADA: en vez de
la señal técnica, la señal es el funding rate (cada 8h) o el basis/premium (1h), y
el retorno futuro se mide sobre el precio spot a horizontes en HORAS.

🥇 REGLA DE ORO (cost hurdle): una señal solo es candidata si el spread de retorno
futuro entre sus cuantiles extremos supera el costo ida-vuelta (comisión+slippage
×2), con t significativo (|t|≥2) y signo de IC consistente en los tramos
walk-forward. Si no, no se programa estrategia.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.config import Settings, load_settings
from backtest.edge import (
    corr_tstat,
    pearson_ic,
    quantile_forward_means,
    spearman_ic,
)


def round_trip_cost(settings: Settings) -> float:
    """Costo ida y vuelta como fracción: (comisión + slippage) × 2 lados."""
    bt = settings.backtest
    return 2.0 * (bt.commission_pct + bt.slippage_pct) / 100.0


@dataclass
class FundingHorizonStats:
    horizon_h: int
    n: int
    n_eff: int
    spearman_ic: float
    pearson_ic: float
    t_eff: float
    quantile_mean_fwd: list[float]
    quantile_spread: float       # cuantil alto − cuantil bajo del retorno futuro
    n_folds: int
    folds_same_sign: int         # tramos cuyo IC comparte signo con la muestra total
    tradable: bool               # |spread|>costo Y |t|≥2 (screen bruto de la Regla)


def _aligned(signal, fwd) -> tuple[pd.Series, pd.Series]:
    a = pd.Series(list(signal)).reset_index(drop=True)
    b = pd.Series(list(fwd)).reset_index(drop=True)
    mask = a.notna() & b.notna()
    return a[mask].reset_index(drop=True), b[mask].reset_index(drop=True)


def horizon_stats(
    signal, fwd, *, horizon_h: int, cadence_h: int, n_quantiles: int,
    cost: float, n_folds: int = 4,
) -> FundingHorizonStats:
    """IC y tradabilidad de una señal a un horizonte, con consistencia por tramos.

    `cadence_h` es el espaciado de las observaciones (8 para funding, 1 para
    basis): los retornos forward de horizonte H se solapan en H/cadence
    observaciones, así que la muestra efectiva es n / (H/cadence).
    """
    a, b = _aligned(signal, fwd)
    n = len(a)
    overlap = max(1, round(horizon_h / cadence_h))
    n_eff = max(n // overlap, 1)
    sp = spearman_ic(a, b)
    pe = pearson_ic(a, b)
    t = corr_tstat(sp, n_eff)
    qm = quantile_forward_means(a, b, n_quantiles)
    spread = (qm[-1] - qm[0]) if len(qm) >= 2 else 0.0

    fold_ics = []
    if n >= n_folds * 10:
        size = n // n_folds
        for k in range(n_folds):
            lo, hi = k * size, ((k + 1) * size if k < n_folds - 1 else n)
            fold_ics.append(spearman_ic(a[lo:hi], b[lo:hi]))
    same = sum(1 for f in fold_ics if f != 0 and (f > 0) == (sp > 0))

    tradable = abs(spread) > cost and abs(t) >= 2.0
    return FundingHorizonStats(
        horizon_h=horizon_h, n=n, n_eff=n_eff, spearman_ic=sp, pearson_ic=pe,
        t_eff=t, quantile_mean_fwd=qm, quantile_spread=spread,
        n_folds=len(fold_ics), folds_same_sign=same, tradable=tradable)


def _hourly_price(price_1h: pd.DataFrame) -> pd.Series:
    """Serie de cierre indexada por open_time horario (para forward returns)."""
    return price_1h.set_index("open_time")["close"].sort_index()


def analyze_funding(
    funding_df: pd.DataFrame, price_1h: pd.DataFrame, settings: Settings | None = None
) -> list[FundingHorizonStats]:
    """IC del funding rate (señal 8h) contra el retorno futuro del precio."""
    settings = settings or load_settings()
    fe = settings.funding_edge
    cost = round_trip_cost(settings)
    price = _hourly_price(price_1h)
    times = pd.DatetimeIndex(funding_df["funding_time"])
    sig = funding_df["funding_rate"].to_numpy()
    out = []
    for h in fe.forward_horizons_hours:
        p_now = price.reindex(times).to_numpy()
        p_fut = price.reindex(times + pd.Timedelta(hours=h)).to_numpy()
        fwd = p_fut / p_now - 1.0
        out.append(horizon_stats(sig, fwd, horizon_h=h, cadence_h=8,
                                 n_quantiles=fe.n_quantiles, cost=cost))
    return out


def analyze_basis(
    premium_df: pd.DataFrame, price_1h: pd.DataFrame, settings: Settings | None = None
) -> list[FundingHorizonStats]:
    """IC del basis/premium (señal 1h) contra el retorno futuro del precio."""
    settings = settings or load_settings()
    fe = settings.funding_edge
    cost = round_trip_cost(settings)
    m = (premium_df[["open_time", "premium_close"]]
         .merge(price_1h[["open_time", "close"]], on="open_time", how="inner")
         .sort_values("open_time").reset_index(drop=True))
    price = m["close"]
    sig = m["premium_close"].to_numpy()
    out = []
    for h in fe.forward_horizons_hours:
        fwd = (price.shift(-h) / price - 1.0).to_numpy()  # 1h → h horas = h velas
        out.append(horizon_stats(sig, fwd, horizon_h=h, cadence_h=1,
                                 n_quantiles=fe.n_quantiles, cost=cost))
    return out
