"""Tests del edge test cross-sectional: factor y agregación de IC.

Panel sintético con drifts distintos por activo: el momentum pasado y el retorno
futuro quedan ambos ordenados por el drift → el IC cross-sectional debe ser ≈1.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.cross_sectional import (
    analyze,
    cross_sectional_ic,
    forward_return,
    momentum_factor,
)


def _panel(drifts, n_days=260, seed=None):
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    data = {}
    rng = np.random.default_rng(seed)
    for i, d in enumerate(drifts):
        noise = rng.normal(0, 0.0, n_days) if seed is None else rng.normal(0, 0.002, n_days)
        data[f"A{i:02d}"] = 100 * np.exp(np.cumsum(np.full(n_days, d) + noise))
    return pd.DataFrame(data, index=dates)


class TestFactor:
    def test_momentum_es_retorno_de_lookback(self):
        close = pd.DataFrame({"A": [100.0, 110, 121, 133.1, 146.41]})
        f = momentum_factor(close, lookback=2, skip=0, vol_adjust=False, vol_lookback=2)
        # en t=2: 121/100-1 = 0.21
        assert f["A"].iloc[2] == pytest.approx(0.21)
        assert np.isnan(f["A"].iloc[1])  # warmup

    def test_forward_return(self):
        close = pd.DataFrame({"A": [100.0, 110, 121]})
        fr = forward_return(close, 1)
        assert fr["A"].iloc[0] == pytest.approx(0.10)
        assert np.isnan(fr["A"].iloc[2])


class TestCrossSectionalIC:
    def test_factor_que_rankea_fuerte_es_significativo(self):
        cfg = load_settings()
        # drifts distintos por activo + ruido leve → momentum y futuro muy
        # alineados pero con varianza realista (sin ruido el IC sería 1 idéntico
        # y el t-stat quedaría indefinido por varianza 0).
        close = _panel(np.linspace(-0.006, 0.006, 20), seed=1)
        r = analyze(close, cfg, lookback=30, vol_adjust=False)
        assert r.mean_ic > 0.5           # momentum y futuro alineados por el drift
        assert r.t_stat > 2              # significativo
        assert r.beats_cost is True
        assert r.fold_same_sign == len(r.fold_mean_ics)  # consistente en los tramos
        assert r.avg_universe == pytest.approx(20, abs=1)

    def test_ruido_no_es_significativo(self):
        cfg = load_settings()
        rng = np.random.default_rng(0)
        # random walks independientes: momentum pasado no informa del futuro
        n_days, n_assets = 300, 25
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
        close = pd.DataFrame(
            {f"A{i:02d}": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n_days)))
             for i in range(n_assets)}, index=dates)
        r = analyze(close, cfg, lookback=30, vol_adjust=False)
        assert abs(r.mean_ic) < 0.25
        assert r.beats_cost is False

    def test_min_assets_filtra_fechas_ralas(self):
        cfg = load_settings()
        close = _panel(np.linspace(-0.002, 0.002, 5))  # solo 5 < min_assets (10)
        r = analyze(close, cfg, lookback=30, vol_adjust=False)
        assert r.n_dates == 0  # ninguna fecha alcanza la cross-section mínima
