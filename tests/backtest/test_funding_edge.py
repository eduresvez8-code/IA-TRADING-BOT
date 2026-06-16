"""Tests del edge test de señales no-precio (funding/basis)."""

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.funding_edge import (
    analyze_basis,
    analyze_funding,
    horizon_stats,
    round_trip_cost,
)


def test_round_trip_cost():
    cfg = load_settings()
    # 2 × (comisión + slippage) en fracción.
    esperado = 2 * (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0
    assert round_trip_cost(cfg) == pytest.approx(esperado)


class TestHorizonStats:
    def test_senal_perfecta_es_tradable(self):
        sig = np.linspace(-1, 1, 200)
        fwd = sig * 0.05  # retorno futuro perfectamente alineado, spread grande
        s = horizon_stats(sig, fwd, horizon_h=8, cadence_h=8,
                          n_quantiles=5, cost=0.0012)
        assert s.spearman_ic == pytest.approx(1.0)
        assert s.quantile_spread > 0.0012  # supera el costo
        assert s.tradable is True
        assert s.folds_same_sign == s.n_folds  # consistente en todos los tramos

    def test_sin_relacion_no_es_tradable(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(size=400)
        fwd = rng.normal(scale=0.0001, size=400)  # ruido, sin relación con sig
        s = horizon_stats(sig, fwd, horizon_h=24, cadence_h=8,
                          n_quantiles=5, cost=0.0012)
        assert abs(s.spearman_ic) < 0.2
        assert s.tradable is False

    def test_n_eff_descuenta_solape(self):
        sig = np.linspace(-1, 1, 240)
        fwd = sig * 0.05
        # basis (cadencia 1h) a 24h → solape 24 → n_eff ≈ n/24
        s = horizon_stats(sig, fwd, horizon_h=24, cadence_h=1,
                          n_quantiles=5, cost=0.0012)
        assert s.n_eff == 10


class TestAnalyze:
    def _price_1h(self, n=600, seed=1):
        closes = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.5, n))
        return pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": closes, "high": closes + 1, "low": closes - 1,
            "close": closes, "volume": np.full(n, 1000.0),
        })

    def test_analyze_funding_devuelve_un_stat_por_horizonte(self):
        cfg = load_settings()
        price = self._price_1h()
        ftimes = pd.date_range("2024-01-01", periods=120, freq="8h", tz="UTC")
        funding = pd.DataFrame({
            "funding_time": ftimes,
            "funding_rate": np.random.default_rng(2).normal(0, 1e-4, len(ftimes)),
        })
        stats = analyze_funding(funding, price, cfg)
        assert len(stats) == len(cfg.funding_edge.forward_horizons_hours)
        assert all(-1.0 <= s.spearman_ic <= 1.0 for s in stats)

    def test_analyze_basis_devuelve_un_stat_por_horizonte(self):
        cfg = load_settings()
        price = self._price_1h()
        premium = pd.DataFrame({
            "open_time": price["open_time"],
            "premium_close": np.random.default_rng(3).normal(0, 1e-4, len(price)),
        })
        stats = analyze_basis(premium, price, cfg)
        assert len(stats) == len(cfg.funding_edge.forward_horizons_hours)
        assert all(-1.0 <= s.spearman_ic <= 1.0 for s in stats)
