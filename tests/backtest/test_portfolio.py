"""Tests del portafolio long-short de reversión."""

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.portfolio import backtest_reversal, inverse_vol_weights


class TestInverseVolWeights:
    def test_menor_vol_mayor_peso(self):
        w = inverse_vol_weights(pd.Series({"a": 0.01, "b": 0.04}), max_weight=1.0)
        assert w["a"] > w["b"]
        assert w.sum() == pytest.approx(1.0)

    def test_respeta_el_tope_y_renormaliza(self):
        # 'a' tiene vol minúscula → dominaría; el tope la limita y redistribuye.
        w = inverse_vol_weights(
            pd.Series({"a": 0.001, "b": 0.1, "c": 0.1, "d": 0.1}), max_weight=0.3)
        assert w.max() <= 0.3 + 1e-9
        assert w.sum() == pytest.approx(1.0)


def _panel(returns_fn, n_days=400, n_assets=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    logp = returns_fn(rng, n_days, n_assets)
    close = pd.DataFrame(100 * np.exp(logp),
                         index=dates, columns=[f"A{i:02d}" for i in range(n_assets)])
    qvol = pd.DataFrame(1e6, index=dates, columns=close.columns)  # liquidez uniforme
    return close, qvol


def _ou(phi):
    def f(rng, n_days, n_assets):
        logp = np.zeros((n_days, n_assets))
        for t in range(1, n_days):
            logp[t] = phi * logp[t - 1] + rng.normal(0, 0.1, n_assets)
        return logp
    return f


class TestBacktest:
    def test_corre_y_da_metricas_finitas(self):
        close, qvol = _panel(_ou(0.99))  # casi random walk
        r = backtest_reversal(close, qvol, load_settings(), lookback=30)
        assert r.n_periods > 0
        assert np.isfinite(r.ann_sharpe) and np.isfinite(r.max_drawdown)
        assert r.avg_universe >= 10

    def test_fuerte_reversion_es_rentable(self):
        # OU con phi bajo = fuerte reversión a la media → los perdedores rebotan,
        # los ganadores caen → el long-short de reversión gana.
        close, qvol = _panel(_ou(0.6))
        r = backtest_reversal(close, qvol, load_settings(), lookback=14)
        assert r.total_return > 0
        assert r.profit_factor > 1.0

    def test_tendencia_pura_no_es_cosechable_por_reversion(self):
        # Drifts persistentes (momentum) → la reversión pierde.
        def trend(rng, n_days, n_assets):
            drifts = np.linspace(-0.01, 0.01, n_assets)
            steps = np.tile(drifts, (n_days, 1)) + rng.normal(0, 0.02, (n_days, n_assets))
            return np.cumsum(steps, axis=0)
        close, qvol = _panel(trend)
        r = backtest_reversal(close, qvol, load_settings(), lookback=30)
        assert r.is_edge is False
