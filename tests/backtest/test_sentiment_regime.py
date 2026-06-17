"""Tests de los 3 métodos de sentimiento por régimen."""

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_settings
from backtest.sentiment_regime import (
    backtest_long_flat,
    build_daily,
    label_regime,
    mr_gated_ic,
    regime_ic,
    regime_stats,
    vol_scaling,
)

SR = load_settings().sentiment_regime


def _df(closes, fngs):
    dates = pd.date_range("2023-01-01", periods=len(closes), freq="D", tz="UTC")
    return build_daily(pd.Series(closes, index=dates, dtype=float),
                       pd.Series(fngs, index=dates, dtype=float))


class TestRegimeLabel:
    def test_fronteras(self):
        assert label_regime(10, SR) == "ExtFear"
        assert label_regime(30, SR) == "Fear"
        assert label_regime(50, SR) == "Neutral"
        assert label_regime(60, SR) == "Greed"
        assert label_regime(90, SR) == "ExtGreed"


class TestBacktestLongFlat:
    def test_uptrend_largo_es_rentable(self):
        close = 100 * 1.01 ** np.arange(120)  # +1%/día
        df = _df(close, [50] * 120)
        m = backtest_long_flat(df, pd.Series(True, index=df.index),
                               one_way=0.0006, pf_min=1.15, name="x")
        assert m.total_return > 0 and m.n_trades == 1 and m.is_edge

    def test_sin_mercado_no_opera(self):
        df = _df(100 * 1.01 ** np.arange(60), [50] * 60)
        m = backtest_long_flat(df, pd.Series(False, index=df.index),
                               one_way=0.0006, pf_min=1.15, name="x")
        assert m.n_trades == 0 and m.total_return == pytest.approx(0.0)
        assert m.is_edge is False


class TestMethods:
    def test_regime_stats_y_ic_corren(self):
        rng = np.random.default_rng(0)
        n = 400
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        df = _df(close, rng.integers(5, 95, n))
        stats = regime_stats(df, SR)
        assert len(stats) == 5
        ic, t = regime_ic(df, SR)
        assert -1.0 <= ic <= 1.0

    def test_mr_gated_ic_tres_subsets(self):
        rng = np.random.default_rng(1)
        n = 400
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        df = _df(close, rng.integers(5, 95, n))
        res = mr_gated_ic(df, SR)
        assert [r.subset for r in res] == ["todos", "extremo", "normal"]

    def test_vol_scaling_reduce_exposicion_en_codicia(self):
        close = 100 * 1.005 ** np.arange(120)        # mercado al alza
        df = _df(close, [90] * 120)                  # codicia extrema constante
        bh, sc = vol_scaling(df, SR, one_way=0.0)
        # menos exposición en la subida → menos retorno que buy-and-hold
        assert sc.total_return < bh.total_return
