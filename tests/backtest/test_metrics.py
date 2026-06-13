"""Tests de las métricas: valores de referencia calculados a mano.

Las funciones son puras, así que se verifican con entradas pequeñas cuyo
resultado conocemos exactamente.
"""

import math

import numpy as np
import pytest

from backtest.metrics import (
    bar_returns,
    bars_per_year,
    cagr,
    compute_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)


class TestBarsPerYear:
    def test_5m(self):
        assert bars_per_year("5m") == 105120

    def test_1h(self):
        assert bars_per_year("1h") == 8760

    def test_1d(self):
        assert bars_per_year("1d") == 365

    def test_15m(self):
        assert bars_per_year("15m") == 35040

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            bars_per_year("5x")


class TestTotalReturn:
    def test_simple(self):
        assert total_return([100.0, 150.0]) == pytest.approx(0.5)

    def test_loss(self):
        assert total_return([100.0, 80.0]) == pytest.approx(-0.2)

    def test_too_short(self):
        assert total_return([100.0]) == 0.0


class TestCAGR:
    def test_doubles_in_one_year(self):
        # 366 puntos diarios = 365 días = 1 año; x2 → CAGR 100%.
        eq = np.linspace(100.0, 200.0, 366)
        assert cagr(eq, "1d") == pytest.approx(1.0, abs=1e-9)

    def test_ruin_is_minus_one(self):
        assert cagr([100.0, 0.0], "1d") == -1.0


class TestBarReturns:
    def test_values(self):
        r = bar_returns([100.0, 110.0, 99.0])
        assert r == pytest.approx([0.1, -0.1])


class TestMaxDrawdown:
    def test_known(self):
        # picos: 100,120,120,150 ; peor dd en 90 = (120-90)/120 = 0.25
        assert max_drawdown([100.0, 120.0, 90.0, 150.0]) == pytest.approx(0.25)

    def test_monotonic_no_drawdown(self):
        assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


class TestSharpe:
    def test_zero_variance_is_zero(self):
        # crecimiento perfectamente constante: std de retornos = 0 → Sharpe 0.
        assert sharpe_ratio([100.0, 110.0, 121.0], "1d") == 0.0

    def test_positive_drift_positive_sharpe(self):
        eq = [100.0, 101.0, 100.5, 102.0, 103.0]
        assert sharpe_ratio(eq, "1d") > 0

    def test_negative_drift_negative_sharpe(self):
        eq = [100.0, 99.0, 99.5, 98.0, 97.0]
        assert sharpe_ratio(eq, "1d") < 0

    def test_annualization_scales(self):
        eq = [100.0, 101.0, 100.5, 102.0, 103.0]
        # Más barras por año (5m) → mayor factor de anualización que 1d.
        assert abs(sharpe_ratio(eq, "5m")) > abs(sharpe_ratio(eq, "1d"))


class TestSortino:
    def test_no_downside_is_inf(self):
        assert math.isinf(sortino_ratio([100.0, 110.0, 121.0], "1d"))

    def test_with_downside_finite_positive(self):
        eq = [100.0, 102.0, 101.0, 104.0]
        s = sortino_ratio(eq, "1d")
        assert math.isfinite(s) and s > 0


class TestWinRate:
    def test_half(self):
        assert win_rate([10.0, -5.0, 20.0, -5.0]) == pytest.approx(0.5)

    def test_empty(self):
        assert win_rate([]) == 0.0


class TestProfitFactor:
    def test_known(self):
        # ganancias 30, pérdidas 10 → 3.0
        assert profit_factor([10.0, -5.0, 20.0, -5.0]) == pytest.approx(3.0)

    def test_no_losses_is_inf(self):
        assert math.isinf(profit_factor([10.0, 20.0]))

    def test_no_trades_is_zero(self):
        assert profit_factor([]) == 0.0


class TestComputeMetrics:
    def test_composes_all_fields(self):
        eq = [100.0, 120.0, 90.0, 150.0]
        pnls = [10.0, -5.0, 20.0, -5.0]
        bars = [3, 2, 4, 1]
        m = compute_metrics(eq, pnls, bars, bars_in_market=2, timeframe="1h")
        assert m.n_trades == 4
        assert m.win_rate == pytest.approx(0.5)
        assert m.profit_factor == pytest.approx(3.0)
        assert m.max_drawdown == pytest.approx(0.25)
        assert m.total_return == pytest.approx(0.5)
        assert m.exposure == pytest.approx(2 / 4)  # bars_in_market / total_bars
        assert m.expectancy == pytest.approx(np.mean(pnls))
        assert m.avg_bars_held == pytest.approx(np.mean(bars))
