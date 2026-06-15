"""Tests de los indicadores técnicos contra valores de referencia.

Estrategia de testing:
- EMA: valores calculados a mano con α=0.5 (period=3), exactos en IEEE 754.
- RSI: valores calculados con fracciones exactas usando suavizado de Wilder
  (α=1/3, period=3) sobre una serie pequeña de 6 precios.
- ATR: mercado plano con rango constante → ATR debe converger al rango.
"""

import pandas as pd
import pytest

from src.quant.indicators import (
    atr,
    bollinger_bands,
    donchian_channel,
    ema,
    rsi,
    sma,
)


# ─── EMA ──────────────────────────────────────────────────────────────────────


class TestEMA:
    def test_known_values_period3(self):
        """Valores calculados a mano con α = 2/(3+1) = 0.5.

        y_0 = 10         (semilla)
        y_1 = 0.5*11 + 0.5*10    = 10.5
        y_2 = 0.5*12 + 0.5*10.5  = 11.25   ← primer válido (min_periods=3)
        y_3 = 0.5*13 + 0.5*11.25 = 12.125
        y_4 = 0.5*14 + 0.5*12.125 = 13.0625
        y_5 = 0.5*15 + 0.5*13.0625 = 14.03125
        """
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        result = ema(prices, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(11.25, rel=1e-9)
        assert result.iloc[3] == pytest.approx(12.125, rel=1e-9)
        assert result.iloc[4] == pytest.approx(13.0625, rel=1e-9)
        assert result.iloc[5] == pytest.approx(14.03125, rel=1e-9)

    def test_nan_before_period(self):
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(prices, period=5)

        assert all(pd.isna(result.iloc[i]) for i in range(4))
        assert not pd.isna(result.iloc[4])

    def test_constant_series_equals_constant(self):
        """EMA de una constante debe ser esa constante."""
        prices = pd.Series([7.5] * 30)
        result = ema(prices, period=10)
        assert result.dropna().iloc[-1] == pytest.approx(7.5)

    def test_output_same_length_and_index(self):
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[10, 20, 30, 40, 50])
        result = ema(prices, period=3)
        assert len(result) == len(prices)
        assert list(result.index) == [10, 20, 30, 40, 50]

    def test_uptrend_ema_fast_above_slow(self):
        """En tendencia alcista fuerte, EMA rápida debe superar a la lenta."""
        prices = pd.Series([float(i) for i in range(1, 51)])
        fast = ema(prices, period=9)
        slow = ema(prices, period=21)
        assert fast.iloc[-1] > slow.iloc[-1]


# ─── RSI ──────────────────────────────────────────────────────────────────────


class TestRSI:
    def test_nan_before_period_plus_one(self):
        """Con period=3 necesitamos 3 deltas → primero válido en índice 3."""
        prices = pd.Series([10.0, 11.0, 10.0, 12.0, 11.0, 13.0])
        result = rsi(prices, period=3)

        assert all(pd.isna(result.iloc[i]) for i in range(3))
        assert not pd.isna(result.iloc[3])

    def test_known_values_period3(self):
        """Valores calculados a mano con α = 1/3 (Wilder, period=3).

        Precios: [10, 11, 10, 12, 11, 13]
        Deltas:  [NaN, +1, -1, +2, -1, +2]

        avg_gain en [3]: ewm de gains no-NaN [1,0,2] → 10/9
        avg_loss en [3]: ewm de losses no-NaN [0,1,0] → 2/9
        RS[3] = (10/9)/(2/9) = 5 → RSI = 100 - 100/6 ≈ 83.333

        avg_gain en [4]: (1/3)*0 + (2/3)*(10/9) = 20/27
        avg_loss en [4]: (1/3)*1 + (2/3)*(2/9) = 13/27
        RS[4] = 20/13 → RSI = 100 - 1300/33 ≈ 60.606

        avg_gain en [5]: (1/3)*2 + (2/3)*(20/27) = 94/81
        avg_loss en [5]: (1/3)*0 + (2/3)*(13/27) = 26/81
        RS[5] = 94/26 = 47/13 → RSI = 100 - 1300/60 ≈ 78.333
        """
        prices = pd.Series([10.0, 11.0, 10.0, 12.0, 11.0, 13.0])
        result = rsi(prices, period=3)

        assert result.iloc[3] == pytest.approx(100 - 100 / 6, rel=1e-6)
        assert result.iloc[4] == pytest.approx(100 - 1300 / 33, rel=1e-6)
        assert result.iloc[5] == pytest.approx(100 - 1300 / 60, rel=1e-6)

    def test_rsi_range_0_to_100(self):
        """El RSI siempre debe estar en [0, 100]."""
        prices = pd.Series([100.0 + i * 0.3 - (i % 5) * 0.8 for i in range(60)])
        result = rsi(prices, period=14)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_uptrend_rsi_high(self):
        """Precios estrictamente crecientes → RSI cerca de 100."""
        prices = pd.Series([float(i) for i in range(1, 51)])
        result = rsi(prices, period=14)
        assert result.dropna().iloc[-1] > 70

    def test_downtrend_rsi_low(self):
        """Precios estrictamente decrecientes → RSI cerca de 0."""
        prices = pd.Series([float(50 - i) for i in range(50)])
        result = rsi(prices, period=14)
        assert result.dropna().iloc[-1] < 30

    def test_insufficient_data_returns_all_nan(self):
        """Con menos de period+1 valores no hay ningún RSI válido."""
        prices = pd.Series([1.0, 2.0, 3.0])
        result = rsi(prices, period=14)
        assert result.dropna().empty


# ─── ATR ──────────────────────────────────────────────────────────────────────


class TestATR:
    def _constant_ohlc_df(self, n: int, h: float, l: float, c: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [c] * n,
                "high": [h] * n,
                "low": [l] * n,
                "close": [c] * n,
                "volume": [1000.0] * n,
            }
        )

    def test_constant_range_converges(self):
        """Con H=11, L=9, C=10 en todas las velas, TR=2 siempre → ATR=2."""
        df = self._constant_ohlc_df(30, h=11.0, l=9.0, c=10.0)
        result = atr(df, period=3)

        assert not pd.isna(result.iloc[-1])
        assert result.iloc[-1] == pytest.approx(2.0, rel=1e-6)

    def test_atr_always_positive(self):
        """ATR debe ser siempre > 0 en datos con movimiento."""
        n = 40
        df = pd.DataFrame(
            {
                "open": [10.0 + i * 0.1 for i in range(n)],
                "high": [11.0 + i * 0.1 for i in range(n)],
                "low": [9.0 + i * 0.1 for i in range(n)],
                "close": [10.0 + i * 0.1 for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        result = atr(df, period=14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_nan_before_period(self):
        df = self._constant_ohlc_df(20, h=11.0, l=9.0, c=10.0)
        result = atr(df, period=14)

        # Primeros 13 índices son NaN (min_periods=14 → necesita 14 valores)
        assert all(pd.isna(result.iloc[i]) for i in range(13))
        assert not pd.isna(result.iloc[13])

    def test_higher_range_higher_atr(self):
        """Mayor rango H-L → mayor ATR."""
        n = 30
        narrow = pd.DataFrame(
            {"open": [10.0] * n, "high": [10.5] * n, "low": [9.5] * n, "close": [10.0] * n, "volume": [1.0] * n}
        )
        wide = pd.DataFrame(
            {"open": [10.0] * n, "high": [12.0] * n, "low": [8.0] * n, "close": [10.0] * n, "volume": [1.0] * n}
        )
        assert atr(narrow, 3).iloc[-1] < atr(wide, 3).iloc[-1]


# ─── SMA ──────────────────────────────────────────────────────────────────────


class TestSMA:
    def test_known_values_period3(self):
        prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        result = sma(prices, period=3)
        assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(11.0)  # (10+11+12)/3
        assert result.iloc[3] == pytest.approx(12.0)  # (11+12+13)/3
        assert result.iloc[4] == pytest.approx(13.0)  # (12+13+14)/3


# ─── Bollinger ────────────────────────────────────────────────────────────────


class TestBollinger:
    def test_bandas_simetricas_alrededor_de_la_media(self):
        # Serie con desviación poblacional conocida: [9,10,11], σ=√(2/3).
        close = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0])
        mid, up, lo = bollinger_bands(close, period=3, num_std=2.0)
        import math
        sd = math.sqrt(2.0 / 3.0)  # poblacional de {9,10,11}
        assert mid.iloc[2] == pytest.approx(10.0)
        assert up.iloc[2] == pytest.approx(10.0 + 2.0 * sd)
        assert lo.iloc[2] == pytest.approx(10.0 - 2.0 * sd)
        # la media siempre equidista de ambas bandas
        assert (up.iloc[2] - mid.iloc[2]) == pytest.approx(mid.iloc[2] - lo.iloc[2])

    def test_mas_desviaciones_bandas_mas_anchas(self):
        close = pd.Series([float(x) for x in range(1, 21)])
        _, up2, lo2 = bollinger_bands(close, 20, 2.0)
        _, up3, lo3 = bollinger_bands(close, 20, 3.0)
        assert up3.iloc[-1] > up2.iloc[-1] and lo3.iloc[-1] < lo2.iloc[-1]


# ─── Donchian ─────────────────────────────────────────────────────────────────


class TestDonchian:
    def test_max_y_min_de_la_ventana(self):
        high = pd.Series([10.0, 12.0, 11.0, 9.0, 15.0])
        low = pd.Series([8.0, 9.0, 7.0, 6.0, 10.0])
        up, lo = donchian_channel(high, low, period=3)
        assert pd.isna(up.iloc[1]) and pd.isna(lo.iloc[1])
        assert up.iloc[2] == pytest.approx(12.0)  # max(10,12,11)
        assert lo.iloc[2] == pytest.approx(7.0)   # min(8,9,7)
        assert up.iloc[4] == pytest.approx(15.0)  # max(11,9,15)
        assert lo.iloc[4] == pytest.approx(6.0)   # min(7,6,10)

    def test_incluye_la_vela_actual(self):
        # El canal incluye t; el shift anti-look-ahead lo aplica la estrategia.
        high = pd.Series([1.0, 2.0, 3.0, 10.0])
        low = pd.Series([1.0, 2.0, 3.0, 10.0])
        up, _ = donchian_channel(high, low, period=2)
        assert up.iloc[3] == pytest.approx(10.0)  # max(3,10) usa la vela actual
