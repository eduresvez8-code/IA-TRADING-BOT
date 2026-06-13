"""Tests de la estrategia EMA-cross + RSI.

Verifica que compute_signal:
    - devuelve None cuando los datos son insuficientes
    - genera señal positiva en tendencia alcista clara
    - genera señal negativa en tendencia bajista clara
    - siempre produce score en [-1, 1]
    - incluye las features diagnósticas esperadas
"""

import pandas as pd
import pytest

from src.quant.strategy import compute_signal


def _make_df(close_prices: list[float]) -> pd.DataFrame:
    """Construye un DataFrame OHLCV mínimo a partir de cierres."""
    n = len(close_prices)
    c = pd.Series(close_prices)
    return pd.DataFrame(
        {
            "open": c,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": [1000.0] * n,
        }
    )


class TestComputeSignal:
    def test_returns_none_when_too_few_candles(self):
        df = _make_df([100.0] * 10)
        assert compute_signal(df, "BTCUSDT") is None

    def test_returns_signal_with_enough_data(self):
        close = [100.0 + i for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "BTCUSDT")
        assert sig is not None

    def test_bullish_trend_positive_score(self):
        """Tendencia alcista sostenida: EMA9 > EMA21 y RSI alto → score > 0."""
        close = [100.0 + i * 0.5 for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "BTCUSDT")
        assert sig is not None
        assert sig.score > 0

    def test_bearish_trend_negative_score(self):
        """Tendencia bajista sostenida: EMA9 < EMA21 y RSI bajo → score < 0."""
        close = [200.0 - i * 0.5 for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "ETHUSDT")
        assert sig is not None
        assert sig.score < 0

    def test_score_within_range(self):
        """El score siempre debe estar en [-1, 1]."""
        # Serie con movimientos bruscos para estresar el normalizador
        import math
        close = [100.0 + math.sin(i * 0.5) * 20 for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "BTCUSDT")
        assert sig is not None
        assert -1.0 <= sig.score <= 1.0

    def test_signal_has_required_features(self):
        close = [100.0 + i for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "BTCUSDT")
        assert sig is not None
        for key in ("ema_fast", "ema_slow", "rsi", "atr", "ema_diff_pct", "ema_score", "rsi_score"):
            assert key in sig.features, f"feature '{key}' ausente en Signal.features"

    def test_signal_symbol_matches(self):
        close = [100.0 + i for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "SOLUSDT")
        assert sig is not None
        assert sig.symbol == "SOLUSDT"

    def test_signal_strategy_name(self):
        close = [100.0 + i for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "BTCUSDT")
        assert sig is not None
        assert sig.strategy == "ema_cross_rsi"

    def test_ema_diff_pct_sign_matches_score(self):
        """El signo del spread EMA debe coincidir con el signo del score."""
        close = [100.0 + i for i in range(100)]
        df = _make_df(close)
        sig = compute_signal(df, "BTCUSDT")
        assert sig is not None
        ema_diff = sig.features["ema_diff_pct"]
        assert (ema_diff > 0) == (sig.score > 0)
