"""Estrategia EMA-cross + RSI: primera señal cuantitativa del bot.

Consume un DataFrame de velas cerradas y emite un Signal normalizado
en [-1, +1]. El diseño es funcional: compute_signal() no tiene estado
propio y puede evaluarse sobre cualquier ventana histórica.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from src.core.config import load_settings
from src.core.models import Signal
from src.quant.indicators import atr, ema, rsi

STRATEGY_NAME = "ema_cross_rsi"


def compute_signal(candles_df: pd.DataFrame, symbol: str) -> Signal | None:
    """Calcula la señal de trading sobre las últimas velas cerradas.

    Lógica:
        1. EMA rápida (9) vs lenta (21) → spread porcentual → tanh → ema_score
        2. RSI(14) centrado en 50 → rsi_score
        3. score = ema_weight * ema_score + (1 - ema_weight) * rsi_score

    Args:
        candles_df: DataFrame con columnas 'open','high','low','close','volume'.
                    Solo velas cerradas; el orden debe ser cronológico ascendente.
        symbol:     Par de trading, ej. "BTCUSDT".

    Returns:
        Signal con score en [-1, +1], o None si no hay suficientes datos.
    """
    cfg = load_settings()
    q = cfg.quant

    # Mínimo de filas para que todos los indicadores tengan al menos
    # una observación válida en el último índice.
    min_rows = q.ema_slow_period + q.rsi_period
    if len(candles_df) < min_rows:
        return None

    close = candles_df["close"]

    ema_fast_s = ema(close, q.ema_fast_period)
    ema_slow_s = ema(close, q.ema_slow_period)
    rsi_s = rsi(close, q.rsi_period)
    atr_s = atr(candles_df, cfg.risk.atr_period)

    last_ema_fast = ema_fast_s.iloc[-1]
    last_ema_slow = ema_slow_s.iloc[-1]
    last_rsi = rsi_s.iloc[-1]
    last_atr = atr_s.iloc[-1]

    # Cualquier NaN en los indicadores → datos insuficientes para señal fiable
    if any(
        math.isnan(v)
        for v in (last_ema_fast, last_ema_slow, last_rsi, last_atr)
    ):
        return None

    # Spread porcentual entre EMAs, luego squash con tanh al rango (-1, 1).
    # Factor 50: un spread del 1% da tanh(0.5)≈0.46; del 3% da tanh(1.5)≈0.91.
    ema_diff_pct = (last_ema_fast - last_ema_slow) / last_ema_slow
    ema_score = math.tanh(ema_diff_pct * 50)

    # RSI centrado en 50 y escalado a (-1, 1). Lineal: RSI=70 → +0.4, RSI=30 → -0.4.
    rsi_score = (last_rsi - 50.0) / 50.0

    raw_score = q.ema_weight * ema_score + (1.0 - q.ema_weight) * rsi_score

    # Clamp defensivo: la suma de componentes ya está en (-1,1), pero si los
    # pesos fueran >1 por error de configuración el clamp lo contiene.
    score = max(-1.0, min(1.0, raw_score))

    return Signal(
        symbol=symbol,
        score=score,
        strategy=STRATEGY_NAME,
        timestamp=datetime.now(timezone.utc),
        features={
            "ema_fast": float(last_ema_fast),
            "ema_slow": float(last_ema_slow),
            "ema_diff_pct": float(ema_diff_pct * 100),  # en % para legibilidad
            "rsi": float(last_rsi),
            "atr": float(last_atr),
            "ema_score": float(ema_score),
            "rsi_score": float(rsi_score),
        },
    )
