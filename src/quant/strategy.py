"""Estrategia EMA-cross + RSI: primera señal cuantitativa del bot.

Consume un DataFrame de velas cerradas y emite un Signal normalizado
en [-1, +1]. El diseño es funcional: compute_signal() no tiene estado
propio y puede evaluarse sobre cualquier ventana histórica.

Dos puntos de entrada que comparten EXACTAMENTE la misma matemática:
    - compute_signal()        → un Signal con la última vela (uso en vivo).
    - compute_signal_series() → una pd.Series de scores para todo el histórico
                                (uso en backtesting, O(n) en vez de O(n²)).
El núcleo `_scores_from_indicators()` es el único lugar donde vive la fórmula,
así que el motor en vivo y el backtester nunca pueden divergir.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.core.models import Signal
from src.quant.indicators import atr, ema, rsi, sma

STRATEGY_NAME = "ema_cross_rsi"


def _scores_from_indicators(ema_fast, ema_slow, rsi_vals, ema_weight,
                            squash_factor):
    """Convierte valores de indicadores en componentes de score.

    Funciona igual con escalares (float) o con pd.Series gracias a numpy:
    np.tanh y la aritmética se vectorizan transparentemente. Es el ÚNICO
    sitio donde se define la fórmula de la señal.

    Returns:
        (raw_score, ema_diff_pct, ema_score, rsi_score) — sin clamp todavía.
    """
    # Spread porcentual entre EMAs, squash con tanh al rango (-1, 1).
    # squash_factor (config quant.score_squash_factor; histórico 50): un spread
    # del 1% da tanh(0.5)≈0.46; del 3% da tanh(1.5)≈0.91. Vive en settings.yaml
    # porque fija la ESCALA del score e interactúa con los umbrales de confluencia.
    ema_diff_pct = (ema_fast - ema_slow) / ema_slow
    ema_score = np.tanh(ema_diff_pct * squash_factor)

    # RSI centrado en 50 y escalado a (-1, 1). Lineal: RSI=70 → +0.4, RSI=30 → -0.4.
    rsi_score = (rsi_vals - 50.0) / 50.0

    raw_score = ema_weight * ema_score + (1.0 - ema_weight) * rsi_score
    return raw_score, ema_diff_pct, ema_score, rsi_score


def compute_signal_series(candles_df: pd.DataFrame, settings=None) -> pd.Series:
    """Score [-1,+1] vectorizado para CADA vela del DataFrame.

    Los indicadores son causales (el valor en t solo depende de velas ≤ t),
    por eso calcularlos sobre toda la serie de una vez NO introduce sesgo de
    anticipación: el score en t es idéntico al que daría compute_signal() sobre
    la ventana que termina en t. Esto convierte el backtest de O(n²) a O(n).

    Args:
        candles_df: DataFrame con columnas 'open','high','low','close','volume',
                    en orden cronológico ascendente.
        settings:   config inyectable; por defecto la global (load_settings). Que sea
                    inyectable DESACOPLA el motor/tests de la config enviada: el
                    backtest usa SU cfg, no la global (evita que cambiar settings.yaml
                    altere silenciosamente el comportamiento de todos los tests).

    Returns:
        pd.Series alineada al índice de entrada, con NaN donde los indicadores
        aún no tienen suficientes datos.
    """
    cfg = settings or load_settings()
    q = cfg.quant

    close = candles_df["close"]
    _ma = sma if q.ma_type == "sma" else ema
    ema_fast_s = _ma(close, q.ema_fast_period)
    ema_slow_s = _ma(close, q.ema_slow_period)
    rsi_s = rsi(close, q.rsi_period)

    raw, _, _, _ = _scores_from_indicators(
        ema_fast_s, ema_slow_s, rsi_s, q.ema_weight, q.score_squash_factor)

    # Clamp defensivo idéntico al de compute_signal. np.clip conserva los NaN.
    return raw.clip(lower=-1.0, upper=1.0)


def compute_signal(candles_df: pd.DataFrame, symbol: str, settings=None) -> Signal | None:
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
    cfg = settings or load_settings()
    q = cfg.quant

    # Mínimo de filas para que todos los indicadores tengan al menos
    # una observación válida en el último índice.
    min_rows = q.ema_slow_period + q.rsi_period
    if len(candles_df) < min_rows:
        return None

    close = candles_df["close"]

    _ma = sma if q.ma_type == "sma" else ema
    ema_fast_s = _ma(close, q.ema_fast_period)
    ema_slow_s = _ma(close, q.ema_slow_period)
    rsi_s = rsi(close, q.rsi_period)
    atr_s = atr(candles_df, cfg.risk.atr_period)

    last_ema_fast = ema_fast_s.iloc[-1]
    last_ema_slow = ema_slow_s.iloc[-1]
    last_rsi = rsi_s.iloc[-1]
    last_atr = atr_s.iloc[-1]

    # Cualquier NaN en los indicadores → datos insuficientes para señal fiable
    if any(
        np.isnan(v)
        for v in (last_ema_fast, last_ema_slow, last_rsi, last_atr)
    ):
        return None

    raw_score, ema_diff_pct, ema_score, rsi_score = _scores_from_indicators(
        last_ema_fast, last_ema_slow, last_rsi, q.ema_weight, q.score_squash_factor
    )

    # Clamp defensivo: la suma de componentes ya está en (-1,1), pero si los
    # pesos fueran >1 por error de configuración el clamp lo contiene.
    score = float(max(-1.0, min(1.0, raw_score)))

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
