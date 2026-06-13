"""Indicadores de análisis técnico: EMA, RSI, ATR.

Funciones puras sobre pd.Series / pd.DataFrame. Sin efectos secundarios
ni dependencias de configuración. Devuelven una Series con el mismo índice
que la entrada, con NaN donde no hay suficientes datos.
"""

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Media Móvil Exponencial con factor estándar α = 2/(period+1).

    Args:
        series: Precios de cierre (u otra serie numérica).
        period: Número de períodos para calcular α.

    Returns:
        EMA con NaN en las primeras period-1 posiciones.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Suavizado de Wilder: α = 1/period  (equivalente a com = period-1).

    Más conservador que la EMA estándar (α más pequeño → más inercia).
    Usado internamente por ATR. Para RSI usar rsi() directamente.
    """
    return series.ewm(com=period - 1, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder, 1978).

    RSI = 100 - 100 / (1 + RS)   donde   RS = avg_gain / avg_loss

    Ambas medias usan suavizado de Wilder con ignore_na=True para saltar
    el NaN inicial que produce diff() — no existe variación antes del
    primer precio.

    Casos límite IEEE 754 (se resuelven sin branching):
        avg_loss = 0.0 → RS = inf  → RSI = 100  (pura subida en el período)
        avg_gain = 0.0 → RS = 0    → RSI = 0    (pura bajada en el período)
        ambos = 0.0    → RS = NaN  → RSI = NaN  (mercado completamente plano)

    Args:
        series: Precios de cierre.
        period: Número de períodos (Wilder recomendaba 14).

    Returns:
        RSI en [0, 100]. NaN donde no hay suficientes datos.
    """
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # ignore_na=True: el NaN de delta[0] se salta sin afectar la recursión.
    # min_periods cuenta observaciones no-NaN, por lo que el primer RSI válido
    # aparece en el índice `period` (después de `period` variaciones de precio).
    avg_gain = gains.ewm(
        com=period - 1, adjust=False, min_periods=period, ignore_na=True
    ).mean()
    avg_loss = losses.ewm(
        com=period - 1, adjust=False, min_periods=period, ignore_na=True
    ).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder, 1978): volatilidad real por vela.

    TR_t = max(H-L,  |H - C_prev|,  |L - C_prev|)
    ATR  = Wilder_smooth(TR, period)

    El TR captura tres tipos de movimiento:
        H - L           rango intradía normal
        |H - C_prev|    gap alcista entre velas
        |L - C_prev|    gap bajista entre velas

    Args:
        df: DataFrame con columnas 'high', 'low', 'close'.
        period: Número de períodos (Wilder recomendaba 14).

    Returns:
        ATR en las mismas unidades que el precio. NaN en los primeros
        period-1 índices.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    # max(axis=1) tiene skipna=True por defecto: en la primera vela,
    # donde prev_close = NaN, usa solo H-L como True Range.
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return _wilder_smooth(true_range, period)
