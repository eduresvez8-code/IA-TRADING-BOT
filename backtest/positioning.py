"""Investigación de POSICIONAMIENTO (2026-07): flujo taker + OI + long/short ratio.

Contexto: hasta ahora estas hipótesis eran NO-backtesteables porque la API de
Binance solo sirve ~30 días de open interest / long-short ratio. El hallazgo de
infraestructura de esta investigación es que Binance Vision (data.binance.vision)
publica el histórico COMPLETO a 5 minutos, gratis, desde ~2021-12
(`src/data/download_metrics.py` lo descarga). Eso desbloqueó tres familias:

    E1  Flujo agresor (proxy CVD):   imb = (2·taker_buy − vol) / vol por vela.
    E2  OI + funding + L/S ratios a 1h (divergencias, squeeze, smart-vs-dumb).
    E2b Posicionamiento a frecuencia diaria (menos turnover → menos costos).

RESULTADO (protocolo honesto: config elegida SOLO en train < 2024-12-15, test
medido UNA vez): ninguna familia pasó el listón pre-registrado (Sharpe > 0.5 en
train Y test). El flujo taker contrarian tiene IC real (t≈−4..−6 en los 5
majors) pero NO cubre el costo taker — mismo patrón que la reversión a VWAP.
Ver el informe de la sesión 2026-07-06 para las tablas completas.

Este módulo conserva las FUNCIONES PURAS del estudio para reproducirlo o
extenderlo (p. ej. si algún día hay ejecución maker). Sin estado, sin I/O:
igual que backtest/metrics.py, cada función es testeable con valores a mano.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def taker_imbalance(volume: pd.Series, taker_buy: pd.Series) -> pd.Series:
    """Desequilibrio comprador/vendedor agresivo por vela, en [-1, +1].

    imb = (compra_agresiva − venta_agresiva) / volumen
        = (taker_buy − (vol − taker_buy)) / vol = (2·taker_buy − vol) / vol.

    +1 = todo el volumen fue compra a mercado; −1 = todo venta a mercado.
    Es el proxy de CVD (cumulative volume delta) que las klines públicas
    permiten construir gratis, sin stream L2. Velas con volumen 0 → NaN
    (no hay información, no un desequilibrio de 0).
    """
    vol = volume.replace(0, np.nan)
    return (2.0 * taker_buy - volume) / vol


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """z-score móvil causal: (x_t − media_ventana) / std_ventana.

    Normaliza features con escalas distintas (OI en contratos, ratios, funding)
    a unidades comparables de "sorpresa". Solo usa datos ≤ t (rolling), así que
    es causal por construcción. std 0 → NaN (sin varianza no hay sorpresa).
    """
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd.replace(0.0, np.nan)


def threshold_positions(z: pd.Series, threshold: float, direction: int) -> pd.Series:
    """Posición objetivo por barra a partir de un z-score: {-1, 0, +1}.

    direction=+1 (momentum): z > umbral → LONG; z < −umbral → SHORT.
    direction=−1 (contrarian): lo contrario (fade de la señal).
    |z| ≤ umbral → plano (0): sin convicción no se paga costo.
    NaN en z → 0 (sin dato no hay posición; jamás se adivina).
    """
    if direction not in (1, -1):
        raise ValueError(f"direction debe ser +1 o -1, no {direction!r}")
    pos = np.where(z > threshold, float(direction),
                   np.where(z < -threshold, float(-direction), 0.0))
    return pd.Series(pos, index=z.index).fillna(0.0)


def make_zscore_decider(closes: np.ndarray, atrs: np.ndarray, z: np.ndarray, *,
                        threshold: float, direction: int, atr_mult: float,
                        exit_zscore_abs: float):
    """Decider para BacktestEngine: umbral de z-score + stop ATR real, sin techo.

    Traducción de `threshold_positions()` a la estructura "cortar pérdidas
    rápido, dejar correr las ganancias" (2026-07-08). Diferencias deliberadas:

      1. ENTRADA fresca: solo cuando la condición |z|>threshold se ACTIVA
         (pasa de falsa en la vela previa a verdadera). Evita el churn de
         reentradas inmediatas tras un stop mientras el z sigue extremo —
         el mismo diseño que make_funding_decider.
      2. STOP de riesgo real: close ∓ atr_mult·ATR de la vela de decisión.
         threshold_positions no tenía freno de pérdida; aquí el tamaño lo
         dimensiona el motor por la distancia al stop (riesgo fijo % equity).
      3. SALIDA por reversión: la posición se sostiene MIENTRAS la condición
         persista y se cierra cuando el z vuelve a ±exit_zscore_abs (con 0.0:
         cuando cruza el cero — la "sorpresa" se disipó). tp=None → la
         ganancia no tiene techo además del stop.

    direction=+1 (momentum): z>th → LONG, z<−th → SHORT.
    direction=−1 (contrarian): z>th → SHORT, z<−th → LONG.
    NaN en z → sin señal (jamás se adivina).
    """
    if direction not in (1, -1):
        raise ValueError(f"direction debe ser +1 o -1, no {direction!r}")

    def _raw_side(i: int) -> str | None:
        zi = z[i]
        if math.isnan(zi):
            return None
        if zi > threshold:
            return "LONG" if direction == 1 else "SHORT"
        if zi < -threshold:
            return "SHORT" if direction == 1 else "LONG"
        return None

    def _stop(side: str, i: int) -> float | None:
        a = atrs[i]
        if a is None or math.isnan(a) or a <= 0:
            return None
        return closes[i] - atr_mult * a if side == "LONG" else closes[i] + atr_mult * a

    def decider(i, position_side, score, ts):
        side = _raw_side(i)
        if position_side is None:
            # Entrada fresca: la condición no estaba activa en la vela previa.
            if side is None or (i > 0 and _raw_side(i - 1) == side):
                return None
            stop = _stop(side, i)
            if stop is None:
                return None
            return ("enter", side, 1.0, stop, None)   # tp None → sin techo
        # Sosteniendo: salir cuando el z revierte a la banda de salida.
        zi = z[i]
        if math.isnan(zi):
            return None
        # ¿Qué signo de z sostiene la posición? LONG con dir=+1 vive en z>0;
        # LONG con dir=−1 vive en z<0 (se compró el pánico) — y viceversa.
        holds_positive_z = (position_side == "LONG") == (direction == 1)
        if holds_positive_z and zi <= exit_zscore_abs:
            return ("exit",)
        if not holds_positive_z and zi >= -exit_zscore_abs:
            return ("exit",)
        return None

    return decider


def net_strategy_returns(pos: pd.Series, ret: pd.Series,
                         cost_per_side: float) -> pd.Series:
    """PnL por barra NETO de costos: pos_t · ret_{t+1} − |Δpos_t| · costo_lado.

    Honestidad temporal: la posición decidida al CIERRE de t gana el retorno de
    la barra SIGUIENTE (ret.shift(-1)), nunca el de la barra que generó la
    señal. Cada cambio de exposición paga un lado de costo (entrar 0→1 = 1 lado;
    voltear −1→+1 = 2 lados), que es como cobra el exchange en la práctica.
    """
    pos = pos.fillna(0.0)
    fwd = ret.shift(-1)
    turnover = pos.diff().abs().fillna(pos.abs())
    return (pos * fwd - turnover * cost_per_side).dropna()


def annualized_sharpe(bar_returns: pd.Series, bars_per_year: float) -> float:
    """Sharpe anualizado de una serie de retornos por barra (no equity).

    Complementa a metrics.sharpe_ratio (que recibe curva de equity): aquí el
    insumo natural del estudio vectorizado son los retornos por barra. std
    poblacional (ddof=0), misma convención que backtest/metrics.py. Serie
    corta (<2) o sin varianza → 0.0 por convención.
    """
    r = bar_returns.dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=0)
    # isclose y no ==0: la std de una serie CONSTANTE da ~1e-18 por redondeo
    # flotante, y dividir por eso fabricaría un Sharpe de 10^17 sin sentido.
    if math.isclose(sd, 0.0, abs_tol=1e-12):
        return 0.0
    return float(r.mean() / sd * math.sqrt(bars_per_year))


def split_by_date(obj: pd.Series | pd.DataFrame, split_ts: pd.Timestamp):
    """(train, test) por fecha: train < split_ts ≤ test. Sin solape posible.

    El protocolo anti-selección exige que TODA elección de configuración use
    solo el tramo train y que test se mida una única vez. Centralizar el corte
    aquí evita el error de recortar con >= en un sitio y > en otro.
    """
    return obj[obj.index < split_ts], obj[obj.index >= split_ts]
