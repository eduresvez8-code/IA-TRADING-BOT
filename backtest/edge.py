"""Test de EDGE de la señal quant: ¿predice el retorno futuro mejor que el azar?

El backtest dice CUÁNTO pierde la estrategia, pero mezcla entrada, stops/TP y
costos. Este módulo aísla la pregunta previa a todo: ¿la señal direccional cruda
(`ema_cross_rsi`) tiene PODER PREDICTIVO? Si no lo tiene, ninguna confirmación de
sentimiento ni ajuste de stops puede rescatarla.

Herramientas (estándar de quant research, sin tocar la lógica de trading):

  - Information Coefficient (IC): correlación entre la señal en t y el retorno
    realizado en [t, t+N]. Spearman (rango, robusto a no-linealidad/outliers) y
    Pearson (lineal). IC≈0 ⇒ sin edge.
  - t-stat con muestra EFECTIVA: los retornos forward de velas consecutivas se
    solapan (comparten N-1 barras), así que la muestra independiente es ~n/N. El
    t se calcula con esa n_eff para no inflar la significancia.
  - Monotonicidad por cuantiles: a mayor señal, ¿mayor retorno futuro?
  - Acierto direccional en |señal|≥umbral (donde el bot abriría), comparado con
    la deriva base del mercado.
  - IC por régimen: precio sobre/bajo la EMA lenta.

Todo son funciones puras sobre Series/arrays (como backtest/metrics.py): la señal
en t es causal, por eso correlacionarla con el futuro NO mete look-ahead — el
futuro solo se usa como variable a predecir, nunca para construir la señal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import Settings, load_settings
from src.quant.indicators import ema
from src.quant.strategy import STRATEGY_NAME, compute_signal_series


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Retorno futuro a `horizon` velas: close[t+N]/close[t]-1.

    Las últimas N velas no tienen futuro observable → NaN (se descartan luego).
    """
    return close.shift(-horizon) / close - 1.0


def _aligned(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Pares (a,b) sin NaN en ninguno de los dos. Base de toda correlación."""
    mask = a.notna() & b.notna()
    return a[mask], b[mask]


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de Pearson con numpy (sin scipy).

    Guardamos el caso de varianza nula (serie constante): np.corrcoef devolvería
    NaN y aquí significa "ninguna relación" → 0.0. Calcularlo a mano mantiene la
    misma filosofía $0/sin-dependencias-pesadas que el resto del quant engine.
    """
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if not np.isnan(c) else 0.0


def spearman_ic(signal: pd.Series, fwd: pd.Series) -> float:
    """Correlación de RANGOS señal↔retorno futuro. Robusta; mide monotonicidad.

    Spearman ≡ Pearson sobre los rangos. `Series.rank()` (promedio en empates)
    es puro pandas, así evitamos la ruta `.corr(method="spearman")` que importa
    scipy.
    """
    a, b = _aligned(signal, fwd)
    if len(a) < 3:
        return 0.0
    return _pearson_np(a.rank().to_numpy(), b.rank().to_numpy())


def pearson_ic(signal: pd.Series, fwd: pd.Series) -> float:
    """Correlación LINEAL señal↔retorno futuro."""
    a, b = _aligned(signal, fwd)
    if len(a) < 3:
        return 0.0
    return _pearson_np(a.to_numpy(), b.to_numpy())


def corr_tstat(ic: float, n_eff: int) -> float:
    """t-stat de una correlación con n_eff observaciones independientes.

    t = r·sqrt((n-2)/(1-r²)); bajo H0 (corr=0) ~ t de Student con n-2 gl. Con
    muestras grandes |t|≳2 ≈ p<0.05. Devuelve 0 si n_eff≤2 (sin grados de
    libertad para afirmar nada).
    """
    if n_eff <= 2:
        return 0.0
    denom = 1.0 - ic * ic
    if denom <= 0:
        return math.copysign(math.inf, ic)
    return ic * math.sqrt((n_eff - 2) / denom)


def directional_hit_rate(
    signal: pd.Series, fwd: pd.Series, threshold: float
) -> tuple[float, int]:
    """Fracción de aciertos de dirección entre velas con |señal|≥umbral.

    Solo cuenta las velas donde el bot ABRIRÍA (señal "fuerte"). Acierto =
    sign(retorno futuro) == sign(señal). Devuelve (tasa, nº de velas evaluadas).
    """
    a, b = _aligned(signal, fwd)
    strong = a.abs() >= threshold
    n = int(strong.sum())
    if n == 0:
        return 0.0, 0
    hits = np.sign(b[strong].to_numpy()) == np.sign(a[strong].to_numpy())
    return float(hits.mean()), n


def base_up_rate(fwd: pd.Series) -> float:
    """Deriva del mercado: P(retorno futuro > 0). Benchmark del acierto direccional."""
    f = fwd.dropna()
    return float((f > 0).mean()) if len(f) else 0.0


def quantile_forward_means(
    signal: pd.Series, fwd: pd.Series, n_quantiles: int
) -> list[float]:
    """Retorno futuro MEDIO por cuantil de señal (de la más bajista a la más alta).

    Con edge, la lista crece monótonamente. Plana o desordenada ⇒ la señal no
    discrimina. `duplicates='drop'` puede devolver menos cubos si hay muchos
    empates en la señal (no es un error, es honestidad sobre los datos).
    """
    a, b = _aligned(signal, fwd)
    if len(a) < n_quantiles:
        return []
    try:
        buckets = pd.qcut(a, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return []
    return [float(x) for x in b.groupby(buckets).mean().to_numpy()]


@dataclass
class HorizonStats:
    """Diagnóstico de edge de la señal a un horizonte forward concreto."""

    horizon: int                      # velas hacia adelante
    n: int                            # observaciones solapadas usadas
    n_eff: int                        # muestra efectiva ≈ n / horizon
    spearman_ic: float
    pearson_ic: float
    t_eff: float                      # t-stat de la Spearman IC con n_eff
    hit_rate: float                   # acierto direccional en |señal|≥umbral
    hit_n: int                        # nº de velas que superan el umbral
    base_up_rate: float               # P(retorno>0) incondicional (deriva)
    quantile_mean_fwd: list[float]    # retorno medio por cuantil de señal
    quantile_spread: float            # cubo más alto − cubo más bajo
    ic_regime_up: float               # Spearman IC con close>EMA lenta
    ic_regime_down: float             # Spearman IC con close<=EMA lenta


def horizon_stats(
    signal: pd.Series,
    fwd: pd.Series,
    regime_up: pd.Series,
    *,
    horizon: int,
    threshold: float,
    n_quantiles: int,
) -> HorizonStats:
    """Compone todas las métricas de edge para un horizonte dado. Función pura."""
    a, _ = _aligned(signal, fwd)
    n = len(a)
    n_eff = max(n // horizon, 1)
    sp = spearman_ic(signal, fwd)
    pe = pearson_ic(signal, fwd)
    hr, hit_n = directional_hit_rate(signal, fwd, threshold)
    qmeans = quantile_forward_means(signal, fwd, n_quantiles)
    spread = (qmeans[-1] - qmeans[0]) if len(qmeans) >= 2 else 0.0

    up = regime_up.fillna(False).astype(bool)
    return HorizonStats(
        horizon=horizon, n=n, n_eff=n_eff,
        spearman_ic=sp, pearson_ic=pe, t_eff=corr_tstat(sp, n_eff),
        hit_rate=hr, hit_n=hit_n, base_up_rate=base_up_rate(fwd),
        quantile_mean_fwd=qmeans, quantile_spread=spread,
        ic_regime_up=spearman_ic(signal[up], fwd[up]),
        ic_regime_down=spearman_ic(signal[~up], fwd[~up]),
    )


def analyze_edge(
    df: pd.DataFrame, settings: Settings | None = None
) -> list[HorizonStats]:
    """Edge de la señal quant sobre un DataFrame de velas, por cada horizonte.

    No ejecuta trades ni toca riesgo/ejecución: solo correlaciona la señal con el
    retorno futuro. Los horizontes y los cuantiles vienen de config (`edge.*`); el
    umbral de "señal fuerte" se reutiliza de `confluence.quant_strong_threshold`
    (el mismo que decide aperturas en vivo) y el régimen de `quant.ema_slow_period`.
    """
    settings = settings or load_settings()
    df = df.reset_index(drop=True)
    close = df["close"]
    signal = compute_signal_series(df)
    ema_slow = ema(close, settings.quant.ema_slow_period)
    regime_up = close > ema_slow
    threshold = settings.confluence.quant_strong_threshold
    nq = settings.edge.n_quantiles

    return [
        horizon_stats(signal, forward_returns(close, h), regime_up,
                      horizon=h, threshold=threshold, n_quantiles=nq)
        for h in settings.edge.forward_horizons
    ]


# Nombre de la estrategia analizada, expuesto para el encabezado del reporte.
SIGNAL_NAME = STRATEGY_NAME
