"""Diagnósticos del protocolo anti-sobreajuste: funciones puras.

Son el "detector de mentiras" del proyecto (CLAUDE.md §protocolo, puntos 5-6):
un Sharpe de test positivo NO basta — puede ser una cola de suerte, una mitad
buena tapando una mala, o simple beta. Cada función implementa EXACTAMENTE la
definición pre-registrada en docs/research/2026-07-11_protocolo_sp500.md §6.

Convención: `returns` son retornos simples por unidad (trade o mes o día),
`periods_per_year` los anualiza (252 diario bursátil, 12 mensual). rf=0 en
todo Sharpe (ambos lados de cualquier comparación se miden idéntico).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def sharpe(returns, periods_per_year: float) -> float:
    """Sharpe anualizado: mean(r)/std(r) · √períodos_por_año. std ddof=0.

    0 por convención si no hay varianza o no hay datos (sin riesgo, sin info).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def bootstrap_sharpe_ci(returns, periods_per_year: float, *, iterations: int,
                        ci: float, seed: int = 0) -> tuple[float, float]:
    """CI del Sharpe por bootstrap iid: remuestrea las unidades CON reemplazo.

    Por qué funciona: si el Sharpe positivo depende de 3 trades afortunados,
    muchos remuestreos no los incluirán y la cola inferior del CI cruzará el
    cero. `seed` fijo → reproducible (mismo protocolo, mismo número).

    Returns:
        (lo, hi) — percentiles (1-ci)/2 y 1-(1-ci)/2 de la distribución.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(r)
    samples = rng.choice(r, size=(iterations, n), replace=True)
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(sds > 0, means / sds * math.sqrt(periods_per_year), 0.0)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(sharpes, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def paired_bootstrap_sharpe_diff_ci(strategy_returns, benchmark_returns,
                                    periods_per_year: float, *, iterations: int,
                                    ci: float, seed: int = 0) -> tuple[float, float]:
    """CI de la DIFERENCIA de Sharpe (estrategia − benchmark) por bootstrap
    PAREADO: cada remuestreo elige los MISMOS índices de día para ambas
    series (a diferencia de `bootstrap_sharpe_ci`, que remuestrea cada serie
    por separado), preservando la correlación día-a-día real entre la
    estrategia y el benchmark.

    Por qué hace falta (más allá del gate de 5 criterios): el criterio "supera
    a B&H" del protocolo solo compara dos números puntuales — un Sharpe test
    de +0.86 contra un B&H de +0.85 "pasa" aunque la diferencia (+0.01) sea
    indistinguible del ruido. Este bootstrap responde la pregunta correcta:
    ¿la VENTAJA observada sigue siendo positiva bajo remuestreo, o el CI de
    la diferencia cruza el cero? Exigido por CLAUDE.md §protocolo punto 5
    ("todo resultado que se vea bien pasa diagnóstico antes de llamarse vivo").
    """
    s = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    mask = ~np.isnan(s) & ~np.isnan(b)
    s, b = s[mask], b[mask]
    n = len(s)
    if n < 2:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))

    def _sharpes(samples: np.ndarray) -> np.ndarray:
        means = samples.mean(axis=1)
        sds = samples.std(axis=1, ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sds > 0, means / sds * math.sqrt(periods_per_year), 0.0)

    diff = _sharpes(s[idx]) - _sharpes(b[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(diff, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def max_drawdown(returns) -> float:
    """Peor caída pico-a-valle de la curva de capital, como fracción positiva.

    A diferencia del Sharpe (penaliza TODA la volatilidad, incluso subidas
    rápidas), esto mide solo el dolor real: cuánto llegó a caer el capital
    desde su máximo previo. curva = producto acumulado de (1+r); drawdown_t =
    curva_t / máximo_hasta_t − 1 (≤0); devolvemos |mínimo| (0 = nunca cayó).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    curve = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(curve)
    drawdown = curve / running_max - 1.0
    return float(-drawdown.min())


def calmar_ratio(returns, periods_per_year: float) -> float:
    """CAGR / |drawdown máximo| — retorno anual por unidad de PEOR caída.

    La métrica que Faber (2007) usa para justificar el timing de tendencia:
    un viaje más suave puede valer la pena aunque el retorno total sea
    similar o algo menor, porque el drawdown (no la volatilidad total) es lo
    que de verdad empuja a alguien a vender en pánico. 0 si no hubo caída
    (drawdown=0, sin riesgo que dividir) o sin datos suficientes.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    years = len(r) / periods_per_year
    if years <= 0:
        return 0.0
    total_return = float(np.prod(1.0 + r) - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    dd = max_drawdown(r)
    if dd == 0.0:
        return 0.0
    return float(cagr / dd)


def win_rate(pnls) -> float:
    """Fracción de unidades (trades) con retorno neto > 0 — DESCRIPTIVO.

    NO es un criterio de éxito del protocolo: un win-rate alto no equivale a
    tener ventaja (una estrategia puede acertar el 90% de las veces y perder
    dinero si el 10% de pérdidas es mucho más grande que las ganancias
    típicas — el perfil de vender opciones). Se reporta solo para responder
    la pregunta "¿acierta la mayoría de las veces?" sin redefinir el gate.
    """
    p = np.asarray(pnls, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return 0.0
    return float((p > 0).sum() / len(p))


def concentration_top_decile(pnls) -> float:
    """Fracción de la ganancia neta total aportada por el top 10% de unidades.

    Definición pre-registrada: ordenar por PnL descendente, sumar el top
    ⌈10%⌉, dividir por la ganancia neta total. Solo tiene sentido con ganancia
    total > 0 (si no, la estrategia ya falló el criterio 1); en ese caso
    devuelve NaN para que nadie lea un número sin sentido.

    >0.6 = la "estrategia" es unas pocas operaciones de suerte (el patrón que
    mató a TSMOM-JNJ y compañía en 2026-07-08).
    """
    p = np.asarray(pnls, dtype=float)
    p = p[~np.isnan(p)]
    total = p.sum()
    if len(p) == 0 or total <= 0:
        return float("nan")
    k = math.ceil(len(p) * 0.10)
    top = np.sort(p)[::-1][:k].sum()
    return float(top / total)


def halves_stability(returns, timestamps, periods_per_year: float) -> tuple[float, float]:
    """Sharpe de cada mitad del periodo, cortado por el PUNTO MEDIO DEL
    CALENDARIO (no por número de observaciones: una estrategia que solo opera
    al final tendría mitades vacías engañosas con el corte por conteo).

    Un edge real se sostiene en ambas mitades; ruido con suerte solo en una.
    """
    r = np.asarray(returns, dtype=float)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if len(r) == 0 or len(r) != len(ts):
        return (0.0, 0.0)
    mid = ts.min() + (ts.max() - ts.min()) / 2
    first = r[ts <= mid]
    second = r[ts > mid]
    return (sharpe(first, periods_per_year), sharpe(second, periods_per_year))


@dataclass
class GateResult:
    """Veredicto de los 5 criterios pre-registrados sobre un resultado de TEST."""

    sharpe_test: float
    ci_lo: float
    ci_hi: float
    concentration: float
    sharpe_h1: float
    sharpe_h2: float
    sharpe_buyhold: float
    passes_sharpe: bool
    passes_bootstrap: bool
    passes_concentration: bool
    passes_halves: bool
    passes_vs_buyhold: bool

    @property
    def passes_all(self) -> bool:
        return (self.passes_sharpe and self.passes_bootstrap
                and self.passes_concentration and self.passes_halves
                and self.passes_vs_buyhold)


def evaluate_gate(returns, timestamps, units, periods_per_year: float, *,
                  sharpe_min: float, iterations: int, ci: float,
                  concentration_max: float, sharpe_buyhold: float,
                  units_per_year: float | None = None,
                  seed: int = 0) -> GateResult:
    """Aplica los 5 criterios del protocolo a un resultado de test.

    `returns`/`timestamps`: la serie periódica de la estrategia (para el
    Sharpe del criterio 1 y las mitades del criterio 4). `units`: las UNIDADES
    pre-registradas del bootstrap y la concentración — meses para carteras
    mensuales (units == returns), trades para estrategias por-trade.
    `units_per_year` anualiza el bootstrap de esas unidades (si None, usa
    periods_per_year — el caso mensual).
    """
    sh = sharpe(returns, periods_per_year)
    uppy = units_per_year if units_per_year is not None else periods_per_year
    lo, hi = bootstrap_sharpe_ci(units, uppy,
                                 iterations=iterations, ci=ci, seed=seed)
    conc = concentration_top_decile(units)
    h1, h2 = halves_stability(returns, timestamps, periods_per_year)
    return GateResult(
        sharpe_test=sh, ci_lo=lo, ci_hi=hi, concentration=conc,
        sharpe_h1=h1, sharpe_h2=h2, sharpe_buyhold=sharpe_buyhold,
        passes_sharpe=sh > sharpe_min,
        passes_bootstrap=lo > 0.0,
        # NaN (ganancia total ≤ 0) nunca pasa: nan < x es False.
        passes_concentration=bool(conc < concentration_max),
        passes_halves=(h1 > 0.0 and h2 > 0.0),
        passes_vs_buyhold=sh > sharpe_buyhold,
    )
