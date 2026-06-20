"""Matriz de research del Slow Path — embudo de 2 etapas (lógica pura, testeable).

Tras 6 familias quant descartadas con rigor (ver memoria del proyecto), esta
matriz re-evalúa hipótesis NUEVAS bajo un embudo anti-overfit:

    ETAPA 1 (gate de significancia, barato)
        - señales predictivas (A/C/D) → IC de Spearman + t-stat con n_eff (edge.py).
        - yield/spread (E/B)          → t-stat de la MEDIA del retorno neto.
      Descarta si no supera la Regla de Oro (|t|≥golden_min_tstat).
    ETAPA 2 (P&L, caro) → equity curve con costos → Sharpe · MaxDD · PF.

Esta sesión implementa SOLO la Familia E (carry). B/C/D quedan como stubs
explícitos (una sesión = un módulo).

— Familia E · Cash-and-Carry de funding (delta-neutral) —
Estructura: long spot + short perp del MISMO notional → delta direccional ≈ 0.
El yield es el funding que el short COBRA cuando funding>0 (los longs pagan a los
shorts) y PAGA cuando funding<0. La serie de `funding_rate` sumada en el tiempo,
con su signo, ES el yield bruto sobre el notional.

Física que dicta el modelo de costos (decisión de diseño, no atajo):
  - El carry spot-perp POR CANTIDAD (1 unidad larga, 1 corta) es delta-neutral
    automáticamente y para siempre → NO hay drift de delta que rebalancear (a
    diferencia del cross-sectional o las opciones). Y el perp NO expira → NO hay
    roll obligatorio (a diferencia del carry de futuros con fecha). Por eso el
    "costo de rebalanceo" es de segundo orden aquí.
  - El costo dominante es: (a) ENTRADA/SALIDA = 4 lados taker (comprar spot +
    abrir short + cerrar ambos), y (b) FRICCIÓN DE CAPITAL: para no liquidarte en
    un pump inmovilizas ~2× el notional → el yield sobre capital es la MITAD del
    yield sobre notional. (a) es explícito; (b) va en `carry_capital_multiplier`.
  - `carry_maintenance_bps_per_period` (default 0) deja añadir un haircut de
    mantenimiento (borrow del spot, gestión de margen) si se quiere ser aún más
    conservador, pero la física no lo exige fuertemente.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import QuantMatrixConfig
from backtest.metrics import max_drawdown, profit_factor

# Binance liquida funding cada 8h → 3 periodos/día.
PERIODS_PER_DAY = 3
PERIODS_PER_YEAR = PERIODS_PER_DAY * 365  # 1095


@dataclass
class CarryStats:
    """Resultado del simulador de carry para un activo (una fila del DataFrame)."""
    symbol: str
    n_periods: int
    gross_yield_ann_pct: float   # bruto sobre NOTIONAL, anualizado (%)
    net_yield_ann_pct: float     # neto de costos sobre CAPITAL desplegado (%)
    sharpe: float                # anualizado
    max_drawdown: float          # fracción positiva (0.05 = 5%)
    profit_factor: float
    t_stat: float                # t de la media del retorno neto por periodo
    pct_negative_periods: float  # % de periodos con funding neto < 0 (pagas)
    worst_period_pct: float      # peor funding_rate del histórico (%)
    folds_same_sign: int
    n_folds: int
    passes_golden: bool


def carry_period_returns(
    funding_rate: pd.Series, cfg: QuantMatrixConfig
) -> np.ndarray:
    """Retorno por periodo de 8h sobre el CAPITAL desplegado, neto de mantenimiento.

    short perp + funding>0 ⇒ cobras +funding_rate (sobre el notional). Restamos el
    haircut de mantenimiento y dividimos por `carry_capital_multiplier` para pasar
    de "retorno sobre notional" a "retorno sobre el capital realmente inmovilizado"
    (spot + margen del perp).
    """
    maint = cfg.carry_maintenance_bps_per_period / 1e4   # bps → fracción sobre notional
    gross_on_notional = funding_rate.to_numpy(dtype=float)
    net_on_notional = gross_on_notional - maint
    return net_on_notional / cfg.carry_capital_multiplier


def simulate_carry(
    funding_rate: pd.Series, cfg: QuantMatrixConfig, *, symbol: str = "", n_folds: int = 4
) -> CarryStats:
    """Simula el cash-and-carry delta-neutral sobre la serie de funding (8h).

    Aplica costos one-time de entrada/salida (4 lados taker, repartidos en el
    primer y último periodo para que entren en la equity curve), calcula la Regla
    de Oro (|t| y PF) y la consistencia de signo del yield en `n_folds` tramos.
    """
    r = carry_period_returns(funding_rate, cfg)   # perfil de funding por periodo (recurrente)
    n = len(r)
    if n == 0:
        raise ValueError("serie de funding vacía")

    comm = cfg.taker_commission_pct / 100.0                 # por lado, fracción
    entry_exit = (4.0 * comm) / cfg.carry_capital_multiplier  # 4 lados, sobre capital (ONE-TIME)
    years = n / PERIODS_PER_YEAR

    # Métricas económicas sobre el PERFIL de funding (r puro). El costo one-time de
    # entrada/salida NO se inyecta en un periodo arbitrario: distorsionaría el conteo
    # de periodos negativos y el MaxDD. Se amortiza aparte en el yield neto.
    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1)) if n > 1 else 0.0
    t_stat = mean_r / (std_r / math.sqrt(n)) if std_r > 0 else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(PERIODS_PER_YEAR) if std_r > 0 else 0.0

    # Yield neto = funding anualizado − costo de entrada/salida amortizado por año.
    net_yield_ann = mean_r * PERIODS_PER_YEAR * 100.0 - (entry_exit / years) * 100.0
    gross_yield_ann = float(funding_rate.mean()) * PERIODS_PER_YEAR * 100.0
    pf = profit_factor(r)
    equity = 1.0 + np.cumsum(r)   # equity del perfil de funding (MaxDD = rachas negativas)
    mdd = max_drawdown(equity)
    pct_neg = float((r < 0).mean()) * 100.0
    worst = float(funding_rate.min()) * 100.0

    # Walk-forward: el yield debe tener el MISMO signo en los n_folds tramos
    # (un edge real no vive de un solo régimen macro).
    folds = np.array_split(r, n_folds)
    fold_signs = [1 if f.mean() > 0 else -1 for f in folds]
    folds_same_sign = max(fold_signs.count(1), fold_signs.count(-1))

    passes = (
        abs(t_stat) >= cfg.golden_min_tstat
        and pf > cfg.golden_min_profit_factor
        and mean_r > 0
        and folds_same_sign == n_folds
    )

    return CarryStats(
        symbol=symbol, n_periods=n,
        gross_yield_ann_pct=gross_yield_ann, net_yield_ann_pct=net_yield_ann,
        sharpe=sharpe, max_drawdown=mdd, profit_factor=pf, t_stat=t_stat,
        pct_negative_periods=pct_neg, worst_period_pct=worst,
        folds_same_sign=folds_same_sign, n_folds=n_folds, passes_golden=passes,
    )


# --------------------- Familias pendientes (stubs explícitos) ---------------------
# Firmas listas para las próximas sesiones modulares. Lanzan en vez de devolver
# vacío para que nadie las cuente como "evaluadas y sin candidata" por error.

def run_family_pairs(log_prices: "pd.DataFrame", cfg: QuantMatrixConfig) -> list:
    """Familia B — cointegración de pares (rolling OLS + IC gate + P&L).

    log_prices: DataFrame con columnas = símbolos y valores = log(close).
    Evalúa todos los C(n, 2) pares y devuelve una lista de PairsStats.
    """
    from backtest.pairs import run_pairs_all
    import pandas as pd  # noqa: F401 — type hint en el cuerpo
    return run_pairs_all(log_prices, cfg)


def run_family_volume(cfg: QuantMatrixConfig):
    """Familia C — microestructura/volumen (desviación de VWAP, VPA)."""
    raise NotImplementedError(
        "Familia C (volumen/VWAP) — pendiente de su sesión modular")


def run_family_squeeze(cfg: QuantMatrixConfig):
    """Familia D — squeeze de volatilidad (GARCH/ATR multi-ventana → ruptura)."""
    raise NotImplementedError(
        "Familia D (squeeze de volatilidad) — pendiente de su sesión modular")
