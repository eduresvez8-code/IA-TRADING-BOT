"""Familia D — Squeeze de volatilidad → ruptura (TTM/Bollinger-Keltner, 1h).

Hipótesis: la baja volatilidad (squeeze) precede a una expansión DIRECCIONAL, y
cuando el precio rompe la banda tras un squeeze la ruptura CONTINÚA (momentum).
Es direccionalmente OPUESTA a la reversión de la Familia C: aquí el IC esperado
es > 0 (la ruptura continúa), no < 0.

Construcción de la señal (estándar TTM Squeeze):
    mid_t   = SMA(close, N)                       (línea media compartida)
    σ_t     = std(close, N)                        (dispersión del cierre)
    band_t  = bb_std · σ_t                          (semi-ancho Bollinger)
    ATR_t   = ATR(N)                                (rango verdadero medio)
    KC_t    = keltner_k · ATR_t                      (semi-ancho Keltner)
    squeeze_on_t = band_t < KC_t                     (Bollinger DENTRO de Keltner)
El squeeze es una RAZÓN de volatilidades (σ del cierre vs ATR): cuando la banda
cabe dentro del canal de Keltner el mercado está "enrollado" (comprimido).

    dist_t    = close_t − mid_t                       (desviación firmada de la media)
    breakout_t = dist_t / band_t                       (≈ ±1 al tocar la banda)
    fire_t    = squeeze_on_{t-1} ∧ |dist_t| > thr·band_t
La ruptura DISPARA cuando veníamos comprimidos (t−1) y el cierre supera el umbral
de banda. La dirección de continuación = signo(dist_t): rompe arriba → LONG.

Embudo de 2 etapas (idéntico a B y C):
    ETAPA 1 — IC de Spearman(breakout_t, ret_{t+1 → t+1+h}) CONDICIONAL a las
              barras de ruptura, con corrección n_eff. h = squeeze_forward_horizon.
              El forward arranca en t+1 (bounce-robust: excluye el bid-ask del
              propio breakout). IC > 0 ⇒ la ruptura continúa (hipótesis). El IC
              se mide solo sobre las fires porque ése es el universo que se opera.
    ETAPA 2 — máquina de estados con costos REALES (taker + slippage %+k·ATR).
              Entrada diferida una barra (shift-1: lookahead-free). Salida
              TIME-BASED a las h barras: la continuación, si existe, vive en la
              ventana de expansión; un holding fijo evita inventar un z_exit y
              ata el P&L al mismo horizonte que el IC.

Descomposición GROSS vs NETO: el simulador acumula el M2M bruto y el neto en
series separadas → la lección de la Familia C (separar "no hay señal" de "señal
enterrada por costos") sale de una sola corrida.

Una sesión = un módulo. Reemplaza el stub run_family_squeeze de quant_matrix.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.metrics import max_drawdown, profit_factor
from src.core.config import QuantMatrixConfig

PERIODS_PER_YEAR = 365 * 24  # barras de 1h por año


@dataclass
class SqueezeStats:
    """Resultado del simulador de squeeze→ruptura para un activo."""
    symbol: str
    n_periods: int               # barras válidas (post-warmup de BB/ATR)
    pct_squeeze: float           # % de barras en estado de squeeze (diagnóstico)
    n_fires: int                 # nº de rupturas-tras-squeeze detectadas
    ic_spearman: float           # IC(breakout_t, ret_{+h}) sobre fires, esperado > 0
    ic_tstat: float              # t-stat corregido por autocorr (n_eff)
    n_trades: int
    gross_return_ann_pct: float  # retorno BRUTO anualizado (costos=0) — descomposición
    net_return_ann_pct: float    # retorno NETO anualizado tras costos
    sharpe: float                # anualizado, base-tiempo (incluye barras flat)
    max_drawdown: float          # MaxDD de la equity NETA (fracción positiva)
    profit_factor: float         # PF de los retornos netos por trade
    pct_winning_trades: float
    avg_holding_bars: float
    folds_same_sign: int
    n_folds: int
    passes_golden: bool


# ---------------------------------------------------------------------------
# Spearman sin scipy (scipy no está en el proyecto)
# ---------------------------------------------------------------------------

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=float)
        out[np.argsort(a, kind="stable")] = np.arange(1, n + 1, dtype=float)
        return out

    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


# ---------------------------------------------------------------------------
# Construcción de la señal
# ---------------------------------------------------------------------------

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ATR(period) absoluto (rango verdadero medio). Usado por Keltner y diagnóstico."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def squeeze_signal(
    high: pd.Series, low: pd.Series, close: pd.Series, cfg: QuantMatrixConfig
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Devuelve (squeeze_on, breakout, fire) a partir de OHLC.

    breakout = dist/band (firmado, ≈ ±1 al tocar la banda Bollinger).
    fire     = veníamos en squeeze (t−1) y el cierre rompe el umbral de banda en t.

    El ATR de Keltner usa squeeze_bb_period (NO atr_period): Keltner y Bollinger
    deben compartir ventana para que la comparación banda-vs-canal sea apples-to-
    apples (así se define el TTM Squeeze). atr_period queda reservado al ATR del
    modelo de SLIPPAGE (otra preocupación distinta).
    """
    n = cfg.squeeze_bb_period
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    band = cfg.squeeze_bb_std * sd                       # semi-ancho Bollinger
    kc = cfg.squeeze_keltner_atr_mult * _atr(high, low, close, n)  # semi-ancho Keltner

    squeeze_on = band < kc                                # Bollinger dentro de Keltner
    dist = close - mid
    breakout = dist / band.replace(0.0, np.nan)
    broke = dist.abs() > cfg.squeeze_breakout_threshold * band
    fire = squeeze_on.shift(1).fillna(False) & broke
    return squeeze_on, breakout, fire


# ---------------------------------------------------------------------------
# Etapa 1 — IC gate (condicional a las fires, bounce-robust desde t+1)
# ---------------------------------------------------------------------------

def squeeze_ic(
    breakout: pd.Series, fire: pd.Series, close: pd.Series, horizon: int
) -> tuple[float, float]:
    """IC Spearman(breakout_t, ret_{t+1 → t+1+h}) SOLO sobre barras de ruptura.

    Condicionar a las fires es lo correcto: el universo operable son esas barras,
    no todo el histórico (medir el IC con miles de ceros de "no-ruptura" diluiría
    la señal en empates). El forward arranca en t+1 para excluir el bid-ask del
    propio breakout (igual que la Familia C). Corrección n_eff con la autocorr del
    breakout SOBRE las fires (que, al estar espaciadas, suele dar ρ₁ bajo → n_eff≈n).
    """
    b = breakout.to_numpy(dtype=float)
    f = fire.to_numpy(dtype=bool)
    c = close.to_numpy(dtype=float)
    n = len(c)
    if n <= horizon + 2:
        return 0.0, 0.0

    fwd = np.full(n, np.nan)
    idx = np.arange(1, n - horizon)
    fwd[idx] = c[idx + horizon] / c[idx] - 1.0          # ret_{t+1 → t+1+h}

    valid = f & ~np.isnan(b) & ~np.isnan(fwd)
    b_v, fwd_v = b[valid], fwd[valid]
    nv = len(b_v)
    if nv < 30:
        return 0.0, 0.0

    ic = _spearman_r(b_v, fwd_v)
    rho1 = float(pd.Series(b_v).autocorr(lag=1))
    if math.isnan(rho1) or abs(rho1) >= 1.0:
        n_eff = float(nv)
    else:
        n_eff = max(nv * (1.0 - rho1) / (1.0 + rho1), 2.0)
    denom = math.sqrt(max(1.0 - ic ** 2, 1e-9))
    t = ic * math.sqrt(n_eff) / denom
    return float(ic), float(t)


# ---------------------------------------------------------------------------
# Etapa 2 — máquina de estados (bar a bar, holding time-based)
# ---------------------------------------------------------------------------

def _simulate_bars(
    breakout: pd.Series,
    fire: pd.Series,
    close: pd.Series,
    atr_px: pd.Series,
    cfg: QuantMatrixConfig,
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    """FLAT / LONG / SHORT con entrada diferida una barra y salida time-based.

    Señal de acción = (dirección de la ruptura en la fire) shift(1): se observa en
    t, se ENTRA en close_{t+1} (lookahead-free). Continuación ⇒ se entra en la
    dirección de la ruptura (rompe arriba → LONG). Se mantiene exactamente
    `squeeze_forward_horizon` barras y se cierra (holding fijo = mismo horizonte
    que el IC). No se piramidea: nuevas fires se ignoran si ya hay posición.

    Returns:
        gross_bar: M2M bruto por barra (sin costos) → equity/retorno GROSS.
        net_bar:   M2M neto por barra (costo restado al cerrar) → equity NETA.
        trade_net: retorno neto por trade cerrado (para PF y % ganadores).
        holdings:  nº de barras que duró cada trade.
    """
    # dirección firmada de la ruptura en la barra de fire (signo del breakout), 0 si no fire
    dir_fire = np.where(fire.to_numpy(dtype=bool), np.sign(breakout.to_numpy(dtype=float)), 0.0)
    sig = pd.Series(dir_fire).shift(1).to_numpy(dtype=float)   # actuar una barra tarde
    c = close.to_numpy(dtype=float)
    slip = (cfg.slippage_pct / 100.0) + cfg.slippage_atr_mult * atr_px.to_numpy(dtype=float)
    taker = cfg.taker_commission_pct / 100.0
    horizon = cfg.squeeze_forward_horizon
    n = len(c)

    gross_bar = np.zeros(n, dtype=float)
    net_bar = np.zeros(n, dtype=float)
    trade_net: list[float] = []
    holdings: list[int] = []

    pos = 0            # 0=flat, +1=long, −1=short
    entry_i = -1
    exit_at = -1
    trade_gross = 0.0

    for i in range(1, n):
        # 1. Mark-to-market bruto (si hay posición desde la barra previa)
        if pos != 0:
            r = pos * (c[i] / c[i - 1] - 1.0)
            gross_bar[i] = r
            net_bar[i] = r
            trade_gross += r

        # 2. Salida time-based (tras el M2M): se mantuvo `horizon` barras
        if pos != 0 and i >= exit_at:
            cost = 2.0 * taker + _slip_at(slip, entry_i) + _slip_at(slip, i)
            trade_net.append(trade_gross - cost)
            holdings.append(i - entry_i)
            net_bar[i] -= cost                  # la equity NETA incluye el costo una vez
            pos = 0

        # 3. Entrada (solo si FLAT y hay una ruptura direccional)
        if pos == 0:
            s = sig[i]
            if not math.isnan(s) and s != 0.0:
                pos = 1 if s > 0 else -1         # continuación: en la dirección de la ruptura
                entry_i = i
                exit_at = i + horizon
                trade_gross = 0.0

    # Cierre forzado al final de la serie
    if pos != 0:
        last = n - 1
        cost = 2.0 * taker + _slip_at(slip, entry_i) + _slip_at(slip, last)
        trade_net.append(trade_gross - cost)
        holdings.append(last - entry_i)
        net_bar[last] -= cost

    return gross_bar, net_bar, trade_net, holdings


def _slip_at(slip: np.ndarray, i: int) -> float:
    """Slippage en la barra i; si es NaN (warmup del ATR) usa 0 (conservador-neutral)."""
    v = slip[i]
    return 0.0 if math.isnan(v) else float(v)


# ---------------------------------------------------------------------------
# Simulador principal
# ---------------------------------------------------------------------------

def simulate_squeeze(
    df: pd.DataFrame,
    cfg: QuantMatrixConfig,
    *,
    symbol: str = "",
    n_folds: int = 4,
) -> SqueezeStats:
    """Simula el squeeze→ruptura sobre un DataFrame OHLCV de 1h.

    df debe traer columnas: high, low, close (open_time opcional, no se usa: el
    squeeze no ancla a la hora del día como sí lo hace el VWAP).
    """
    squeeze_on, breakout, fire = squeeze_signal(df["high"], df["low"], df["close"], cfg)
    atr_px = _atr(df["high"], df["low"], df["close"], cfg.atr_period) / df["close"]

    # Etapa 1
    ic, ic_t = squeeze_ic(breakout, fire, df["close"], cfg.squeeze_forward_horizon)

    # Etapa 2
    gross_bar, net_bar, trade_net, holdings = _simulate_bars(
        breakout, fire, df["close"], atr_px, cfg
    )

    valid_mask = ~np.isnan(breakout.to_numpy(dtype=float))
    rn = net_bar[valid_mask]
    rg = gross_bar[valid_mask]
    n = len(rn)
    n_fires = int(fire.sum())
    pct_sq = float(squeeze_on.mean()) * 100.0

    if n == 0:
        return SqueezeStats(
            symbol=symbol, n_periods=0, pct_squeeze=pct_sq, n_fires=n_fires,
            ic_spearman=ic, ic_tstat=ic_t, n_trades=0,
            gross_return_ann_pct=0.0, net_return_ann_pct=0.0, sharpe=0.0,
            max_drawdown=0.0, profit_factor=0.0, pct_winning_trades=0.0,
            avg_holding_bars=0.0, folds_same_sign=0, n_folds=n_folds,
            passes_golden=False,
        )

    years = n / PERIODS_PER_YEAR
    mean_r = float(rn.mean())
    std_r = float(rn.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(PERIODS_PER_YEAR) if std_r > 0 else 0.0
    equity = 1.0 + np.cumsum(rn)
    mdd = max_drawdown(equity)

    trade_arr = np.array(trade_net, dtype=float)
    n_trades = len(trade_arr)
    pf = profit_factor(trade_arr) if n_trades > 0 else 0.0
    pct_win = float((trade_arr > 0).mean()) * 100.0 if n_trades > 0 else 0.0
    avg_hold = float(np.mean(holdings)) if holdings else 0.0

    net_ann = (float(rn.sum()) / years) * 100.0 if years > 0 else 0.0
    gross_ann = (float(rg.sum()) / years) * 100.0 if years > 0 else 0.0

    folds = np.array_split(rn, n_folds)
    fold_signs = [1 if f.mean() > 0 else -1 for f in folds]
    folds_same_sign = max(fold_signs.count(1), fold_signs.count(-1))

    passes = (
        ic > 0                                       # continuación (opuesto a C)
        and abs(ic_t) >= cfg.golden_min_tstat
        and pf > cfg.golden_min_profit_factor
        and net_ann > 0
        and folds_same_sign == n_folds
    )

    return SqueezeStats(
        symbol=symbol, n_periods=n, pct_squeeze=pct_sq, n_fires=n_fires,
        ic_spearman=ic, ic_tstat=ic_t, n_trades=n_trades,
        gross_return_ann_pct=gross_ann, net_return_ann_pct=net_ann, sharpe=sharpe,
        max_drawdown=mdd, profit_factor=pf, pct_winning_trades=pct_win,
        avg_holding_bars=avg_hold, folds_same_sign=folds_same_sign,
        n_folds=n_folds, passes_golden=passes,
    )
