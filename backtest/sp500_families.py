"""Las 5 familias pre-registradas del protocolo S&P 500 — funciones puras.

Ver docs/research/2026-07-11_protocolo_sp500.md §4 (grids congelados) y §6
(mecánica de ejecución). Este módulo NO decide nada del protocolo: implementa
exactamente lo declarado. Los grids llegan por config (Cero Hardcoding).

Mecánica común (sin look-ahead, regla de todo el proyecto):
    - La señal se calcula con datos hasta el CIERRE del periodo t.
    - La ejecución ocurre a la APERTURA del periodo t+1.
    - Mensual: retorno del mes m = open(1er día hábil de m+1) / open(1er día
      hábil de m) − 1. La posición del mes m se decidió al cierre de m−1.
    - Diario: retorno del día t = open(t+1)/open(t) − 1; la posición del día t
      se decidió al cierre de t−1.
    - Costos: `per_side` (fracción) por unidad de peso operada: Σ|Δw|·per_side.
    - El cash fuera de mercado devenga la T-bill (^IRX) — protocolo §3.

Delistings (declarado): el retorno de tenencia de un ticker usa la primera
apertura del mes y la primera apertura del mes siguiente; si el ticker
desaparece a mitad de mes, se usa su ÚLTIMO precio disponible como salida
(venta al último cierre). Si no hay ningún precio en el mes, el retorno es 0
(posición disuelta al precio de entrada). El sesgo residual de los deslistados
SIN datos se mide aparte con coverage_report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.quant.indicators import rsi as _rsi


# ---------------------------------------------------------------------------
# Utilidades de calendario mensual
# ---------------------------------------------------------------------------

def _series(df: pd.DataFrame, col: str) -> pd.Series:
    """Columna indexada por open_time SIN timezone: los Period mensuales no
    llevan tz y agrupar un índice aware dispara warnings y conversiones
    implícitas — mejor una sola conversión explícita aquí."""
    s = df.set_index("open_time")[col].sort_index()
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s


def first_open_by_month(df: pd.DataFrame) -> pd.Series:
    """Apertura del PRIMER día hábil de cada mes (index: Period 'M')."""
    s = _series(df, "open")
    return s.groupby(s.index.to_period("M")).first()


def last_close_by_month(df: pd.DataFrame) -> pd.Series:
    """Cierre del ÚLTIMO día hábil de cada mes (index: Period 'M')."""
    s = _series(df, "close")
    return s.groupby(s.index.to_period("M")).last()


def monthly_hold_returns(df: pd.DataFrame) -> pd.Series:
    """Retorno de TENER el activo durante el mes m: open(m+1)/open(m) − 1.

    El último mes de la muestra no tiene m+1 → se descarta (no se inventa).
    """
    fo = first_open_by_month(df)
    return (fo.shift(-1) / fo - 1.0).dropna()


def monthly_cash_returns(tbill_daily: pd.Series, index_like: pd.PeriodIndex) -> pd.Series:
    """Retorno mensual del cash: suma de los retornos diarios T-bill del mes.

    `tbill_daily` viene de src.data.sp500.tbill_daily_return (indexado por
    fecha). Se reindexa al calendario de meses de la estrategia; un mes sin
    dato de T-bill devenga 0 (conservador).
    """
    tb = _naive(tbill_daily)
    m = tb.groupby(tb.index.to_period("M")).sum()
    return m.reindex(index_like).fillna(0.0)


def shift_to_holding(weights: pd.DataFrame | pd.Series):
    """Pesos indexados por mes de DECISIÓN → aplicados al mes SIGUIENTE.

    Es el anti-look-ahead estructural: la fila decidida con el cierre de m
    solo toca el retorno de m+1. La primera fila del holding queda NaN → se
    descarta (no había decisión previa).
    """
    return weights.shift(1)


def monthly_strategy_returns(weights: pd.DataFrame, asset_returns: pd.DataFrame,
                             cash_monthly: pd.Series, per_side: float) -> pd.Series:
    """Retornos mensuales netos de una cartera de pesos objetivo.

    Args:
        weights:       pesos OBJETIVO por mes de decisión (columnas = activos,
                       cada fila suma ≤ 1; el resto es cash). Se desplazan
                       internamente al mes de tenencia.
        asset_returns: retornos de tenencia por mes (mismas columnas).
        cash_monthly:  retorno del cash por mes.
        per_side:      costo por lado como FRACCIÓN (0.0002 = 2 pb).

    Returns:
        Serie de retornos mensuales netos (meses con decisión previa válida).
    """
    w = shift_to_holding(weights)
    w, r = w.align(asset_returns, join="inner", axis=0)
    w = w.fillna(0.0)
    r = r.fillna(0.0)
    gross = (w * r).sum(axis=1) + (1.0 - w.sum(axis=1)) * cash_monthly.reindex(w.index).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1)
    # Primer mes: pasar de 0 a los pesos iniciales también se paga.
    if len(w) > 0:
        turnover.iloc[0] = w.iloc[0].abs().sum()
    net = gross - per_side * turnover
    return net.iloc[1:] if len(net) > 1 else net  # la fila 0 no tenía decisión previa


# ---------------------------------------------------------------------------
# Utilidades diarias
# ---------------------------------------------------------------------------

def daily_hold_returns(df: pd.DataFrame) -> pd.Series:
    """Retorno de tener el activo el día t: open(t+1)/open(t) − 1 (índice: fecha t)."""
    s = _series(df, "open")
    return (s.shift(-1) / s - 1.0).dropna()


def _naive(s: pd.Series) -> pd.Series:
    """Índice datetime sin tz (las series diarias se alinean entre sí; una
    mezcla aware/naive alinearía a vacío EN SILENCIO y llenaría de ceros)."""
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s


def daily_strategy_returns(position: pd.Series, hold: pd.Series,
                           tbill_daily: pd.Series, per_side: float) -> pd.Series:
    """Retornos diarios netos de una señal 0/1 YA calculada al cierre de cada día.

    La posición del día t es `position.shift(1)` (señal del cierre de t−1,
    ejecutada en la apertura de t). Cash devenga T-bill. Costos en cada cambio.
    """
    position, hold, tbill_daily = _naive(position), _naive(hold), _naive(tbill_daily)
    pos = position.shift(1).reindex(hold.index).fillna(0.0)
    cash = tbill_daily.reindex(hold.index).fillna(0.0)
    gross = pos * hold + (1.0 - pos) * cash
    turnover = pos.diff().abs().fillna(pos.iloc[0] if len(pos) else 0.0)
    return gross - per_side * turnover


def trades_from_positions(position: pd.Series, hold: pd.Series,
                          per_side: float) -> list[float]:
    """Retorno por TRADE (segmentos de posición=1), neto de los dos lados.

    trade = (1−per_side)² · Π(1+hold_t) − 1 sobre el segmento — la misma
    convención de costo de dos lados del motor. Son las unidades del bootstrap
    y la concentración en familias diarias (protocolo §5-6).
    """
    position, hold = _naive(position), _naive(hold)
    pos = position.shift(1).reindex(hold.index).fillna(0.0).to_numpy()
    h = hold.to_numpy()
    trades: list[float] = []
    growth = None
    for p, r in zip(pos, h):
        if p > 0:
            growth = (growth if growth is not None else 1.0) * (1.0 + r)
        elif growth is not None:
            trades.append((1.0 - per_side) ** 2 * growth - 1.0)
            growth = None
    if growth is not None:
        trades.append((1.0 - per_side) ** 2 * growth - 1.0)
    return trades


# ---------------------------------------------------------------------------
# Familia 2 — TSMOM sobre el índice (definición congelada del Frente C)
# ---------------------------------------------------------------------------

def tsmom_index_weights(monthly_close: pd.Series, lookback_months: int) -> pd.DataFrame:
    """w=1 si el retorno de los últimos L meses EXCLUYENDO el último es >0.

    Momentum en el mes de decisión m: close(m−1)/close(m−1−L) − 1. Se excluye
    el mes m (el más reciente) por la reversión de corto plazo — definición
    idéntica al Frente C 2026-07-07 para comparabilidad.
    """
    mom = monthly_close.shift(1) / monthly_close.shift(1 + lookback_months) - 1.0
    w = (mom > 0).astype(float)
    w[mom.isna()] = np.nan  # sin historia suficiente: sin decisión (no "cash gratis")
    return w.to_frame("asset")


# ---------------------------------------------------------------------------
# Familia 3 — Timing por media móvil (Faber 2007 + golden cross)
# ---------------------------------------------------------------------------

def ma_timing_monthly_weights(daily_df: pd.DataFrame, sma_days: int) -> pd.DataFrame:
    """w=1 si el cierre de fin de mes está por encima de su SMA de N días."""
    s = _series(daily_df, "close")
    sma = s.rolling(sma_days).mean()
    close_m = s.groupby(s.index.to_period("M")).last()
    sma_m = sma.groupby(sma.index.to_period("M")).last()
    w = (close_m > sma_m).astype(float)
    w[sma_m.isna()] = np.nan
    return w.to_frame("asset")


def golden_cross_daily_position(daily_df: pd.DataFrame, fast_days: int,
                                slow_days: int) -> pd.Series:
    """Señal diaria 0/1: SMA(fast) > SMA(slow) al cierre del día (índice: fecha)."""
    s = _series(daily_df, "close")
    fast = s.rolling(fast_days).mean()
    slow = s.rolling(slow_days).mean()
    pos = (fast > slow).astype(float)
    pos[slow.isna()] = 0.0  # sin historia: plano
    return pos


# ---------------------------------------------------------------------------
# Familia 4 — Reversión RSI-2 (Connors 2008)
# ---------------------------------------------------------------------------

def rsi_reversion_daily_position(daily_df: pd.DataFrame, *, rsi_period: int,
                                 entry_below: float, exit_above: float,
                                 trend_sma_days: int) -> pd.Series:
    """Señal diaria 0/1 con histéresis: entra si RSI<entry Y cierre>SMA200;
    mantiene hasta que RSI>exit. Estado explícito (no vectorizable del todo
    por la histéresis, pero O(n) igual)."""
    s = _series(daily_df, "close")
    r = _rsi(s, rsi_period).to_numpy()
    trend = s.rolling(trend_sma_days).mean().to_numpy()
    close = s.to_numpy()
    pos = np.zeros(len(s))
    holding = False
    for i in range(len(s)):
        if np.isnan(r[i]) or np.isnan(trend[i]):
            pos[i] = 1.0 if holding else 0.0
            continue
        if not holding and r[i] < entry_below and close[i] > trend[i]:
            holding = True
        elif holding and r[i] > exit_above:
            holding = False
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=s.index)


def monthly_regime_to_daily(monthly_holding_weight: pd.Series,
                            daily_index: pd.DatetimeIndex) -> pd.Series:
    """Expande un régimen YA indexado por mes de TENENCIA (post
    `shift_to_holding`) a granularidad diaria: cada día hereda la decisión
    de su propio mes — sin mirar nunca el futuro, porque el régimen mensual
    ya era causal antes de expandirlo. Días de meses sin régimen decidido
    aún (warmup) → NaN.
    """
    daily_periods = pd.PeriodIndex(daily_index, freq="M")
    aligned = monthly_holding_weight.reindex(daily_periods)
    return pd.Series(aligned.to_numpy(), index=daily_index)


def rsi_reversion_regime_gated_position(daily_df: pd.DataFrame, *, rsi_period: int,
                                        entry_below: float, exit_above: float,
                                        trend_sma_days: int,
                                        regime_daily: pd.Series) -> pd.Series:
    """Igual que `rsi_reversion_daily_position`, pero además exige
    `regime_daily[i] == 1.0` para ABRIR una posición nueva (combinación
    multi-plazo: rebote de 2 días + tendencia SMA200 + régimen mensual de
    plazo medio). El filtro NUNCA fuerza una salida anticipada — solo
    bloquea entradas nuevas mientras el régimen no sea favorable; la salida
    sigue siendo únicamente RSI>exit_above, igual que la versión sin filtro.
    NaN en el régimen (warmup) se trata como "no favorable" (fail-closed).
    """
    s = _series(daily_df, "close")
    r = _rsi(s, rsi_period).to_numpy()
    trend = s.rolling(trend_sma_days).mean().to_numpy()
    close = s.to_numpy()
    # _series() ya dejó s.index SIN timezone (ver su docstring); si regime_daily
    # llega con tz, el reindex fallaría en SILENCIO (produce NaN en todo — nunca
    # una excepción) por comparar aware vs naive. Se normaliza aquí, no se confía
    # en que el caller lo haga bien.
    reg_idx = regime_daily.index
    if getattr(reg_idx, "tz", None) is not None:
        regime_daily = regime_daily.copy()
        regime_daily.index = reg_idx.tz_convert("UTC").tz_localize(None)
    regime = regime_daily.reindex(s.index).to_numpy()
    pos = np.zeros(len(s))
    holding = False
    for i in range(len(s)):
        if np.isnan(r[i]) or np.isnan(trend[i]):
            pos[i] = 1.0 if holding else 0.0
            continue
        regime_ok = regime[i] == 1.0
        if not holding and r[i] < entry_below and close[i] > trend[i] and regime_ok:
            holding = True
        elif holding and r[i] > exit_above:
            holding = False
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=s.index)


# ---------------------------------------------------------------------------
# Familia 5 — Rotación dual-momentum (Antonacci 2014, sin grid)
# ---------------------------------------------------------------------------

def dual_momentum_weights(equity_mclose: pd.Series, bond_mclose: pd.Series,
                          cash_monthly: pd.Series, lookback_months: int) -> pd.DataFrame:
    """Pesos {equity, bond} por mes de decisión (el resto de los casos: cash).

    En el mes m: retorno L-meses de equity, bond y cash (T-bill compuesta).
    Se sostiene el MEJOR de (equity, bond) si supera al cash; si no, cash.
    """
    idx = equity_mclose.index.intersection(bond_mclose.index)
    eq = equity_mclose.reindex(idx)
    bd = bond_mclose.reindex(idx)
    r_eq = eq / eq.shift(lookback_months) - 1.0
    r_bd = bd / bd.shift(lookback_months) - 1.0
    cash = cash_monthly.reindex(idx).fillna(0.0)
    r_cash = (1.0 + cash).rolling(lookback_months).apply(np.prod, raw=True) - 1.0
    w = pd.DataFrame(0.0, index=idx, columns=["equity", "bond"])
    valid = r_eq.notna() & r_bd.notna() & r_cash.notna()
    pick_eq = valid & (r_eq >= r_bd) & (r_eq > r_cash)
    pick_bd = valid & (r_bd > r_eq) & (r_bd > r_cash)
    w.loc[pick_eq, "equity"] = 1.0
    w.loc[pick_bd, "bond"] = 1.0
    w[~valid] = np.nan
    return w


# ---------------------------------------------------------------------------
# Familia 1 — Momentum cross-sectional (Jegadeesh-Titman 1993)
# ---------------------------------------------------------------------------

def xs_monthly_hold_returns(opens_m: pd.DataFrame, last_close_m: pd.DataFrame) -> pd.DataFrame:
    """Retorno de tenencia mensual por ticker con manejo de delistings.

    Caso normal: open(m+1)/open(m) − 1. Si el ticker no tiene apertura en m+1
    (deslistó), la salida es su ÚLTIMO cierre disponible en el mes m. Si
    tampoco hay precios en m, el retorno es 0 (posición disuelta a la entrada).
    """
    normal = opens_m.shift(-1) / opens_m - 1.0
    fallback = last_close_m / opens_m - 1.0
    out = normal.where(~normal.isna(), fallback)
    return out.fillna(0.0)


def xs_momentum_weights(monthly_close: pd.DataFrame,
                        members_by_month: pd.Series, *,
                        lookback_months: int, skip_months: int, top_n: int,
                        min_history_months: int,
                        min_coverage: float) -> tuple[pd.DataFrame, pd.Series]:
    """Pesos top-N equiponderados del ranking J-T, punto-en-el-tiempo.

    En el mes de decisión m:
        formación = close(m−skip) / close(m−skip−L) − 1
        elegibles = miembros del índice EN m (members_by_month[m]) con
                    ≥min_history_months de cierres y formación calculable.
        cartera   = top_n por formación, peso 1/top_n cada uno.
    Si la cobertura (elegibles/miembros) < min_coverage, el mes queda SIN
    decisión (NaN → excluido de la muestra, no rellenado — protocolo §4.1).

    Returns:
        (weights por mes de decisión, coverage por mes)
    """
    formation = monthly_close.shift(skip_months) / monthly_close.shift(skip_months + lookback_months) - 1.0
    history_ok = monthly_close.notna().rolling(min_history_months, min_periods=1).sum() >= min_history_months

    weights = pd.DataFrame(np.nan, index=monthly_close.index,
                           columns=monthly_close.columns)
    coverage = pd.Series(np.nan, index=monthly_close.index, dtype=float)

    for m in monthly_close.index:
        members = members_by_month.get(m)
        if not members:
            continue
        members_in_data = [t for t in members if t in monthly_close.columns]
        row_f = formation.loc[m]
        row_h = history_ok.loc[m]
        eligible = [t for t in members_in_data
                    if not np.isnan(row_f[t]) and bool(row_h[t])]
        cov = len(eligible) / len(members)
        coverage[m] = cov
        if cov < min_coverage or len(eligible) < top_n:
            continue
        top = row_f[eligible].nlargest(top_n).index
        weights.loc[m, :] = 0.0
        weights.loc[m, top] = 1.0 / top_n
    return weights, coverage
