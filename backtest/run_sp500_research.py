"""Runner del protocolo pre-registrado S&P 500 (2026-07-11).

    uv run python -m backtest.run_sp500_research

Ejecuta EXACTAMENTE docs/research/2026-07-11_protocolo_sp500.md:
    1. Por familia: grid completo medido en TRAIN (< research.test_start_date).
    2. Selección de la config por Sharpe de TRAIN (jamás por test).
    3. TEST medido UNA vez con la config congelada.
    4. Gate de 5 criterios (Sharpe>0.5, bootstrap, concentración, mitades,
       vs buy-and-hold SPY) + sensibilidad de costos (5 pb).

La columna de test del grid se imprime completa para el REPORTE (ver la
distribución, práctica de todo el proyecto) pero NO participa en la selección.
Cero Hardcoding: todos los grids/umbrales vienen de config/settings.yaml.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import Settings, load_settings
from src.data.sp500 import (
    TRADING_DAYS_PER_YEAR,
    load_membership,
    load_prices,
    members_asof,
    normalize_ticker_for_yahoo,
    tbill_daily_return,
)
from backtest.diagnostics import GateResult, evaluate_gate, sharpe
from backtest.sp500_families import (
    daily_close_series,
    daily_hold_returns,
    daily_strategy_returns,
    dual_momentum_weights,
    first_open_by_month,
    golden_cross_daily_position,
    last_close_by_month,
    ma_timing_monthly_weights,
    monthly_cash_returns,
    monthly_hold_returns,
    monthly_strategy_returns,
    rsi_reversion_daily_position,
    shift_to_holding,
    trades_from_positions,
    tsmom_index_weights,
    xs_momentum_weights,
    xs_monthly_hold_returns,
)

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Split y utilidades de reporte
# ---------------------------------------------------------------------------

def _split_monthly(r: pd.Series, cut: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Serie mensual (PeriodIndex) → (train, test). El mes del corte va al TEST."""
    cut_period = pd.Period(cut.tz_convert("UTC").tz_localize(None), freq="M")
    return r[r.index < cut_period], r[r.index >= cut_period]


def _split_daily(r: pd.Series, cut: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    naive_cut = cut.tz_convert("UTC").tz_localize(None)
    return r[r.index < naive_cut], r[r.index >= naive_cut]


def _month_ts(idx: pd.PeriodIndex) -> np.ndarray:
    return idx.to_timestamp(how="end").values


def _fmt_gate(g: GateResult) -> str:
    def mark(b):
        return "SÍ" if b else "no"
    conc = "n/a (pérdida neta)" if math.isnan(g.concentration) else f"{g.concentration:.0%}"
    return (
        f"    Sharpe test          : {g.sharpe_test:+.2f}   (umbral > 0.5)          → {mark(g.passes_sharpe)}\n"
        f"    Bootstrap CI 90%     : [{g.ci_lo:+.2f}, {g.ci_hi:+.2f}] (excluir 0)     → {mark(g.passes_bootstrap)}\n"
        f"    Concentración top10% : {conc}   (< 60%)             → {mark(g.passes_concentration)}\n"
        f"    Mitades del test     : {g.sharpe_h1:+.2f} / {g.sharpe_h2:+.2f} (ambas > 0)      → {mark(g.passes_halves)}\n"
        f"    vs B&H SPY           : {g.sharpe_test:+.2f} vs {g.sharpe_buyhold:+.2f} (superarlo)     → {mark(g.passes_vs_buyhold)}\n"
        f"    >>> PASA LOS 5: {'SÍ' if g.passes_all else 'NO'}"
    )


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_core_data(cfg: Settings) -> dict:
    data_dir = Path(cfg.data.dir)
    spy = load_prices(data_dir, cfg.market.benchmark_symbol)
    irx = load_prices(data_dir, cfg.market.tbill_symbol)
    bond = load_prices(data_dir, cfg.research.dual_momentum.bond_symbol)
    vix = load_prices(data_dir, cfg.market.vix_symbol)

    irx_close = irx.set_index("open_time")["close"]
    irx_close.index = irx_close.index.tz_convert("UTC").tz_localize(None)
    tbill_d = tbill_daily_return(irx_close)

    spy_hold_m = monthly_hold_returns(spy)
    cash_m = monthly_cash_returns(tbill_d, spy_hold_m.index)

    return {
        "spy": spy, "bond": bond, "tbill_d": tbill_d,
        "spy_mclose": last_close_by_month(spy),
        "spy_hold_m": spy_hold_m.to_frame("asset"),
        "cash_m": cash_m,
        "spy_hold_d": daily_hold_returns(spy),
        "vix_close": daily_close_series(vix),
    }


def build_xs_matrices(cfg: Settings) -> dict:
    """Matrices mensuales del universo (cierres, aperturas, último cierre).

    Derivadas de los parquet por-ticker; se cachean en data.dir (regenerables:
    borrar los xs_*.parquet fuerza el rebuild).
    """
    data_dir = Path(cfg.data.dir)
    cache = {k: data_dir / f"xs_monthly_{k}.parquet"
             for k in ("close", "open", "lastclose")}
    if all(p.exists() for p in cache.values()):
        out = {}
        for k, p in cache.items():
            df = pd.read_parquet(p)
            df.index = pd.PeriodIndex(df.index, freq="M")
            out[k] = df
        return out

    prices_dir = data_dir / "prices"
    skip = {cfg.market.benchmark_symbol, cfg.market.index_symbol,
            cfg.market.tbill_symbol, *cfg.data.extra_symbols}
    closes, opens, lastcloses = {}, {}, {}
    for f in sorted(prices_dir.glob("*.parquet")):
        sym = f.stem
        if sym in skip:
            continue
        df = load_prices(data_dir, sym)
        closes[sym] = last_close_by_month(df)
        opens[sym] = first_open_by_month(df)
        lastcloses[sym] = last_close_by_month(df)
    out = {
        "close": pd.DataFrame(closes).sort_index(),
        "open": pd.DataFrame(opens).sort_index(),
        "lastclose": pd.DataFrame(lastcloses).sort_index(),
    }
    for k, p in cache.items():
        tosave = out[k].copy()
        tosave.index = tosave.index.to_timestamp()
        tosave.to_parquet(p)
    return out


def build_xs_daily_close(cfg: Settings) -> pd.DataFrame:
    """Cierres DIARIOS del universo punto-en-el-tiempo (Familia 6 — amplitud).

    A diferencia de `build_xs_matrices` (mensual, para XS momentum), la
    amplitud compara cada ticker con su propia SMA de N días — necesita
    resolución diaria. Reutiliza los parquet por-ticker ya descargados (no
    hace falta bajar nada nuevo); se cachea en data.dir (regenerable:
    borrar xs_daily_close.parquet fuerza el rebuild).
    """
    data_dir = Path(cfg.data.dir)
    cache = data_dir / "xs_daily_close.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    prices_dir = data_dir / "prices"
    skip = {cfg.market.benchmark_symbol, cfg.market.index_symbol,
            cfg.market.tbill_symbol, cfg.market.vix_symbol, *cfg.data.extra_symbols}
    closes = {}
    for f in sorted(prices_dir.glob("*.parquet")):
        sym = f.stem
        if sym in skip:
            continue
        closes[sym] = daily_close_series(load_prices(data_dir, sym))
    out = pd.DataFrame(closes).sort_index()
    out.to_parquet(cache)
    return out


def build_members_by_month(cfg: Settings, months: pd.PeriodIndex) -> pd.Series:
    membership = load_membership(Path(cfg.data.dir) / "constituents.csv")
    data = {}
    for m in months:
        when = m.to_timestamp(how="end").tz_localize("UTC")
        data[m] = [normalize_ticker_for_yahoo(t) for t in members_asof(membership, when)]
    return pd.Series(data)


# ---------------------------------------------------------------------------
# Evaluadores por familia (mensual y diario)
# ---------------------------------------------------------------------------

def _monthly_row(r: pd.Series, cut: pd.Timestamp) -> dict:
    tr, te = _split_monthly(r.dropna(), cut)
    return {"sh_train": sharpe(tr, MONTHS_PER_YEAR), "sh_test": sharpe(te, MONTHS_PER_YEAR),
            "n_train": len(tr), "n_test": len(te), "train": tr, "test": te}


def _drop_warmup_months(r: pd.Series, weights: pd.DataFrame) -> pd.Series:
    """Excluye de la muestra los meses SIN decisión previa válida (warmup de
    indicadores, cobertura insuficiente). Sin esto, el warmup contaría como
    "cash devengando T-bill" — meses planos que no son la estrategia."""
    w_hold = shift_to_holding(weights)
    valid = w_hold.notna().any(axis=1).reindex(r.index).fillna(False)
    return r[valid]


def _daily_row(r: pd.Series, cut: pd.Timestamp) -> dict:
    tr, te = _split_daily(r.dropna(), cut)
    return {"sh_train": sharpe(tr, TRADING_DAYS_PER_YEAR),
            "sh_test": sharpe(te, TRADING_DAYS_PER_YEAR),
            "n_train": len(tr), "n_test": len(te), "train": tr, "test": te}


def _print_grid(rows: list[dict], keys: list[str]) -> None:
    header = keys + ["Sh train", "Sh test", "n_tr", "n_te"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        vals = [str(row[k]) for k in keys]
        print(f"| {' | '.join(vals)} | {row['sh_train']:+.2f} | {row['sh_test']:+.2f} "
              f"| {row['n_train']} | {row['n_test']} |")


def _gate_monthly(test_r: pd.Series, cfg: Settings, bh_sharpe: float) -> GateResult:
    rc = cfg.research
    return evaluate_gate(
        test_r.to_numpy(), _month_ts(test_r.index), test_r.to_numpy(),
        MONTHS_PER_YEAR, sharpe_min=rc.success_sharpe_min,
        iterations=rc.bootstrap_iterations, ci=rc.bootstrap_ci,
        concentration_max=rc.concentration_max, sharpe_buyhold=bh_sharpe)


def _gate_daily(test_r: pd.Series, trades: list[float], cfg: Settings,
                bh_sharpe: float) -> GateResult:
    rc = cfg.research
    years = max(len(test_r) / TRADING_DAYS_PER_YEAR, 1e-9)
    upy = max(len(trades) / years, 1e-9)
    return evaluate_gate(
        test_r.to_numpy(), test_r.index.values, np.asarray(trades),
        TRADING_DAYS_PER_YEAR, sharpe_min=rc.success_sharpe_min,
        iterations=rc.bootstrap_iterations, ci=rc.bootstrap_ci,
        concentration_max=rc.concentration_max, sharpe_buyhold=bh_sharpe,
        units_per_year=upy)


# ---------------------------------------------------------------------------
# Selección por familia (config elegida SOLO por train) — reutilizada por
# run_sp500_research.py (gate de 5 criterios) Y por run_sp500_drawdown_
# analysis.py (Calmar/drawdown, 2026-07-25): AMBOS deben ver EXACTAMENTE la
# misma config ganadora y la misma serie de test — una sola fuente de verdad,
# nunca dos selecciones que puedan desincronizarse.
# ---------------------------------------------------------------------------

def select_tsmom_index(cfg: Settings, d: dict, cut: pd.Timestamp,
                       per_side: float) -> tuple[str, pd.Series, float]:
    rc = cfg.research
    rows = []
    for L in rc.tsmom_index.lookback_months_grid:
        w = tsmom_index_weights(d["spy_mclose"], L)
        r = _drop_warmup_months(
            monthly_strategy_returns(w, d["spy_hold_m"], d["cash_m"], per_side), w)
        rows.append({"L": L, **_monthly_row(r, cut), "_w": w})
    best = max(rows, key=lambda x: x["sh_train"])
    return f"L={best['L']}", best["test"], MONTHS_PER_YEAR


def select_ma_timing(cfg: Settings, d: dict, cut: pd.Timestamp,
                     per_side: float) -> tuple[str, pd.Series, float]:
    rc = cfg.research
    rows3 = []
    for N in rc.ma_timing.sma_days_grid:
        w = ma_timing_monthly_weights(d["spy"], N)
        r = _drop_warmup_months(
            monthly_strategy_returns(w, d["spy_hold_m"], d["cash_m"], per_side), w)
        rows3.append({"config": f"SMA{N}@fin-de-mes", "freq": "M",
                      **_monthly_row(r, cut), "_w": w})
    for fast, slow in rc.ma_timing.cross_pairs:
        pos = golden_cross_daily_position(d["spy"], fast, slow)
        r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
        rows3.append({"config": f"cruce{fast}/{slow}@diario", "freq": "D",
                      **_daily_row(r, cut), "_pos": pos})
    best3 = max(rows3, key=lambda x: x["sh_train"])
    freq = MONTHS_PER_YEAR if best3["freq"] == "M" else TRADING_DAYS_PER_YEAR
    return best3["config"], best3["test"], freq


def select_rsi_reversion(cfg: Settings, d: dict, cut: pd.Timestamp,
                         per_side: float) -> tuple[str, pd.Series, float]:
    rc = cfg.research
    rows4 = []
    for e in rc.rsi_reversion.entry_grid:
        for x in rc.rsi_reversion.exit_grid:
            pos = rsi_reversion_daily_position(
                d["spy"], rsi_period=rc.rsi_reversion.rsi_period,
                entry_below=e, exit_above=x,
                trend_sma_days=rc.rsi_reversion.trend_sma_days)
            r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
            rows4.append({"entry": e, "exit": x, **_daily_row(r, cut), "_pos": pos})
    best4 = max(rows4, key=lambda x: x["sh_train"])
    return (f"entry<{best4['entry']},exit>{best4['exit']}", best4["test"],
            TRADING_DAYS_PER_YEAR)


def select_rsi_reversion_params(cfg: Settings, d: dict, cut: pd.Timestamp,
                                per_side: float) -> tuple[float, float]:
    """Como `select_rsi_reversion` pero devuelve (entry, exit) crudos —
    reutilizado por los combos de regime-gating (TSMOM, amplitud, VIX) que
    necesitan la config de RSI-2 YA elegida como entrada FIJA, sin re-tunearla
    ni duplicar la selección en cada runner (una sola fuente de verdad)."""
    rc = cfg.research
    rows = []
    for e in rc.rsi_reversion.entry_grid:
        for x in rc.rsi_reversion.exit_grid:
            pos = rsi_reversion_daily_position(
                d["spy"], rsi_period=rc.rsi_reversion.rsi_period,
                entry_below=e, exit_above=x,
                trend_sma_days=rc.rsi_reversion.trend_sma_days)
            r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
            rows.append({"entry": e, "exit": x, **_daily_row(r, cut)})
    best = max(rows, key=lambda x: x["sh_train"])
    return best["entry"], best["exit"]


def select_dual_momentum(cfg: Settings, d: dict, cut: pd.Timestamp,
                         per_side: float) -> tuple[str, pd.Series, float]:
    rc = cfg.research
    bond_hold_m = monthly_hold_returns(d["bond"])
    assets_m = pd.concat({"equity": d["spy_hold_m"]["asset"], "bond": bond_hold_m},
                         axis=1).dropna()
    cash_m5 = monthly_cash_returns(d["tbill_d"], assets_m.index)
    w5 = dual_momentum_weights(d["spy_mclose"].reindex(assets_m.index),
                               last_close_by_month(d["bond"]).reindex(assets_m.index),
                               cash_m5, rc.dual_momentum.lookback_months)
    r5 = _drop_warmup_months(
        monthly_strategy_returns(w5, assets_m, cash_m5, per_side), w5)
    row5 = _monthly_row(r5, cut)
    return f"L={rc.dual_momentum.lookback_months}", row5["test"], MONTHS_PER_YEAR


def select_xs_momentum(cfg: Settings, d: dict, cut: pd.Timestamp,
                       per_side: float) -> tuple[str, pd.Series, float]:
    rc = cfg.research
    xs = build_xs_matrices(cfg)
    members_by_month = build_members_by_month(cfg, xs["close"].index)
    xs_hold = xs_monthly_hold_returns(xs["open"], xs["lastclose"])
    min_hist_months = math.ceil(rc.xs_momentum.min_history_days
                                / (TRADING_DAYS_PER_YEAR / MONTHS_PER_YEAR))
    cash_xs = monthly_cash_returns(d["tbill_d"], xs["close"].index)
    rows1 = []
    for L in rc.xs_momentum.lookback_months_grid:
        for S in rc.xs_momentum.skip_months_grid:
            for N in rc.xs_momentum.top_n_grid:
                w, cov = xs_momentum_weights(
                    xs["close"], members_by_month, lookback_months=L,
                    skip_months=S, top_n=N,
                    min_history_months=min_hist_months,
                    min_coverage=rc.xs_momentum.min_coverage)
                r = _drop_warmup_months(
                    monthly_strategy_returns(w, xs_hold, cash_xs, per_side), w)
                rows1.append({"L": L, "S": S, "N": N, **_monthly_row(r, cut),
                              "_w": w, "_cov": cov})
    best1 = max(rows1, key=lambda x: x["sh_train"])
    return (f"L={best1['L']},S={best1['S']},N={best1['N']}", best1["test"],
            MONTHS_PER_YEAR)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    cfg = load_settings()
    rc = cfg.research
    cut = pd.Timestamp(rc.test_start_date)
    per_side = (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0
    stress_side = (cfg.backtest.commission_pct + rc.slippage_stress_pct) / 100.0

    print("=" * 86)
    print("PROTOCOLO S&P 500 — corrida pre-registrada 2026-07-11")
    print(f"split: TRAIN < {rc.test_start_date} ≤ TEST | costos {per_side*1e4:.0f} pb/lado "
          f"(estrés {stress_side*1e4:.0f} pb) | cash = T-bill")
    print("=" * 86)

    d = load_core_data(cfg)

    # ---- Benchmarks B&H (la vara del criterio 5) ----
    bh_m = d["spy_hold_m"]["asset"]
    bh_m_tr, bh_m_te = _split_monthly(bh_m, cut)
    bh_d_tr, bh_d_te = _split_daily(d["spy_hold_d"], cut)
    bh_te_sh_m = sharpe(bh_m_te, MONTHS_PER_YEAR)
    bh_te_sh_d = sharpe(bh_d_te, TRADING_DAYS_PER_YEAR)
    print("\nB&H SPY (retorno total):")
    print(f"  mensual: train {sharpe(bh_m_tr, 12):+.2f} | test {bh_te_sh_m:+.2f} "
          f"(ret anual test {(1+bh_m_te.mean())**12-1:+.1%})")
    print(f"  diario : train {sharpe(bh_d_tr, 252):+.2f} | test {bh_te_sh_d:+.2f}")

    verdicts: list[tuple[str, str, GateResult]] = []

    # ================= Familia 2 — TSMOM sobre el índice =================
    print("\n" + "=" * 86)
    print("FAMILIA 2 — TSMOM índice (SPY, mensual, long/cash, excluye último mes)")
    rows = []
    for L in rc.tsmom_index.lookback_months_grid:
        w = tsmom_index_weights(d["spy_mclose"], L)
        r = _drop_warmup_months(
            monthly_strategy_returns(w, d["spy_hold_m"], d["cash_m"], per_side), w)
        rows.append({"L": L, **_monthly_row(r, cut), "_w": w})
    _print_grid(rows, ["L"])
    best = max(rows, key=lambda x: x["sh_train"])
    print(f">> elegida por TRAIN: L={best['L']} (Sh train {best['sh_train']:+.2f})")
    g = _gate_monthly(best["test"], cfg, bh_te_sh_m)
    print(_fmt_gate(g))
    r_stress = monthly_strategy_returns(best["_w"], d["spy_hold_m"], d["cash_m"], stress_side)
    print(f"    estrés 5 pb: Sharpe test {_monthly_row(r_stress, cut)['sh_test']:+.2f}")
    verdicts.append(("TSMOM índice", f"L={best['L']}", g))

    # ================= Familia 3 — Timing por media móvil =================
    print("\n" + "=" * 86)
    print("FAMILIA 3 — timing por media móvil (SPY, long/cash)")
    rows3 = []
    for N in rc.ma_timing.sma_days_grid:
        w = ma_timing_monthly_weights(d["spy"], N)
        r = _drop_warmup_months(
            monthly_strategy_returns(w, d["spy_hold_m"], d["cash_m"], per_side), w)
        rows3.append({"config": f"SMA{N}@fin-de-mes", "freq": "M",
                      **_monthly_row(r, cut), "_w": w})
    for fast, slow in rc.ma_timing.cross_pairs:
        pos = golden_cross_daily_position(d["spy"], fast, slow)
        r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
        rows3.append({"config": f"cruce{fast}/{slow}@diario", "freq": "D",
                      **_daily_row(r, cut), "_pos": pos})
    _print_grid(rows3, ["config", "freq"])
    best3 = max(rows3, key=lambda x: x["sh_train"])
    print(f">> elegida por TRAIN: {best3['config']} (Sh train {best3['sh_train']:+.2f})")
    if best3["freq"] == "M":
        g3 = _gate_monthly(best3["test"], cfg, bh_te_sh_m)
        r_stress = monthly_strategy_returns(best3["_w"], d["spy_hold_m"], d["cash_m"], stress_side)
        stress_sh = _monthly_row(r_stress, cut)["sh_test"]
    else:
        pos = best3["_pos"]
        r_all = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
        _, te = _split_daily(r_all, cut)
        cut_naive = cut.tz_convert("UTC").tz_localize(None)
        trades = trades_from_positions(pos[pos.index >= cut_naive],
                                       d["spy_hold_d"][d["spy_hold_d"].index >= cut_naive],
                                       per_side)
        g3 = _gate_daily(te, trades, cfg, bh_te_sh_d)
        r_stress = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], stress_side)
        stress_sh = _daily_row(r_stress, cut)["sh_test"]
    print(_fmt_gate(g3))
    print(f"    estrés 5 pb: Sharpe test {stress_sh:+.2f}")
    verdicts.append(("MA timing", best3["config"], g3))

    # ================= Familia 4 — Reversión RSI-2 =================
    print("\n" + "=" * 86)
    print("FAMILIA 4 — reversión RSI-2 (SPY, diario, long/cash, filtro SMA200)")
    rows4 = []
    for e in rc.rsi_reversion.entry_grid:
        for x in rc.rsi_reversion.exit_grid:
            pos = rsi_reversion_daily_position(
                d["spy"], rsi_period=rc.rsi_reversion.rsi_period,
                entry_below=e, exit_above=x,
                trend_sma_days=rc.rsi_reversion.trend_sma_days)
            r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
            rows4.append({"entry": e, "exit": x, **_daily_row(r, cut), "_pos": pos})
    _print_grid(rows4, ["entry", "exit"])
    best4 = max(rows4, key=lambda x: x["sh_train"])
    print(f">> elegida por TRAIN: entry<{best4['entry']}, exit>{best4['exit']} "
          f"(Sh train {best4['sh_train']:+.2f})")
    cut_naive = cut.tz_convert("UTC").tz_localize(None)
    hold_te = d["spy_hold_d"][d["spy_hold_d"].index >= cut_naive]
    trades4 = trades_from_positions(best4["_pos"][best4["_pos"].index >= cut_naive],
                                    hold_te, per_side)
    g4 = _gate_daily(best4["test"], trades4, cfg, bh_te_sh_d)
    print(f"    trades en test: {len(trades4)}")
    print(_fmt_gate(g4))
    r_stress = daily_strategy_returns(best4["_pos"], d["spy_hold_d"], d["tbill_d"], stress_side)
    print(f"    estrés 5 pb: Sharpe test {_daily_row(r_stress, cut)['sh_test']:+.2f}")
    verdicts.append(("RSI-2", f"entry<{best4['entry']},exit>{best4['exit']}", g4))

    # ================= Familia 5 — Dual momentum (sin grid) =================
    print("\n" + "=" * 86)
    print("FAMILIA 5 — rotación dual-momentum (SPY vs bono vs T-bill, mensual, sin grid)")
    bond_hold_m = monthly_hold_returns(d["bond"])
    assets_m = pd.concat({"equity": d["spy_hold_m"]["asset"], "bond": bond_hold_m},
                         axis=1).dropna()
    cash_m5 = monthly_cash_returns(d["tbill_d"], assets_m.index)
    w5 = dual_momentum_weights(d["spy_mclose"].reindex(assets_m.index),
                               last_close_by_month(d["bond"]).reindex(assets_m.index),
                               cash_m5, rc.dual_momentum.lookback_months)
    r5 = _drop_warmup_months(
        monthly_strategy_returns(w5, assets_m, cash_m5, per_side), w5)
    row5 = _monthly_row(r5, cut)
    print(f"única config (L={rc.dual_momentum.lookback_months}m): "
          f"Sh train {row5['sh_train']:+.2f} | Sh test {row5['sh_test']:+.2f} "
          f"| n {row5['n_train']}/{row5['n_test']}")
    g5 = _gate_monthly(row5["test"], cfg, bh_te_sh_m)
    print(_fmt_gate(g5))
    r_stress = monthly_strategy_returns(w5, assets_m, cash_m5, stress_side)
    print(f"    estrés 5 pb: Sharpe test {_monthly_row(r_stress, cut)['sh_test']:+.2f}")
    verdicts.append(("Dual momentum", f"L={rc.dual_momentum.lookback_months}", g5))

    # ================= Familia 1 — Momentum cross-sectional =================
    print("\n" + "=" * 86)
    print("FAMILIA 1 — momentum cross-sectional (universo punto-en-el-tiempo)")
    xs = build_xs_matrices(cfg)
    members_by_month = build_members_by_month(cfg, xs["close"].index)
    xs_hold = xs_monthly_hold_returns(xs["open"], xs["lastclose"])
    min_hist_months = math.ceil(rc.xs_momentum.min_history_days
                                / (TRADING_DAYS_PER_YEAR / MONTHS_PER_YEAR))
    cash_xs = monthly_cash_returns(d["tbill_d"], xs["close"].index)

    rows1 = []
    for L in rc.xs_momentum.lookback_months_grid:
        for S in rc.xs_momentum.skip_months_grid:
            for N in rc.xs_momentum.top_n_grid:
                w, cov = xs_momentum_weights(
                    xs["close"], members_by_month, lookback_months=L,
                    skip_months=S, top_n=N,
                    min_history_months=min_hist_months,
                    min_coverage=rc.xs_momentum.min_coverage)
                r = _drop_warmup_months(
                    monthly_strategy_returns(w, xs_hold, cash_xs, per_side), w)
                rows1.append({"L": L, "S": S, "N": N, **_monthly_row(r, cut),
                              "_w": w, "_cov": cov})
    _print_grid(rows1, ["L", "S", "N"])
    best1 = max(rows1, key=lambda x: x["sh_train"])
    cov = best1["_cov"].dropna()
    cov_tr, cov_te = _split_monthly(cov, cut)
    print(f">> elegida por TRAIN: L={best1['L']} S={best1['S']} N={best1['N']} "
          f"(Sh train {best1['sh_train']:+.2f})")
    print(f"    cobertura punto-en-el-tiempo: train media {cov_tr.mean():.0%} "
          f"| test media {cov_te.mean():.0%} (piso de operabilidad pre-registrado: 85%)")
    g1 = _gate_monthly(best1["test"], cfg, bh_te_sh_m)
    print(_fmt_gate(g1))
    r_stress = _drop_warmup_months(
        monthly_strategy_returns(best1["_w"], xs_hold, cash_xs, stress_side), best1["_w"])
    print(f"    estrés 5 pb: Sharpe test {_monthly_row(r_stress, cut)['sh_test']:+.2f}")
    op_flag = "NO operable (cobertura test < 85%)" if cov_te.mean() < 0.85 else "cobertura OK"
    print(f"    operabilidad por supervivencia: {op_flag}")
    verdicts.append(("XS momentum", f"L={best1['L']},S={best1['S']},N={best1['N']}", g1))

    # ================= Veredicto integrador =================
    print("\n" + "=" * 86)
    print("VEREDICTO (5 familias, config elegida por train, test medido una vez):")
    any_pass = False
    for name, config, g in verdicts:
        status = "PASA los 5 criterios" if g.passes_all else "no pasa"
        any_pass = any_pass or g.passes_all
        print(f"  {name:16s} {config:22s} Sh test {g.sharpe_test:+.2f} "
              f"vs B&H {g.sharpe_buyhold:+.2f} → {status}")
    if not any_pass:
        print("\n  NINGUNA familia pasa el listón pre-registrado. Según el criterio de")
        print("  parada del protocolo: se DETIENE la búsqueda y el veredicto honesto es")
        print("  indexación pasiva (comprar y mantener el índice), no un bot de timing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
