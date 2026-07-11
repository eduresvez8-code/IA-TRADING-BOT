"""Familias 6-7 (amplitud de mercado, régimen VIX) + combos con RSI-2.

Pre-registrado en docs/research/2026-07-25_familias67_breadth_vix_protocolo.md
— ÚLTIMA ronda de búsqueda del proyecto sobre la ventana 2015-2026.

    uv run python -m backtest.run_sp500_breadth_vix

4 configuraciones ganadoras evaluadas contra TEST (una vez cada una, elegidas
SOLO por Sharpe de TRAIN):
    1. Familia 6 standalone — amplitud de mercado (timing del índice).
    2. Familia 7 standalone — régimen de VIX (timing del índice).
    3. RSI-2 (config ya publicada, sin re-tunear) + gate de amplitud.
    4. RSI-2 (config ya publicada, sin re-tunear) + gate de régimen VIX.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.data.sp500 import TRADING_DAYS_PER_YEAR
from backtest.diagnostics import (
    calmar_ratio,
    max_drawdown,
    paired_bootstrap_sharpe_diff_ci,
    sharpe,
    win_rate,
)
from backtest.run_sp500_research import (
    _fmt_gate,
    _gate_daily,
    _split_daily,
    build_members_by_month,
    build_xs_daily_close,
    load_core_data,
    select_rsi_reversion_params,
)
from backtest.sp500_families import (
    breadth_timing_position,
    daily_strategy_returns,
    market_breadth_daily,
    rsi_reversion_regime_gated_position,
    trades_from_positions,
    vix_regime_position,
)


def _eval_daily(pos: pd.Series, d: dict, cut: pd.Timestamp, per_side: float) -> dict:
    r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
    tr, te = _split_daily(r.dropna(), cut)
    return {"sh_train": sharpe(tr, TRADING_DAYS_PER_YEAR),
            "sh_test": sharpe(te, TRADING_DAYS_PER_YEAR),
            "n_train": len(tr), "n_test": len(te), "_pos": pos, "_test": te}


def _report_winner(name: str, best: dict, d: dict, cfg, cut: pd.Timestamp,
                   cut_naive: pd.Timestamp, per_side: float, bh_sh: float,
                   verdicts: list) -> None:
    hold_te = d["spy_hold_d"][d["spy_hold_d"].index >= cut_naive]
    trades = trades_from_positions(
        best["_pos"][best["_pos"].index >= cut_naive], hold_te, per_side)
    g = _gate_daily(best["_test"], trades, cfg, bh_sh)
    print(f"    trades en test: {len(trades)}  |  win-rate: {win_rate(trades):.0%} (descriptivo, no es criterio)")
    print(_fmt_gate(g))
    dd = max_drawdown(best["_test"].to_numpy())
    cal = calmar_ratio(best["_test"].to_numpy(), TRADING_DAYS_PER_YEAR)
    print(f"    MaxDD test: {dd:.1%}  |  Calmar test: {cal:+.2f}")

    passes_diff = True
    if g.passes_all:
        # Extra escrutinio (CLAUDE.md protocolo punto 5): el gate compara dos
        # Sharpe puntuales, pero no dice si la VENTAJA sobre B&H es distinguible
        # del ruido. El bootstrap pareado sí lo dice — se exige antes de llamar
        # esto un hallazgo, no después.
        rc = cfg.research
        bh_aligned = d["spy_hold_d"].reindex(best["_test"].index)
        dlo, dhi = paired_bootstrap_sharpe_diff_ci(
            best["_test"].to_numpy(), bh_aligned.to_numpy(), TRADING_DAYS_PER_YEAR,
            iterations=rc.bootstrap_iterations, ci=rc.bootstrap_ci)
        passes_diff = dlo > 0.0
        print(f"    >>> PASA LOS 5 — diagnóstico extra obligatorio:")
        print(f"    Bootstrap pareado de la VENTAJA vs B&H: [{dlo:+.2f}, {dhi:+.2f}] "
              f"(excluir 0) → {'SÍ' if passes_diff else 'no'}")
        if not passes_diff:
            print("    La ventaja de +{:.2f} Sharpe sobre B&H NO es distinguible del ruido —"
                  .format(g.sharpe_test - bh_sh))
            print("    bajo remuestreo pareado, el CI de la diferencia cruza el cero. No se")
            print("    puede llamar esto un hallazgo real pese a que el gate de 5 lo marcó.")

    verdicts.append((name, g, passes_diff))


def main() -> int:
    cfg = load_settings()
    rc = cfg.research
    cut = pd.Timestamp(rc.test_start_date)
    cut_naive = cut.tz_convert("UTC").tz_localize(None)
    per_side = (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0

    print("=" * 86)
    print("FAMILIAS 6-7 (amplitud de mercado, régimen VIX) + combos con RSI-2")
    print("ÚLTIMA ronda de búsqueda sobre la ventana 2015-2026 — pre-registrado 2026-07-25")
    print("=" * 86)

    d = load_core_data(cfg)
    bh_te_sh_d = sharpe(d["spy_hold_d"][d["spy_hold_d"].index >= cut_naive],
                        TRADING_DAYS_PER_YEAR)

    xs_daily = build_xs_daily_close(cfg)
    months = xs_daily.index.to_period("M").unique().sort_values()
    members_by_month = build_members_by_month(cfg, months)

    entry, exit_ = select_rsi_reversion_params(cfg, d, cut, per_side)
    print(f"\nRSI-2 reproducido (fijo, sin re-tunear): entry<{entry}, exit>{exit_}")

    verdicts: list[tuple[str, object]] = []

    # ================= Familia 6 — Amplitud de mercado (standalone) =================
    print("\n" + "=" * 86)
    print("FAMILIA 6 — amplitud de mercado (long SPY si amplitud > umbral, si no cash)")
    breadth_by_N = {}
    rows6 = []
    for N in rc.breadth.sma_days_grid:
        breadth = market_breadth_daily(xs_daily, members_by_month, sma_days=N,
                                       min_coverage=rc.xs_momentum.min_coverage)
        breadth_by_N[N] = breadth
        for thr in rc.breadth.threshold_grid:
            pos = breadth_timing_position(breadth, threshold=thr)
            rows6.append({"N": N, "umbral": thr, **_eval_daily(pos, d, cut, per_side)})
    print("| N (SMA) | umbral | Sh train | Sh test | n_tr | n_te |")
    print("|---|---|---|---|---|---|")
    for row in rows6:
        print(f"| {row['N']} | {row['umbral']} | {row['sh_train']:+.2f} | "
              f"{row['sh_test']:+.2f} | {row['n_train']} | {row['n_test']} |")
    best6 = max(rows6, key=lambda x: x["sh_train"])
    print(f">> elegida por TRAIN: N={best6['N']}, umbral={best6['umbral']} "
          f"(Sh train {best6['sh_train']:+.2f})")
    _report_winner("Amplitud (standalone)", best6, d, cfg, cut, cut_naive, per_side,
                   bh_te_sh_d, verdicts)

    # ================= Familia 7 — Régimen de VIX (standalone) =================
    print("\n" + "=" * 86)
    print("FAMILIA 7 — régimen de VIX (long SPY según VIX vs su propia SMA)")
    rows7 = []
    for N in rc.vix_regime.sma_days_grid:
        for direction in rc.vix_regime.directions:
            pos = vix_regime_position(d["vix_close"], sma_days=N, direction=direction)
            rows7.append({"N": N, "dir": direction, **_eval_daily(pos, d, cut, per_side)})
    print("| N (SMA) | dirección | Sh train | Sh test | n_tr | n_te |")
    print("|---|---|---|---|---|---|")
    for row in rows7:
        print(f"| {row['N']} | {row['dir']} | {row['sh_train']:+.2f} | "
              f"{row['sh_test']:+.2f} | {row['n_train']} | {row['n_test']} |")
    best7 = max(rows7, key=lambda x: x["sh_train"])
    print(f">> elegida por TRAIN: N={best7['N']}, dirección={best7['dir']} "
          f"(Sh train {best7['sh_train']:+.2f})")
    _report_winner("Régimen VIX (standalone)", best7, d, cfg, cut, cut_naive, per_side,
                   bh_te_sh_d, verdicts)

    # ================= Combo RSI-2 + Familia 6 (gate de amplitud) =================
    print("\n" + "=" * 86)
    print("RSI-2 + gate de amplitud de mercado (RSI-2 fijo; se barre el gate)")
    rowsC6 = []
    for N in rc.breadth.sma_days_grid:
        breadth = breadth_by_N[N]
        for thr in rc.breadth.threshold_grid:
            gate_pos = breadth_timing_position(breadth, threshold=thr)
            pos = rsi_reversion_regime_gated_position(
                d["spy"], rsi_period=rc.rsi_reversion.rsi_period,
                entry_below=entry, exit_above=exit_,
                trend_sma_days=rc.rsi_reversion.trend_sma_days,
                regime_daily=gate_pos)
            rowsC6.append({"N": N, "umbral": thr, **_eval_daily(pos, d, cut, per_side)})
    print("| N (SMA) | umbral | Sh train | Sh test | n_tr | n_te |")
    print("|---|---|---|---|---|---|")
    for row in rowsC6:
        print(f"| {row['N']} | {row['umbral']} | {row['sh_train']:+.2f} | "
              f"{row['sh_test']:+.2f} | {row['n_train']} | {row['n_test']} |")
    bestC6 = max(rowsC6, key=lambda x: x["sh_train"])
    print(f">> elegida por TRAIN: N={bestC6['N']}, umbral={bestC6['umbral']} "
          f"(Sh train {bestC6['sh_train']:+.2f})")
    _report_winner("RSI-2 + gate amplitud", bestC6, d, cfg, cut, cut_naive, per_side,
                   bh_te_sh_d, verdicts)

    # ================= Combo RSI-2 + Familia 7 (gate de régimen VIX) =================
    print("\n" + "=" * 86)
    print("RSI-2 + gate de régimen VIX (RSI-2 fijo; se barre el gate)")
    rowsC7 = []
    for N in rc.vix_regime.sma_days_grid:
        for direction in rc.vix_regime.directions:
            gate_pos = vix_regime_position(d["vix_close"], sma_days=N, direction=direction)
            pos = rsi_reversion_regime_gated_position(
                d["spy"], rsi_period=rc.rsi_reversion.rsi_period,
                entry_below=entry, exit_above=exit_,
                trend_sma_days=rc.rsi_reversion.trend_sma_days,
                regime_daily=gate_pos)
            rowsC7.append({"N": N, "dir": direction, **_eval_daily(pos, d, cut, per_side)})
    print("| N (SMA) | dirección | Sh train | Sh test | n_tr | n_te |")
    print("|---|---|---|---|---|---|")
    for row in rowsC7:
        print(f"| {row['N']} | {row['dir']} | {row['sh_train']:+.2f} | "
              f"{row['sh_test']:+.2f} | {row['n_train']} | {row['n_test']} |")
    bestC7 = max(rowsC7, key=lambda x: x["sh_train"])
    print(f">> elegida por TRAIN: N={bestC7['N']}, dirección={bestC7['dir']} "
          f"(Sh train {bestC7['sh_train']:+.2f})")
    _report_winner("RSI-2 + gate VIX", bestC7, d, cfg, cut, cut_naive, per_side,
                   bh_te_sh_d, verdicts)

    # ================= Veredicto integrador =================
    print("\n" + "=" * 86)
    print("VEREDICTO (4 configs, elegidas por train, test medido una vez):")
    any_pass = False
    for name, g, passes_diff in verdicts:
        if g.passes_all and passes_diff:
            status = "PASA los 5 criterios + diagnóstico extra"
        elif g.passes_all and not passes_diff:
            status = "pasa el gate de 5 pero NO el diagnóstico extra (ventaja = ruido)"
        else:
            status = "no pasa"
        any_pass = any_pass or (g.passes_all and passes_diff)
        print(f"  {name:26s} Sh test {g.sharpe_test:+.2f} vs B&H {g.sharpe_buyhold:+.2f} → {status}")
    print(f"\n  (referencia: RSI-2 solo, ya publicado 2026-07-11: Sh test +0.81 vs B&H +0.85)")
    if not any_pass:
        print("\n  NINGUNA de las 4 sobrevive el gate de 5 criterios MÁS el diagnóstico extra de")
        print("  significancia de la ventaja. Esta era la ÚLTIMA ronda de búsqueda sobre la")
        print("  ventana 2015-2026 (declarado en el pre-registro): el proyecto NO genera más")
        print("  variantes sobre este periodo. Veredicto final: indexación pasiva. El único")
        print("  paso que queda es forward/paper trading real de RSI-2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
