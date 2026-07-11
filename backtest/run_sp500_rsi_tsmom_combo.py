"""RSI-2 + filtro de régimen TSMOM (2026-07-25) — pre-registrado en
docs/research/2026-07-25_rsi2_tsmom_combo_protocolo.md.

    uv run python -m backtest.run_sp500_rsi_tsmom_combo

RSI-2 propio (entry/exit) NO se re-tunea para esta combinación: se reproduce
la MISMA selección por train ya publicada el 2026-07-11 (mismo grid, misma
regla), nunca un valor literal nuevo. Se barre SOLO el lookback del filtro
de régimen TSMOM sobre el grid YA existente ({3,6,12}), elegido por Sharpe
de train entre esos 3 — el test se mide una vez con el ganador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.data.sp500 import TRADING_DAYS_PER_YEAR
from backtest.diagnostics import calmar_ratio, max_drawdown, sharpe
from backtest.run_sp500_research import (
    _fmt_gate,
    _gate_daily,
    _split_daily,
    load_core_data,
)
from backtest.sp500_families import (
    daily_strategy_returns,
    monthly_regime_to_daily,
    rsi_reversion_daily_position,
    rsi_reversion_regime_gated_position,
    shift_to_holding,
    trades_from_positions,
    tsmom_index_weights,
)


def _select_rsi_params(cfg, d, cut, per_side) -> tuple[float, float]:
    """Reproduce EXACTAMENTE la selección de RSI-2 ya publicada (mismo grid,
    misma regla: máximo Sharpe de train) — no se hardcodea el resultado, se
    recalcula para garantizar que es la misma config congelada."""
    rc = cfg.research
    rows = []
    for e in rc.rsi_reversion.entry_grid:
        for x in rc.rsi_reversion.exit_grid:
            pos = rsi_reversion_daily_position(
                d["spy"], rsi_period=rc.rsi_reversion.rsi_period,
                entry_below=e, exit_above=x,
                trend_sma_days=rc.rsi_reversion.trend_sma_days)
            r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
            tr, _ = _split_daily(r.dropna(), cut)
            rows.append({"entry": e, "exit": x, "sh_train": sharpe(tr, TRADING_DAYS_PER_YEAR)})
    best = max(rows, key=lambda x: x["sh_train"])
    return best["entry"], best["exit"]


def main() -> int:
    cfg = load_settings()
    rc = cfg.research
    cut = pd.Timestamp(rc.test_start_date)
    cut_naive = cut.tz_convert("UTC").tz_localize(None)
    per_side = (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0

    print("=" * 86)
    print("RSI-2 + FILTRO DE RÉGIMEN TSMOM — pre-registrado 2026-07-25")
    print("RSI-2 fijo en su config ya elegida por train (2026-07-11, no se re-tunea);")
    print("se barre L del régimen TSMOM sobre {3,6,12} (grid ya existente), L elegido")
    print("SOLO por train.")
    print("=" * 86)

    d = load_core_data(cfg)
    bh_te_sh_d = sharpe(d["spy_hold_d"][d["spy_hold_d"].index >= cut_naive],
                        TRADING_DAYS_PER_YEAR)

    entry, exit_ = _select_rsi_params(cfg, d, cut, per_side)
    print(f"\nRSI-2 reproducido: entry<{entry}, exit>{exit_} "
          f"(debe coincidir con el publicado el 2026-07-11)")

    rows = []
    for L in rc.tsmom_index.lookback_months_grid:
        w_raw = tsmom_index_weights(d["spy_mclose"], L)
        w_holding = shift_to_holding(w_raw)["asset"]
        regime_daily = monthly_regime_to_daily(w_holding, d["spy_hold_d"].index)
        pos = rsi_reversion_regime_gated_position(
            d["spy"], rsi_period=rc.rsi_reversion.rsi_period,
            entry_below=entry, exit_above=exit_,
            trend_sma_days=rc.rsi_reversion.trend_sma_days,
            regime_daily=regime_daily)
        r = daily_strategy_returns(pos, d["spy_hold_d"], d["tbill_d"], per_side)
        tr, te = _split_daily(r.dropna(), cut)
        rows.append({"L": L, "sh_train": sharpe(tr, TRADING_DAYS_PER_YEAR),
                    "sh_test": sharpe(te, TRADING_DAYS_PER_YEAR),
                    "n_train": len(tr), "n_test": len(te),
                    "_pos": pos, "_test": te})

    print("\n| L (régimen TSMOM) | Sh train | Sh test | n_tr | n_te |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['L']} | {row['sh_train']:+.2f} | {row['sh_test']:+.2f} "
              f"| {row['n_train']} | {row['n_test']} |")

    best = max(rows, key=lambda x: x["sh_train"])
    print(f"\n>> elegido por TRAIN: L={best['L']} (Sh train {best['sh_train']:+.2f})")

    hold_te = d["spy_hold_d"][d["spy_hold_d"].index >= cut_naive]
    trades = trades_from_positions(
        best["_pos"][best["_pos"].index >= cut_naive], hold_te, per_side)
    print(f"    trades en test: {len(trades)}")

    g = _gate_daily(best["_test"], trades, cfg, bh_te_sh_d)
    print(_fmt_gate(g))

    dd = max_drawdown(best["_test"].to_numpy())
    cal = calmar_ratio(best["_test"].to_numpy(), TRADING_DAYS_PER_YEAR)
    print(f"    MaxDD test: {dd:.1%}  |  Calmar test: {cal:+.2f}  "
          f"(referencia 2026-07-25: RSI-2 solo 10.9% DD / +0.52 Calmar, "
          f"B&H 23.3% DD / +0.59 Calmar)")

    print("\n" + "=" * 86)
    print("VEREDICTO:", "PASA los 5 criterios" if g.passes_all else "no pasa el listón")
    if not g.passes_all:
        print("\nÉsta es la SEGUNDA extensión sobre la ventana 2015-2026 desde el veredicto")
        print("original. Por la nota de honestidad del pre-registro: no generar una TERCERA")
        print("variante sobre esta misma ventana — el riesgo de sobreajuste por búsqueda")
        print("repetida ya empieza a acumularse aunque cada intento individual sea disciplinado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
