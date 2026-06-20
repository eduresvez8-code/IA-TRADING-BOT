"""Runner de la matriz de research del Slow Path (embudo de 2 etapas).

    uv run python -m backtest.run_quant_matrix

Ejecuta las familias IMPLEMENTADAS sobre los 5 activos de `scan.symbols` con el
perfil de costos conservador de `quant_matrix` (taker 0.05% VIP0):

    Familia B · Cointegración de pares (1h)      → IC del z-score del spread
    Familia C · Reversión a VWAP intradía (5m)   → IC de la desviación al VWAP
    Familia E · Cash-and-Carry de funding (8h)   → yield estructural delta-neutral

Cada familia imprime su propia tabla (esquemas distintos: B/C son señales
predictivas con IC; E es un yield sin IC). La Regla de Oro corona solo lo que
pasa |t|≥golden_min_tstat ∧ PF>golden_min_profit_factor ∧ signo 4/4 folds.

D (squeeze de volatilidad) queda como stub para su sesión modular.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_settings
from backtest.pairs import run_pairs_all
from backtest.quant_matrix import simulate_carry
from backtest.vwap import simulate_vwap


# --------------------------------- Familia B ---------------------------------

def _run_pairs(cfg, cdir: Path) -> list[dict]:
    series = {}
    for sym in cfg.scan.symbols:
        path = cdir / f"{sym}_1h.parquet"
        if path.exists():
            series[sym] = np.log(pd.read_parquet(path)["close"].to_numpy(dtype=float))
    if len(series) < 2:
        return []
    log_prices = pd.DataFrame(series)
    rows = []
    for st in run_pairs_all(log_prices, cfg.quant_matrix):
        rows.append({
            "Par": st.pair, "IC": round(st.ic_spearman, 4),
            "t_corr": round(st.ic_tstat, 2), "Trades": st.n_trades,
            "NetoAnual%": round(st.net_return_ann_pct, 1),
            "Sharpe": round(st.sharpe, 2), "MaxDD%": round(st.max_drawdown * 100, 1),
            "PF": round(st.profit_factor, 2),
            "Folds": f"{st.folds_same_sign}/{st.n_folds}",
            "Golden": "✅" if st.passes_golden else "—",
        })
    return rows


# --------------------------------- Familia C ---------------------------------

def _run_vwap(cfg, cdir: Path) -> list[dict]:
    rows = []
    for sym in cfg.scan.symbols:
        path = cdir / f"{sym}_5m.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        st = simulate_vwap(df, cfg.quant_matrix, symbol=sym)
        rows.append({
            "Activo": st.symbol, "IC": round(st.ic_spearman, 4),
            "t_corr": round(st.ic_tstat, 2), "Trades": st.n_trades,
            "HoldMed": round(st.avg_holding_bars, 1),
            "NetoAnual%": round(st.net_return_ann_pct, 2),
            "Sharpe": round(st.sharpe, 2), "MaxDD%": round(st.max_drawdown * 100, 1),
            "PF": round(st.profit_factor, 2), "%Gana": round(st.pct_winning_trades, 1),
            "Folds": f"{st.folds_same_sign}/{st.n_folds}",
            "Golden": "✅" if st.passes_golden else "—",
        })
    return rows


# --------------------------------- Familia E ---------------------------------

def _run_carry(cfg, fdir: Path) -> list[dict]:
    rows = []
    for sym in cfg.scan.symbols:
        path = fdir / f"{sym}_funding.parquet"
        if not path.exists():
            continue
        funding = pd.read_parquet(path).sort_values("funding_time")
        st = simulate_carry(funding["funding_rate"], cfg.quant_matrix, symbol=sym)
        rows.append({
            "Activo": st.symbol, "IC": "N/A", "t-stat": round(st.t_stat, 1),
            "Sharpe": round(st.sharpe, 2), "MaxDD%": round(st.max_drawdown * 100, 2),
            "PF": round(st.profit_factor, 2),
            "YieldBruto%": round(st.gross_yield_ann_pct, 2),
            "YieldNeto%": round(st.net_yield_ann_pct, 2),
            "%PerNeg": round(st.pct_negative_periods, 1),
            "Folds": f"{st.folds_same_sign}/{st.n_folds}",
            "Golden": "✅" if st.passes_golden else "—",
        })
    return rows


def _section(title: str, rows: list[dict], note: str = "") -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if not rows:
        print("  (sin datos)")
        return
    print(pd.DataFrame(rows).to_string(index=False))
    winners = [r for r in rows if r.get("Golden") == "✅"]
    key = "Par" if "Par" in rows[0] else "Activo"
    if winners:
        print(f"\n  Pasan la Regla de Oro: {', '.join(r[key] for r in winners)}")
    else:
        print("\n  Ninguno pasa la Regla de Oro.")
    if note:
        print(f"  {note}")


def main() -> int:
    cfg = load_settings()
    qm = cfg.quant_matrix
    cdir = Path(cfg.storage.candles_dir)
    fdir = Path(cfg.storage.funding_dir)

    print("MATRIZ CUANTITATIVA — Slow Path research (embudo de 2 etapas)")
    print(f"Costos: taker {qm.taker_commission_pct}%/lado · slippage {qm.slippage_pct}% + "
          f"{qm.slippage_atr_mult}·ATR · Regla de Oro |t|≥{qm.golden_min_tstat} ∧ "
          f"PF>{qm.golden_min_profit_factor} ∧ 4/4 folds")

    _section(
        "Familia B · Cointegración de pares (1h) — espera IC < 0 (reversión del spread)",
        _run_pairs(cfg, cdir),
        note="IC>0 = el spread hace momentum, no revierte (par NO cointegrado a 1h).",
    )
    _section(
        "Familia C · Reversión a VWAP intradía (5m) — espera IC < 0, entrada bounce-robust",
        _run_vwap(cfg, cdir),
        note="El IC se mide saltando la barra inmediata (sin bid-ask bounce). "
             "Lo decisivo es NetoAnual% tras costos, no el IC.",
    )
    _section(
        "Familia E · Cash-and-Carry de funding (8h) — yield estructural (sin IC)",
        _run_carry(cfg, fdir),
        note="Para un yield el t-stat pasa trivial; mirar MaxDD y %PerNeg, no el ✅.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
