"""Runner de la matriz de research del Slow Path (embudo de 2 etapas).

    uv run python -m backtest.run_quant_matrix

Esta sesión ejecuta SOLO la Familia E (cash-and-carry de funding, delta-neutral)
sobre los 5 activos de `scan.symbols`, con el perfil de costos conservador de
`quant_matrix` (taker 0.05% VIP0, capital 2× delta-neutral). Escupe el DataFrame
comparativo (IC · t-stat · Sharpe · MaxDD · PF · yield neto) y aplica la Regla de
Oro. Las familias B/C/D están registradas como stubs para sus sesiones.

IC sale `N/A` para el carry A PROPÓSITO: el IC de Spearman mide correlación
señal→retorno futuro y el carry NO es una señal predictiva, es un yield
estructural. Su gate de significancia es el t-stat de la media del retorno neto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from backtest.quant_matrix import simulate_carry


def _carry_row(stats) -> dict:
    return {
        "Familia": "E·carry",
        "Activo": stats.symbol,
        "IC": "N/A",  # no aplica a yield estructural (ver docstring)
        "t-stat": round(stats.t_stat, 1),
        "Sharpe": round(stats.sharpe, 2),
        "MaxDD%": round(stats.max_drawdown * 100, 2),
        "PF": round(stats.profit_factor, 2),
        "YieldBruto%": round(stats.gross_yield_ann_pct, 2),
        "YieldNeto%": round(stats.net_yield_ann_pct, 2),
        "%PerNeg": round(stats.pct_negative_periods, 1),
        "PeorPer%": round(stats.worst_period_pct, 3),
        "Folds": f"{stats.folds_same_sign}/{stats.n_folds}",
        "Golden": "✅" if stats.passes_golden else "—",
    }


def main() -> int:
    cfg = load_settings()
    qm = cfg.quant_matrix
    fdir = Path(cfg.storage.funding_dir)

    print("=" * 100)
    print("MATRIZ CUANTITATIVA — Slow Path research · Familia E: Cash-and-Carry (delta-neutral)")
    print(f"Costos: taker {qm.taker_commission_pct}%/lado · capital {qm.carry_capital_multiplier}× "
          f"notional · mantenimiento {qm.carry_maintenance_bps_per_period} bps/8h")
    print(f"Regla de Oro: |t| ≥ {qm.golden_min_tstat} · PF > {qm.golden_min_profit_factor} · "
          f"signo consistente 4/4 folds")
    print("=" * 100)

    rows = []
    for sym in cfg.scan.symbols:
        path = fdir / f"{sym}_funding.parquet"
        if not path.exists():
            print(f"[WARN] falta {path} — sáltalo")
            continue
        funding = pd.read_parquet(path).sort_values("funding_time")
        stats = simulate_carry(funding["funding_rate"], qm, symbol=sym)
        rows.append(_carry_row(stats))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    winners = [r for r in rows if r["Golden"] == "✅"]
    print("\n## Veredicto (Regla de Oro)")
    if not winners:
        print("Ningún activo pasa la Regla de Oro como estrategia coronable.")
    else:
        print(f"{len(winners)}/{len(rows)} activos pasan |t|≥{qm.golden_min_tstat} ∧ "
              f"PF>{qm.golden_min_profit_factor} ∧ 4/4 folds: "
              f"{', '.join(r['Activo'] for r in winners)}")
    print("\nLectura honesta: para un YIELD estructural el t-stat es fácil de pasar "
          "(funding casi siempre positivo). Lo discriminante es el Sharpe ajustado por "
          "fricción de capital y el MaxDD de los episodios de funding negativo. Mirar "
          "esas dos columnas antes que el ✅.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
