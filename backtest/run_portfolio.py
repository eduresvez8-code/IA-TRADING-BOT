"""Backtest del portafolio long-short de reversión cross-sectional (neto de costos).

    uv run python -m backtest.run_portfolio

Construye el portafolio robusto a la cola (filtro de liquidez + winsorización +
pesos inversos a la vol con tope), largo perdedores / corto ganadores, rebalanceo
semanal. Reporta Sharpe, Max DD, Profit Factor y el aporte del lado LARGO vs
SHORT por separado. Regla de Oro: PF>umbral, retorno neto positivo y 4/4 tramos
walk-forward positivos, o se descarta por "no cosechable".
"""

from __future__ import annotations

import math

from src.core.config import load_settings
from backtest.portfolio import backtest_reversal, load_universe_fields


def _pf(x: float) -> str:
    return "∞" if math.isinf(x) else f"{x:.2f}"


def main() -> int:
    cfg = load_settings()
    xs = cfg.cross_sectional
    close, qvol = load_universe_fields(cfg)
    one_way = (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0

    print("Backtest LONG-SHORT de REVERSIÓN cross-sectional — neto de costos.")
    print(f"Panel: {close.shape[1]} activos · {close.index.min():%Y-%m-%d} → "
          f"{close.index.max():%Y-%m-%d}")
    print(f"Robustez: fuera {xs.liquidity_drop_pct:.0%} menor volumen · winsor "
          f"{xs.winsorize_quantile:.0%}/{1 - xs.winsorize_quantile:.0%} · "
          f"peso máx {xs.max_weight:.0%}/activo · pesos inv-vol")
    print(f"Costo {one_way * 100:.2f}%/lado (turnover) · rebalanceo {xs.rebalance_days}d · "
          f"PF mínimo (edge) {cfg.scan.edge_profit_factor_min}")

    print("\n| mom | n sem | univ | Ret neto | Sharpe | MaxDD | PF | Win% | "
          "Largo Q1 | Short Q5 | WF + | EDGE |")
    print("|-----|-------|------|----------|--------|-------|----|----|---------|"
          "----------|------|------|")
    for lb in (14, 30):
        r = backtest_reversal(close, qvol, cfg, lookback=lb)
        print(f"| {lb}d | {r.n_periods:>5} | {r.avg_universe:>4.0f} | "
              f"{r.total_return * 100:>+7.1f}% | {r.ann_sharpe:>+5.2f} | "
              f"{r.max_drawdown * 100:>4.1f}% | {_pf(r.profit_factor)} | "
              f"{r.win_rate * 100:>3.0f}% | {r.avg_long_leg * 100:>+6.2f}% | "
              f"{r.avg_short_contrib * 100:>+6.2f}% | {r.folds_positive}/4 | "
              f"{'★' if r.is_edge else '·'} |")

    print("\nDesglose por tramo walk-forward (retorno neto compuesto):")
    for lb in (14, 30):
        r = backtest_reversal(close, qvol, cfg, lookback=lb)
        tramos = "  ".join(f"{x * 100:+.1f}%" for x in r.fold_returns)
        print(f"  {lb}d: {tramos}")

    print("\n## Veredicto (Regla de Oro)")
    edge_found = False
    for lb in (14, 30):
        r = backtest_reversal(close, qvol, cfg, lookback=lb)
        if r.is_edge:
            edge_found = True
            print(f"  ★ momentum {lb}d: PF {_pf(r.profit_factor)}, Sharpe {r.ann_sharpe:+.2f}, "
                  f"ret {r.total_return * 100:+.1f}%, 4/4 tramos positivos.")
    if not edge_found:
        print("Ningún lookback cumple PF>umbral + retorno positivo + 4/4 tramos. "
              "La reversión NO es cosechable con este portafolio: se descarta.")
    print("\n⚠ Recordatorio: sesgo de supervivencia (perps vivos hoy) sigue presente — "
          "un resultado positivo aún debería re-validarse con universo point-in-time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
