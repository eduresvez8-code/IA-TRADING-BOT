"""Reversión cross-sectional entre los 5 perps LÍQUIDOS (market.symbols).

    uv run python -m backtest.run_xs_reversion_liquid

La única forma TRADEABLE de nuestro único lead real (IC negativo significativo,
[[finding-cross-sectional-reversal]]): restringirlo a los 5 majors mata el sesgo de
supervivencia (no se deslistan) y el de iliquidez (costo real ≈0.12%).

Mecánica: cada `rebalance_days`, rankear los 5 por su retorno de los últimos N días.
Ir LARGO de los `n_side` peores (apuesta a que rebotan) y CORTO de los `n_side`
mejores (apuesta a que corrigen), pesos iguales, dollar-neutral. Retorno del periodo
= media(largos) − media(cortos) − costo por rotación. NETO de costos + split OOS.

Es research puro (no opera el bot). Cero Hardcoding: universo, lookbacks, n_side,
forward/rebalance viven en config (market + cross_sectional).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import load_settings
from backtest.funding_edge import round_trip_cost
from backtest.run_backtest import load_parquet
from backtest.run_quant_hypotheses import resample


def _daily_closes(symbols: list[str]) -> pd.DataFrame:
    """Panel ancho de cierres diarios de los símbolos líquidos, alineado por fecha."""
    cols = {}
    for s in symbols:
        d = resample(load_parquet(s, "1h"), "1D").set_index("open_time")["close"]
        cols[s] = d
    return pd.DataFrame(cols).dropna().sort_index()


def _sharpe(period_returns: np.ndarray, periods_per_year: float) -> float:
    if len(period_returns) < 2 or period_returns.std() == 0:
        return 0.0
    return period_returns.mean() / period_returns.std() * np.sqrt(periods_per_year)


def _backtest(close: pd.DataFrame, lookback: int, n_side: int,
              forward: int, rebalance: int, cost_rt: float):
    """Devuelve (retornos por periodo, sharpe_anual, retorno_total, n_periodos)."""
    px = close.to_numpy(dtype=float)                 # filas=fechas, cols=activos
    n_days, n_assets = px.shape
    rets = []
    t = lookback
    while t + forward < n_days:
        mom = px[t] / px[t - lookback] - 1.0         # retorno de N días (causal)
        order = np.argsort(mom)                        # ascendente: peores primero
        losers, winners = order[:n_side], order[-n_side:]
        fwd = px[t + forward] / px[t] - 1.0           # retorno futuro del periodo
        # Reversión: LARGO perdedores, CORTO ganadores. Dollar-neutral, pesos iguales.
        gross = fwd[losers].mean() - fwd[winners].mean()
        # Costo: se abren/cierran 2·n_side patas cada rebalanceo (round-trip por lado).
        rets.append(gross - cost_rt)
        t += rebalance
    rets = np.array(rets)
    ppy = 365.0 / rebalance
    total = float(np.prod(1.0 + rets) - 1.0) if len(rets) else 0.0
    return rets, _sharpe(rets, ppy), total, len(rets)


def main() -> int:
    cfg = load_settings()
    xs = cfg.cross_sectional
    symbols = list(cfg.market.symbols)
    close = _daily_closes(symbols)
    cost_rt = round_trip_cost(cfg)

    print("=" * 84)
    print("REVERSIÓN CROSS-SECTIONAL entre 5 perps LÍQUIDOS — neto de costos, split OOS")
    print(f"Universo: {symbols}")
    print(f"Panel: {close.index.min():%Y-%m-%d} → {close.index.max():%Y-%m-%d} "
          f"({len(close)} días) | fwd {xs.forward_days}d | rebal {xs.rebalance_days}d | "
          f"costo/lado ≈ {cost_rt * 100:.2f}%")
    print("Largo de los N peores, corto de los N mejores (apuesta a reversión).")
    print("=" * 84)

    header = ["Lookback", "N/lado", "Períodos", "Sharpe", "Ret.Total",
              "Sharpe 1ª", "Sharpe 2ª", "¿estable?"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    for lb in xs.xs_liquid_lookback_days_grid:
        for n_side in xs.xs_liquid_n_side_grid:
            if 2 * n_side > len(symbols):
                continue
            rets, sh, tot, n = _backtest(
                close, lb, n_side, xs.forward_days, xs.rebalance_days, cost_rt)
            mid = len(rets) // 2
            ppy = 365.0 / xs.rebalance_days
            s1 = _sharpe(rets[:mid], ppy)
            s2 = _sharpe(rets[mid:], ppy)
            stable = ("sí" if (s1 > 0 and s2 > 0)
                      else ("no" if (s1 < 0 and s2 < 0) else "mixto"))
            print(f"| {lb}d | {n_side} | {n} | {sh:+.2f} | {tot * 100:+.1f}% | "
                  f"{s1:+.2f} | {s2:+.2f} | {stable} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
