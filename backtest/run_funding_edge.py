"""Edge test de funding rate y basis sobre el histórico real.

    uv run python -m backtest.run_funding_edge

Para cada símbolo de `scan.symbols`: carga el funding (8h) y el basis (1h) desde
`storage.funding_dir`, el precio spot 1h de `data/candles/`, y mide si predicen el
retorno futuro a los horizontes de `funding_edge.forward_horizons_hours`.

Aplica la REGLA DE ORO: marca ★ una celda solo si |spread de cuantiles| supera el
costo ida-vuelta Y |t|≥2. Y resume la consistencia de signo entre los 5 activos:
una señal con edge real debe predecir en la MISMA dirección en todos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from backtest.funding_edge import analyze_basis, analyze_funding, round_trip_cost
from backtest.run_backtest import load_parquet


def _cell(s) -> str:
    mark = "★" if s.tradable else " "
    return f"{s.spearman_ic:+.3f}/{s.quantile_spread * 100:+.2f}%{mark}"


def _matrix(title: str, per_symbol: dict, horizons: list[int]) -> str:
    head = " | ".join(f"{h}h" for h in horizons)
    lines = [f"\n### {title}  (celda = IC Spearman / spread cuantil ; ★ = supera costo y |t|≥2)",
             "",
             f"| Símbolo | {head} |",
             "|---|" + "---|" * len(horizons)]
    for sym, stats in per_symbol.items():
        cells = " | ".join(_cell(s) for s in stats)
        lines.append(f"| {sym} | {cells} |")
    return "\n".join(lines)


def _sign_consistency(per_symbol: dict, horizons: list[int]) -> str:
    """Por horizonte: ¿cuántos de los 5 activos comparten el signo de la IC?"""
    lines = ["", "Consistencia de signo entre activos (nº de 5 con el mismo signo de IC):"]
    for j, h in enumerate(horizons):
        ics = [stats[j].spearman_ic for stats in per_symbol.values()]
        pos = sum(1 for ic in ics if ic > 0)
        neg = sum(1 for ic in ics if ic < 0)
        dom = max(pos, neg)
        lines.append(f"  {h:>3}h: {dom}/5 mismo signo "
                     f"({'positivo' if pos >= neg else 'negativo'})")
    return "\n".join(lines)


def main() -> int:
    cfg = load_settings()
    fdir = Path(cfg.storage.funding_dir)
    interval = cfg.funding_edge.premium_interval
    horizons = cfg.funding_edge.forward_horizons_hours
    cost = round_trip_cost(cfg)

    print("Edge test de señales NO-precio (funding rate / basis) — diagnóstico puro.")
    print(f"Costo ida-vuelta (Regla de Oro): {cost * 100:.2f}% · "
          f"IC>0 = momentum, IC<0 = contrario · |t|≥2 = significativo.")

    funding_by_sym: dict = {}
    basis_by_sym: dict = {}
    for sym in cfg.scan.symbols:
        try:
            price_1h = load_parquet(sym, "1h")
            funding = pd.read_parquet(fdir / f"{sym}_funding.parquet")
            premium = pd.read_parquet(fdir / f"{sym}_premium_{interval}.parquet")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        funding_by_sym[sym] = analyze_funding(funding, price_1h, cfg)
        basis_by_sym[sym] = analyze_basis(premium, price_1h, cfg)

    print(_matrix("FUNDING RATE → retorno futuro", funding_by_sym, horizons))
    print(_sign_consistency(funding_by_sym, horizons))
    print(_matrix("BASIS / PREMIUM → retorno futuro", basis_by_sym, horizons))
    print(_sign_consistency(basis_by_sym, horizons))

    tradables = []
    for label, d in (("funding", funding_by_sym), ("basis", basis_by_sym)):
        for sym, stats in d.items():
            for s in stats:
                if s.tradable:
                    tradables.append((label, sym, s))
    print("\n## Veredicto (Regla de Oro)")
    if not tradables:
        print("Ninguna señal supera el costo con significancia. Sin candidata a estrategia.")
    else:
        print(f"{len(tradables)} celda(s) superan el screen bruto (revisar consistencia "
              "cross-activo y walk-forward antes de programar):")
        for label, sym, s in sorted(tradables, key=lambda x: abs(x[2].quantile_spread),
                                    reverse=True):
            print(f"  ★ {label} · {sym} · {s.horizon_h}h → IC {s.spearman_ic:+.3f} "
                  f"(t {s.t_eff:+.1f}), spread {s.quantile_spread * 100:+.2f}% vs costo "
                  f"{cost * 100:.2f}%, tramos mismo signo {s.folds_same_sign}/{s.n_folds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
