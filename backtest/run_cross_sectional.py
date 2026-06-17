"""Edge test del momentum cross-sectional sobre el universo de perps.

    uv run python -m backtest.run_cross_sectional

Carga el panel diario (storage.universe_dir/daily.parquet) y mide el IC
cross-sectional del factor momentum contra el retorno futuro. Corre varias
especificaciones (lookbacks distintos + variante ajustada por volatilidad) como
CHEQUEO DE ROBUSTEZ: un edge real sobrevive a parámetros cercanos; un espejismo
solo aparece en uno. Aplica la Regla de Oro (spread>costo, |t|≥2, consistencia).
"""

from __future__ import annotations

from backtest.cross_sectional import analyze, load_universe_panel
from src.core.config import load_settings


def _row(r) -> str:
    spec = f"{r.lookback}d{'·vol' if r.vol_adjust else ''}"
    mark = "★" if r.beats_cost else " "
    return (f"| {spec:<8} | {r.n_dates:>3} | {r.avg_universe:>5.0f} | "
            f"{r.mean_ic:>+6.3f} | {r.t_stat:>+5.1f} | {r.ic_positive_rate * 100:>4.0f}% | "
            f"{r.mean_quantile_spread * 100:>+6.2f}% | {r.fold_same_sign}/{len(r.fold_mean_ics)} | {mark} |")


def main() -> int:
    cfg = load_settings()
    xs = cfg.cross_sectional
    close = load_universe_panel(cfg)

    print("Edge test de MOMENTUM CROSS-SECTIONAL — diagnóstico puro (no opera).")
    print(f"Panel: {close.shape[1]} activos · {close.index.min():%Y-%m-%d} → "
          f"{close.index.max():%Y-%m-%d} · {close.shape[0]} días")
    print(f"Factor: retorno de N días · forward {xs.forward_days}d · rebalanceo "
          f"{xs.rebalance_days}d · quintiles {xs.n_quantiles}")
    cost = analyze(close, cfg).cost
    print(f"Costo ida-vuelta (Regla de Oro): {cost * 100:.2f}% · IC = corr(ranking factor, "
          f"retorno futuro) · |t|≥2 significativo · ★ = spread>costo y |t|≥2\n")

    print("| spec     |  n | univ |  IC m |  t-st | IC>0 | spread |  WF  | ★ |")
    print("|----------|----|------|-------|-------|------|--------|------|---|")
    specs = [(14, False), (30, False), (60, False), (90, False), (30, True)]
    results = []
    for lb, va in specs:
        r = analyze(close, cfg, lookback=lb, vol_adjust=va)
        results.append(r)
        print(_row(r))

    print("\n## Veredicto (Regla de Oro)")
    edges = [r for r in results if r.beats_cost]
    consistent = [r for r in results if abs(r.t_stat) >= 2
                  and r.fold_same_sign == len(r.fold_mean_ics) and len(r.fold_mean_ics) > 0]
    if edges and consistent:
        print(f"Señal con base: {len(edges)} spec(s) superan el screen y "
              f"{len(consistent)} son consistentes en TODOS los tramos walk-forward.")
        for r in consistent:
            print(f"  ★ {r.lookback}d{'·vol' if r.vol_adjust else ''}: IC {r.mean_ic:+.3f} "
                  f"(t {r.t_stat:+.1f}), spread {r.mean_quantile_spread * 100:+.2f}% vs costo "
                  f"{cost * 100:.2f}%, WF {r.fold_same_sign}/{len(r.fold_mean_ics)}")
        print("\n⚠ Recordatorio: sesgo de supervivencia (solo perps vivos hoy) infla esto. "
              "Confirmar con universo point-in-time antes de programar.")
    else:
        print("Ninguna especificación supera la Regla de Oro con consistencia. "
              "El factor no muestra un edge robusto y significativo en este universo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
