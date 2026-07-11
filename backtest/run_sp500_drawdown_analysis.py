"""Análisis de drawdown/Calmar sobre el protocolo YA CONGELADO (2026-07-25).

    uv run python -m backtest.run_sp500_drawdown_analysis

NO es un experimento nuevo: no se elige ninguna configuración nueva, no se
mueve el split, no se toca ningún dato. Llama a las MISMAS funciones de
selección que `run_sp500_research.py` (`select_*`, una sola fuente de verdad
para "qué config ganó por train") y a la MISMA serie de test ya fijada —
solo le aplica una métrica objetiva y mecánica adicional (drawdown máximo,
ratio de Calmar) a resultados que YA EXISTEN. No hay ninguna decisión nueva
que tomar, el número sale solo de la fórmula: por eso esto no reabre la
puerta al sobreajuste.

Motivación (docs/research/2026-07-25_calmar_sp500.md): las 5 familias del
protocolo 2026-07-11 redujeron el drawdown de forma estable, pero ninguna
superó el Sharpe de comprar-y-mantener en la década más alcista del índice.
La pregunta honesta que sigue es distinta: ¿alguna da un viaje más suave
(Calmar) que el índice, aunque no le gane en retorno bruto? Esa es la
pregunta de un ingreso pasivo sostenible, no la de vencer al mercado.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from backtest.diagnostics import calmar_ratio, max_drawdown
from backtest.run_sp500_research import (
    _split_daily,
    _split_monthly,
    load_core_data,
    select_dual_momentum,
    select_ma_timing,
    select_rsi_reversion,
    select_tsmom_index,
    select_xs_momentum,
)


def _row(label: str, config: str, test_r: pd.Series, periods_per_year: float) -> dict:
    arr = test_r.dropna().to_numpy()
    return {
        "label": label, "config": config,
        "dd": max_drawdown(arr), "calmar": calmar_ratio(arr, periods_per_year),
    }


def main() -> int:
    cfg = load_settings()
    rc = cfg.research
    cut = pd.Timestamp(rc.test_start_date)
    per_side = (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0

    print("=" * 86)
    print("DRAWDOWN / CALMAR sobre el protocolo YA CONGELADO — 2026-07-25")
    print("(mismas configs elegidas por TRAIN, mismo TEST 2015-2026 de siempre;")
    print(" métrica nueva sobre resultados YA fijados — no es un experimento nuevo)")
    print("=" * 86)

    d = load_core_data(cfg)
    bh_m = d["spy_hold_m"]["asset"]
    _, bh_m_te = _split_monthly(bh_m, cut)
    _, bh_d_te = _split_daily(d["spy_hold_d"], cut)

    rows = [_row("B&H SPY (mensual)", "comprar-y-mantener", bh_m_te, 12)]
    bh_calmar_m = rows[0]["calmar"]

    families = [
        ("TSMOM índice", select_tsmom_index),
        ("MA timing", select_ma_timing),
        ("RSI-2", select_rsi_reversion),
        ("Dual momentum", select_dual_momentum),
        ("XS momentum", select_xs_momentum),
    ]
    for label, selector in families:
        config, test_r, ppy = selector(cfg, d, cut, per_side)
        rows.append(_row(label, config, test_r, ppy))

    header = ["Estrategia", "Config (elegida por train, ya publicada)",
              "MaxDD test", "Calmar test", "vs B&H Calmar"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    bh_row = rows[0]
    print(f"| {bh_row['label']} | {bh_row['config']} | {bh_row['dd']:.1%} "
          f"| {bh_row['calmar']:+.2f} | — (la vara) |")
    any_beats = False
    for r in rows[1:]:
        beats = r["calmar"] > bh_calmar_m
        any_beats = any_beats or beats
        mark = "SÍ" if beats else "no"
        print(f"| {r['label']} | {r['config']} | {r['dd']:.1%} "
              f"| {r['calmar']:+.2f} | {mark} |")

    print("\n" + "=" * 86)
    print("VEREDICTO (Calmar = CAGR / |peor caída|; criterio: superar el Calmar de B&H):")
    if any_beats:
        print("  Al menos una familia da un viaje más suave que el índice por unidad")
        print("  de peor caída, aunque ninguna le gane en Sharpe/retorno bruto (§2026-07-11).")
    else:
        print("  NINGUNA familia supera el Calmar de comprar-y-mantener tampoco. El")
        print("  índice no solo ganó en retorno — ganó en 'viaje por unidad de dolor'.")
        print("  Verdicto reforzado: indexación pasiva es la respuesta honesta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
