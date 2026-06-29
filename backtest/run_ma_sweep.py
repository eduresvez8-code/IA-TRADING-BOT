"""Sweep de cruces de medias móviles sobre el histórico real (guiado por foros).

    uv run python -m backtest.run_ma_sweep

Barre TODOS los pares fast/slow más citados por la comunidad cripto (9/21, golden
cross 50/200, etc.) × tipo (EMA/SMA) × timeframe (1h/4h/1d) × 5 perps, NETO de costos.
Se EXCLUYE <1h a propósito: foros y nuestro propio hallazgo coinciden en que el ruido
y el costo a 5m/15m matan el edge.

Agrega por configuración (Sharpe medio entre los 5 activos) para ver cuáles son
realmente las "mejores" medias en NUESTRA data, y destaca el golden cross 50/200 SMA
en diario. Escribe el detalle completo en backtest/reports/.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr
from backtest.engine import BacktestEngine
from backtest.quant_hypotheses import make_macross_decider, moving_average
from backtest.run_backtest import load_parquet
from backtest.run_quant_hypotheses import ASSETS, Row, resample

_RESAMPLE = {"1h": None, "4h": "4h", "1d": "1D"}


def _frame(asset: str, tf: str) -> pd.DataFrame:
    df = load_parquet(asset, "1h")
    rule = _RESAMPLE[tf]
    return df if rule is None else resample(df, rule)


def run_sweep(cfg, engine) -> list[Row]:
    qh = cfg.quant_hypotheses
    rows: list[Row] = []
    for tf in qh.ma_cross_timeframes:
        for asset in ASSETS:
            df = _frame(asset, tf)
            closes = df["close"].to_numpy(dtype=float)
            atrs = atr(df, cfg.risk.atr_period).to_numpy()
            for kind in qh.ma_cross_types:
                for fast_p, slow_p in qh.ma_cross_pairs:
                    fast = moving_average(closes, fast_p, kind)
                    slow = moving_average(closes, slow_p, kind)
                    decider = make_macross_decider(
                        closes, atrs, fast, slow,
                        atr_mult=qh.atr_stop_mult, allow_short=qh.ma_cross_allow_short)
                    res = engine.run(df, asset, tf, decider=decider)
                    label = f"{kind.upper()}{fast_p}/{slow_p}@{tf}"
                    rows.append(Row(label, asset, tf, res))
    return rows


def _aggregate(rows: list[Row]) -> list[dict]:
    """Agrupa por configuración (label sin activo) y promedia Sharpe/CAGR entre activos."""
    by_cfg: dict[str, list[Row]] = {}
    for r in rows:
        by_cfg.setdefault(r.label, []).append(r)
    out = []
    for label, rs in by_cfg.items():
        sharpes = [x.sharpe for x in rs]
        out.append({
            "label": label,
            "mean_sharpe": float(np.mean(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "n_pos": sum(1 for s in sharpes if s > 0),
            "mean_cagr": float(np.mean([x.cagr for x in rs])),
            "mean_trades": float(np.mean([x.n_trades for x in rs])),
        })
    out.sort(key=lambda d: d["mean_sharpe"], reverse=True)
    return out


def _summary_table(agg: list[dict], top: int) -> str:
    head = ["Config", "Sharpe medio", "Sharpe mín", "Pos/5", "CAGR medio", "Trades medio"]
    lines = [f"\n### Top {top} configuraciones (ordenadas por Sharpe medio entre los 5 activos)",
             "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for d in agg[:top]:
        lines.append(f"| {d['label']} | {d['mean_sharpe']:+.2f} | {d['min_sharpe']:+.2f} | "
                     f"{d['n_pos']}/5 | {d['mean_cagr']:+.1f}% | {d['mean_trades']:.0f} |")
    return "\n".join(lines)


def _golden_cross_table(rows: list[Row]) -> str:
    head = ["Activo", "Trades", "Win", "PF", "Sharpe", "CAGR", "MaxDD"]
    lines = ["\n### Golden cross clásico — SMA 50/200 en diario (1d), por activo",
             "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if r.label == "SMA50/200@1d":
            pf = "∞" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
            lines.append(f"| {r.asset} | {r.n_trades} | {r.win_rate:.0f}% | {pf} | "
                         f"{r.sharpe:+.2f} | {r.cagr:+.1f}% | {r.max_dd:.0f}% |")
    return "\n".join(lines)


def main() -> int:
    cfg = load_settings()
    engine = BacktestEngine(cfg)
    rt_cost = 2 * (cfg.backtest.commission_pct + cfg.backtest.slippage_pct)

    print("=" * 78)
    print("SWEEP DE CRUCES DE MEDIAS MÓVILES — neto de costos, histórico real (~3 años)")
    print(f"Pares×tipos×timeframes×5 activos · costo ida-vuelta base ≈ {rt_cost:.2f}% + ATR")
    print("Excluido <1h (ruido + costo). Sharpe anualizado. Agregado entre los 5 perps.")
    print("=" * 78)

    rows = run_sweep(cfg, engine)
    agg = _aggregate(rows)
    n_total = len(agg)
    n_pos = sum(1 for d in agg if d["mean_sharpe"] > 0)

    print(_summary_table(agg, top=15))
    print(f"\nConfiguraciones con Sharpe medio > 0: {n_pos}/{n_total}")
    print(_golden_cross_table(rows))

    report = [
        "# Sweep de cruces de medias móviles (guiado por foros)",
        f"\nGenerado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · "
        f"costo ida-vuelta base ≈ {rt_cost:.2f}% + slippage ATR · ~3 años (2023-06→2026-06).",
        "\nMétricas NETAS de costos. Pares más citados por la comunidad cripto. "
        "Se excluye <1h a propósito (ruido + costo, confirmado por nuestro hallazgo previo).",
        _summary_table(agg, top=20),
        f"\nConfiguraciones con Sharpe medio > 0: **{n_pos}/{n_total}**.",
        _golden_cross_table(rows),
    ]
    REPORTS_DIR = Path("backtest/reports")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"ma_sweep_{stamp}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Reporte escrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
