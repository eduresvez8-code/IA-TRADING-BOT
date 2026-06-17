"""Edge test de 3 métodos de sentimiento con el Fear & Greed como ancla de régimen.

    uv run python -m backtest.run_sentiment_regime

Carga el F&G diario (storage.funding_dir/fng_daily.parquet) y el precio diario de
BTC/ETH, y aplica los 3 métodos (regime-switching, MR gateado, vol-scaling).
Regla de Oro: un método pasa solo si su test muestra |t|>2 Y una estrategia con
PF>umbral neto de costos. Si los 3 fallan, es la prueba final de que el dataset
no tiene alpha de sentimiento explotable con herramientas públicas.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.core.config import load_settings
from backtest.portfolio import load_universe_fields
from backtest.sentiment_regime import (
    REGIME_ORDER,
    backtest_long_flat,
    build_daily,
    label_regime,
    mr_gated_ic,
    mr_signal,
    regime_ic,
    regime_stats,
    vol_scaling,
)


def _pf(x: float) -> str:
    return "∞" if math.isinf(x) else f"{x:.2f}"


def _load_fng(cfg) -> pd.Series:
    path = Path(cfg.storage.funding_dir) / "fng_daily.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    return df.set_index("date")["value"]


def _daily_close(cfg, symbol: str) -> pd.Series:
    close, _ = load_universe_fields(cfg)
    s = close[symbol].copy()
    s.index = s.index.normalize()
    return s[~s.index.duplicated()]


def _metrics_row(m, edge: bool) -> str:
    return (f"| {m.name:<26} | {m.n_trades:>3} | {m.exposure * 100:>3.0f}% | "
            f"{m.total_return * 100:>+7.1f}% | {m.ann_sharpe:>+5.2f} | "
            f"{m.max_drawdown * 100:>4.1f}% | {m.win_rate * 100:>3.0f}% | "
            f"{m.expectancy * 100:>+5.2f}% | {_pf(m.profit_factor)} | "
            f"{'★' if edge else '·'} |")


def main() -> int:
    cfg = load_settings()
    sr = cfg.sentiment_regime
    one_way = (cfg.backtest.commission_pct + cfg.backtest.slippage_pct) / 100.0
    pf_min = cfg.scan.edge_profit_factor_min
    fng = _load_fng(cfg)

    print("Edge test de SENTIMIENTO POR RÉGIMEN (Fear & Greed) — diagnóstico, no opera.")
    print(f"Costo {one_way * 100:.2f}%/lado · forward {sr.forward_days}d · "
          f"PF mínimo (edge) {pf_min} · regímenes <{sr.ext_fear_below}/<{sr.fear_below}/"
          f"<{sr.greed_above}/<{sr.ext_greed_above}")

    any_edge = False
    for symbol in ("BTCUSDT", "ETHUSDT"):
        df = build_daily(_daily_close(cfg, symbol), fng)
        print(f"\n{'=' * 78}\n{symbol} · {len(df)} días "
              f"{df.index.min():%Y-%m-%d}→{df.index.max():%Y-%m-%d}\n{'=' * 78}")

        # --- Método A: regime-switching ---
        stats = regime_stats(df, sr)
        ic, ic_t = regime_ic(df, sr)
        # Significancia del método A: IC del F&G O alguna media de régimen ≠ incond.
        sig_a = abs(ic_t) >= 2 or any(abs(s.t_vs_uncond) >= 2 for s in stats)
        print(f"── A. Regime-switching · IC(F&G→fwd{sr.forward_days}d) = {ic:+.3f} "
              f"(t {ic_t:+.1f}; + = momentum, − = contrarian) · "
              f"{'SIGNIFICATIVO' if sig_a else 'no significativo'} ──")
        print("  régimen   | n días | fwd medio | fwd mediana | win% | t vs incond.")
        for s in stats:
            print(f"  {s.regime:<9} | {s.n_days:>6} | {s.mean_fwd * 100:>+7.2f}% | "
                  f"{s.median_fwd * 100:>+7.2f}% | {s.win_rate * 100:>3.0f}% | {s.t_vs_uncond:>+5.1f}")

        reg = df["fng"].map(lambda v: label_regime(v, sr))
        # Cada estrategia se empareja con la significancia del método que la respalda.
        strategies = [
            (backtest_long_flat(df, reg.isin(["ExtFear", "Fear"]), one_way=one_way,
                                pf_min=pf_min, name="A: long miedo (contrarian)"), sig_a),
            (backtest_long_flat(df, reg.isin(["Greed", "ExtGreed"]), one_way=one_way,
                                pf_min=pf_min, name="A: long codicia (momentum)"), sig_a),
        ]

        # --- Método B: MR gateado por sentimiento ---
        gated = mr_gated_ic(df, sr)
        sig_b = any(g.subset == "extremo" and abs(g.t) >= 2 for g in gated)
        print(f"── B. Mean-reversion gated · IC del señal MR por subset · "
              f"{'SIGNIFICATIVO en extremos' if sig_b else 'no significativo'} ──")
        for g in gated:
            print(f"  {g.subset:<8}: n={g.n:>4} IC={g.ic:+.3f} t={g.t:+.1f}")
        sig = mr_signal(df["close"], sr.mr_lookback_days)
        extreme = (df["fng"] - 50).abs() >= sr.extreme_abs_threshold
        strategies.append((backtest_long_flat(
            df, (sig > 0) & extreme, one_way=one_way, pf_min=pf_min,
            name="B: MR gateado (extremo)"), sig_b))

        # --- Matriz de métricas (★ = significativo |t|>2 Y PF>umbral, la Regla completa) ---
        print("── Matriz de estrategias (neto de costos · ★ exige |t|>2 Y PF>umbral) ──")
        print("| estrategia                 |  n | exp | Ret tot | Sharpe | MaxDD | Win | Expect | PF | ★ |")
        print("|----------------------------|----|-----|---------|--------|-------|-----|--------|----|----|")
        for m, sig in strategies:
            edge = bool(sig and m.is_edge)
            print(_metrics_row(m, edge))
            any_edge = any_edge or edge

        # --- Método C: vol-scaling ---
        bh, sc = vol_scaling(df, sr, one_way=one_way)
        print("── C. Vol-scaling por F&G (reducir en euforia) ──")
        print(f"  buy_hold  : ret {bh.total_return * 100:+.1f}% · Sharpe {bh.ann_sharpe:+.2f} · MaxDD {bh.max_drawdown * 100:.1f}%")
        print(f"  fng_scaled: ret {sc.total_return * 100:+.1f}% · Sharpe {sc.ann_sharpe:+.2f} · MaxDD {sc.max_drawdown * 100:.1f}%")
        better = sc.ann_sharpe > bh.ann_sharpe and sc.max_drawdown < bh.max_drawdown
        print(f"  → {'MEJORA' if better else 'no mejora'} riesgo-ajustado "
              f"(Sharpe {sc.ann_sharpe - bh.ann_sharpe:+.2f}, MaxDD {(sc.max_drawdown - bh.max_drawdown) * 100:+.1f}pp)")
        any_edge = any_edge or better

    print("\n## Veredicto (Regla de Oro)")
    if any_edge:
        print("Algún método mostró ventaja — revisar arriba la(s) celda(s) ★ / MEJORA "
              "y validar consistencia antes de construir.")
    else:
        print("NINGUNO de los 3 métodos de sentimiento supera la Regla de Oro "
              "(|t|>2 + PF>1.15 neto, o mejora de riesgo robusta). Prueba final: el "
              "dataset no contiene alpha de sentimiento explotable con herramientas públicas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
