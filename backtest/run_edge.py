"""Mide el edge de la señal quant sobre el histórico real.

    uv run python -m backtest.run_edge          # timeframe base (5m)
    uv run python -m backtest.run_edge --htf    # timeframe superior (1h)

DIAGNÓSTICO PURO: no ejecuta trades, no toca riesgo ni ejecución. Responde la
pregunta previa a cualquier overlay de sentimiento o ajuste de stops: ¿la señal
`ema_cross_rsi` predice el retorno futuro mejor que el azar? Si el IC es ~0 en
todos los horizontes y regímenes, la pata técnica del bot no tiene edge y hay que
reemplazarla, no confirmarla.
"""

from __future__ import annotations

import argparse

from src.core.config import load_settings
from backtest.edge import SIGNAL_NAME, HorizonStats, analyze_edge
from backtest.run_backtest import load_parquet


def _format_symbol(
    symbol: str, tf: str, n_velas: int, stats: list[HorizonStats], threshold: float
) -> str:
    bar = "=" * 78
    lines = [
        f"\n{bar}",
        f"{symbol} {tf} · {n_velas:,} velas · señal {SIGNAL_NAME} · "
        f"|señal|≥{threshold:.2f} para abrir",
        bar,
        " h | Spearman IC |  t(n_ef) |    n_ef | Pearson IC | acierto dir. (n)   | base ↑",
        "---+-------------+----------+---------+------------+--------------------+-------",
    ]
    for s in stats:
        hit = f"{s.hit_rate * 100:.1f}% ({s.hit_n})"
        lines.append(
            f"{s.horizon:>2} | {s.spearman_ic:>+11.4f} | {s.t_eff:>+8.2f} | "
            f"{s.n_eff:>7,} | {s.pearson_ic:>+10.4f} | {hit:<18} | "
            f"{s.base_up_rate * 100:>4.1f}%"
        )

    lines.append("")
    lines.append("Retorno medio por cuantil de señal (de la más bajista a la más alta), en %:")
    for s in stats:
        cells = "  ".join(f"{m * 100:>+7.3f}" for m in s.quantile_mean_fwd)
        lines.append(
            f"  h={s.horizon:>2} : {cells}   (spread {s.quantile_spread * 100:+.3f}%)"
        )

    lines.append("")
    lines.append("IC por régimen (Spearman) · alcista = close>EMA lenta · bajista = resto:")
    for s in stats:
        lines.append(
            f"  h={s.horizon:>2} :  alcista {s.ic_regime_up:>+.4f}    "
            f"bajista {s.ic_regime_down:>+.4f}"
        )
    return "\n".join(lines)


def main(*, htf: bool) -> int:
    cfg = load_settings()
    tf = cfg.market.htf_timeframe if htf else cfg.market.timeframe
    threshold = cfg.confluence.quant_strong_threshold

    print("Edge test de la señal quant — diagnóstico puro (no ejecuta trades).")
    print("IC = corr(señal, retorno futuro). ~0 ⇒ sin poder predictivo. "
          "|t(n_ef)| ≳ 2 ⇒ significativo")
    print("(n_ef = muestra efectiva ≈ n/horizonte, descuenta el solape de los retornos).")

    for symbol in cfg.market.symbols:
        try:
            df = load_parquet(symbol, tf)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue
        stats = analyze_edge(df, cfg)
        print(_format_symbol(symbol, tf, len(df), stats, threshold))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge test de la señal quant")
    parser.add_argument("--htf", action="store_true", help="usar el timeframe superior")
    args = parser.parse_args()
    raise SystemExit(main(htf=args.htf))
