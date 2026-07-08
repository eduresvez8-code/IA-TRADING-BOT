""""Cortar pérdidas rápido, dejar correr las ganancias" aplicado a los modelos
que NUNCA se probaron con esa estructura (2026-07-08).

    uv run python -m backtest.run_letwinners

Ya probados con tp=None de antes (NO se repiten aquí, ver docs/research/):
TSMOM, cruces de medias (240 combos), lead-lag, breakout/mean-reversion.

Aquí: Donchian (antes con take-profit fijo) y las señales de posicionamiento
z-score (antes sin stop de riesgo real). Protocolo anti-selección: split FIJO
por mitades (igual que run_ma_split.py), config elegida SOLO por train, test
medido una vez, grid completo reportado.
"""

from __future__ import annotations

import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr
from backtest.engine import BacktestEngine
from backtest.positioning import make_zscore_decider, rolling_zscore
from backtest.quant_hypotheses import (
    align_funding_to_bars,
    make_donchian_decider,
)
from backtest.run_backtest import load_parquet
from backtest.run_positioning import load_klines_1h as pos_load_klines_1h
from backtest.run_positioning import load_metrics_1h as pos_load_metrics_1h
from backtest.run_quant_hypotheses import ASSETS, load_funding, resample


def _donchian_sharpe(engine, cfg, df, asset, entry, exit_ema):
    qh = cfg.quant_hypotheses
    df = df.reset_index(drop=True)
    closes = df["close"].to_numpy(dtype=float)
    atrs = atr(df, cfg.risk.atr_period).to_numpy()
    funding_frac = align_funding_to_bars(df, load_funding(asset))
    decider = make_donchian_decider(
        closes, atrs, funding_frac,
        entry_period=entry, exit_ema_period=exit_ema,
        funding_min_frac=qh.donchian_funding_min_8h_pct / 100.0,
        funding_max_frac=qh.donchian_funding_max_8h_pct / 100.0,
        atr_mult=qh.atr_stop_mult, take_profit_rr=None, max_hold_bars=None)
    res = engine.run(df, asset, "4h", decider=decider)
    return res.metrics.sharpe, res.metrics.n_trades


def run_donchian_letwinners(cfg, engine):
    qh = cfg.quant_hypotheses
    print("\n### DONCHIAN sin techo (tp=None, sin time-stop) — split por mitades ###")
    header = ["entry", "exit_ema", "Sh train (5 act.)", "Sh test (5 act.)", "n1", "n2"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    results = []
    for entry in qh.donchian_entry_period_grid:
        for exit_ema in qh.donchian_exit_ema_grid:
            if exit_ema >= entry:
                continue
            s1s, s2s, n1s, n2s = [], [], [], []
            for asset in ASSETS:
                df = resample(load_parquet(asset, "1h"), "4h")
                mid = len(df) // 2
                s1, n1 = _donchian_sharpe(engine, cfg, df.iloc[:mid], asset, entry, exit_ema)
                s2, n2 = _donchian_sharpe(engine, cfg, df.iloc[mid:], asset, entry, exit_ema)
                s1s.append(s1); s2s.append(s2); n1s.append(n1); n2s.append(n2)
            sh1 = sum(s1s) / len(s1s)
            sh2 = sum(s2s) / len(s2s)
            results.append((entry, exit_ema, sh1, sh2, sum(n1s), sum(n2s)))
    results.sort(key=lambda r: -r[2])  # orden SOLO por train
    for entry, exit_ema, sh1, sh2, n1, n2 in results:
        print(f"| {entry} | {exit_ema} | {sh1:+.2f} | {sh2:+.2f} | {n1} | {n2} |")
    best = results[0]
    print(f"\n>> Elegida EN TRAIN: entry={best[0]} exit_ema={best[1]} "
          f"→ Sharpe train {best[2]:.2f}, test {best[3]:.2f}")
    return results


def _zscore_sharpe(engine, cfg, px_slice, sym, glsr_z_slice, th):
    """Corre el decider z-score sobre UN tramo (train o test) ya recortado y
    devuelve (sharpe, n_trades) del BacktestResult — mismo método que Donchian,
    consistente con el resto del proyecto (nunca un Sharpe hecho a mano)."""
    pr = cfg.positioning_research
    df = px_slice.reset_index()
    closes = df["close"].to_numpy(dtype=float)
    atrs = atr(df, cfg.risk.atr_period).to_numpy()
    decider = make_zscore_decider(
        closes, atrs, glsr_z_slice,
        threshold=th, direction=-1, atr_mult=pr.atr_stop_mult,
        exit_zscore_abs=pr.exit_zscore_abs)
    res = engine.run(df, sym, "1h", decider=decider)
    return res.metrics.sharpe, res.metrics.n_trades


def run_positioning_letwinners(cfg, engine):
    pr = cfg.positioning_research
    split_ts = pd.Timestamp(pr.train_test_split_date)
    print("\n### POSICIONAMIENTO con stop ATR real (tp=None) — glsr_z contrarian, 1h ###")
    header = ["config", "Sh train (5 act.)", "Sh test (5 act.)", "n1", "n2"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    # z-score y precio se calculan sobre la serie COMPLETA (rolling causal) y
    # LUEGO se recorta por fecha — igual que run_positioning.py original: el
    # z en la barra t solo usa datos ≤t, así que recortar después no fuga nada.
    per_symbol = {}
    for sym in cfg.market.symbols:
        px = pos_load_klines_1h(f"data/klines_taker/{sym}.parquet")
        m = pos_load_metrics_1h(f"data/metrics/{sym}_metrics_5m.parquet",
                                px.index, pr.metrics_ffill_limit_bars)
        glsr_z = rolling_zscore(m["count_long_short_ratio"], pr.zscore_window_bars)
        per_symbol[sym] = (px, glsr_z.to_numpy(dtype=float))

    results = []
    for th in pr.entry_threshold_grid:
        s1s, s2s, n1s, n2s = [], [], [], []
        for sym, (px, glsr_z) in per_symbol.items():
            mask_tr = px.index < split_ts
            mask_te = ~mask_tr
            s1, n1 = _zscore_sharpe(engine, cfg, px[mask_tr], sym, glsr_z[mask_tr], th)
            s2, n2 = _zscore_sharpe(engine, cfg, px[mask_te], sym, glsr_z[mask_te], th)
            s1s.append(s1); s2s.append(s2); n1s.append(n1); n2s.append(n2)
        sh1 = sum(s1s) / len(s1s)
        sh2 = sum(s2s) / len(s2s)
        results.append((th, sh1, sh2, sum(n1s), sum(n2s)))
    results.sort(key=lambda r: -r[1])  # orden SOLO por train
    for th, sh1, sh2, n1, n2 in results:
        print(f"| z{th} | {sh1:+.2f} | {sh2:+.2f} | {n1} | {n2} |")
    best = results[0]
    print(f"\n>> Elegida EN TRAIN: z{best[0]} → Sharpe train {best[1]:.2f}, test {best[2]:.2f}")
    return results


def run_donchian_por_activo(cfg, engine):
    """Igual que run_donchian_letwinners, pero SIN promediar entre activos: cada
    fila es un par (config, cripto) específico. Selección SOLO por train, entre
    los 9×5=45 pares — no entre las 9 configs ya promediadas."""
    qh = cfg.quant_hypotheses
    print("\n### DONCHIAN sin techo — POR ACTIVO (sin promediar) ###")
    pares = []
    for entry in qh.donchian_entry_period_grid:
        for exit_ema in qh.donchian_exit_ema_grid:
            if exit_ema >= entry:
                continue
            for asset in ASSETS:
                df = resample(load_parquet(asset, "1h"), "4h")
                mid = len(df) // 2
                s1, n1 = _donchian_sharpe(engine, cfg, df.iloc[:mid], asset, entry, exit_ema)
                s2, n2 = _donchian_sharpe(engine, cfg, df.iloc[mid:], asset, entry, exit_ema)
                pares.append((entry, exit_ema, asset, s1, s2, n1, n2))
    pares.sort(key=lambda r: -r[3])  # orden SOLO por train
    header = ["entry", "exit_ema", "activo", "Sh train", "Sh test", "n1", "n2"]
    print("| " + " | ".join(header) + " | (top 10 por train)")
    print("|" + "---|" * len(header))
    for entry, exit_ema, asset, s1, s2, n1, n2 in pares[:10]:
        print(f"| {entry} | {exit_ema} | {asset} | {s1:+.2f} | {s2:+.2f} | {n1} | {n2} |")
    pos = sum(1 for r in pares if r[4] > 0)
    print(f"Test positivo: {pos}/{len(pares)} pares del universo completo "
          f"(referencia de ruido: cuántos 'aciertan' por puro azar)")
    return pares


def run_positioning_por_activo(cfg, engine):
    """Igual que run_positioning_letwinners, pero SIN promediar: cada fila es
    (umbral, cripto) específico. Selección SOLO por train, entre los 2×5=10 pares."""
    pr = cfg.positioning_research
    split_ts = pd.Timestamp(pr.train_test_split_date)
    print("\n### POSICIONAMIENTO glsr_z contrarian — POR ACTIVO (sin promediar) ###")
    per_symbol = {}
    for sym in cfg.market.symbols:
        px = pos_load_klines_1h(f"data/klines_taker/{sym}.parquet")
        m = pos_load_metrics_1h(f"data/metrics/{sym}_metrics_5m.parquet",
                                px.index, pr.metrics_ffill_limit_bars)
        glsr_z = rolling_zscore(m["count_long_short_ratio"], pr.zscore_window_bars)
        per_symbol[sym] = (px, glsr_z.to_numpy(dtype=float))

    pares = []
    for th in pr.entry_threshold_grid:
        for sym, (px, glsr_z) in per_symbol.items():
            mask_tr = px.index < split_ts
            mask_te = ~mask_tr
            s1, n1 = _zscore_sharpe(engine, cfg, px[mask_tr], sym, glsr_z[mask_tr], th)
            s2, n2 = _zscore_sharpe(engine, cfg, px[mask_te], sym, glsr_z[mask_te], th)
            pares.append((th, sym, s1, s2, n1, n2))
    pares.sort(key=lambda r: -r[2])  # orden SOLO por train
    header = ["z", "activo", "Sh train", "Sh test", "n1", "n2"]
    print("| " + " | ".join(header) + " | (top 10 por train)")
    print("|" + "---|" * len(header))
    for th, sym, s1, s2, n1, n2 in pares[:10]:
        print(f"| z{th} | {sym} | {s1:+.2f} | {s2:+.2f} | {n1} | {n2} |")
    pos = sum(1 for r in pares if r[3] > 0)
    print(f"Test positivo: {pos}/{len(pares)} pares del universo completo "
          f"(referencia de ruido: cuántos 'aciertan' por puro azar)")
    return pares


def main() -> int:
    cfg = load_settings()
    engine = BacktestEngine(cfg)
    print("=" * 84)
    print("\"DEJAR CORRER\" — modelos NUNCA probados con esta estructura (2026-07-08)")
    print("=" * 84)
    run_donchian_letwinners(cfg, engine)
    run_positioning_letwinners(cfg, engine)
    run_donchian_por_activo(cfg, engine)
    run_positioning_por_activo(cfg, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
