"""Matriz completa estrategia × cripto (2026-07-08): cierra el último hueco de
desglose por activo que faltaba (MA-cross, funding-extremo, posicionamiento
completo). Reutiliza SOLO deciders y grids ya existentes y verificados — cero
parámetros nuevos (Cero Hardcoding intacto).

    uv run python -m backtest.run_full_matrix

Ya per-activo de rondas anteriores (NO se recalculan aquí, se listan en el
resumen final): TSMOM (run_tsmom_split.py), Donchian y posicionamiento glsr_z
(run_letwinners.py), lead-lag (run_leadlag.py), estacionalidad/día-semana/
RSI-reversión (run_seasonality_reversion.py), pares/VWAP/squeeze/carry
(run_quant_matrix.py, con su propia validación walk-forward de 4 folds).

Protocolo idéntico a siempre: split por mitades (igual que run_ma_split.py y
run_tsmom_split.py), config elegida SOLO por train, test medido una vez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.config import load_settings
from src.quant.indicators import atr
from backtest.engine import BacktestEngine
from backtest.positioning import make_zscore_decider, rolling_zscore, taker_imbalance
from backtest.quant_hypotheses import (
    align_funding_to_bars,
    annualize_funding_pct,
    daily_ma_on_bars,
    make_funding_decider,
    make_macross_decider,
    moving_average,
)
from backtest.run_backtest import load_parquet
from backtest.run_positioning import load_klines_1h as pos_load_klines_1h
from backtest.run_positioning import load_metrics_1h as pos_load_metrics_1h
from backtest.run_quant_hypotheses import ASSETS, load_funding, resample


def run_ma_cross_por_activo(cfg, engine):
    """MA-cross (240 configs) SIN promediar: 240×5=1200 pares (config, cripto)."""
    qh = cfg.quant_hypotheses
    print("\n### CRUCE DE MEDIAS — POR ACTIVO (1200 pares, top 10 por train) ###")
    pares = []
    for tf in qh.ma_cross_timeframes:
        for kind in qh.ma_cross_types:
            for fast_p, slow_p in qh.ma_cross_pairs:
                for asset in ASSETS:
                    df = load_parquet(asset, "1h") if tf == "1h" else resample(load_parquet(asset, "1h"), tf)
                    df = df.reset_index(drop=True)
                    closes = df["close"].to_numpy(dtype=float)
                    atrs = atr(df, cfg.risk.atr_period).to_numpy()
                    fast = moving_average(closes, fast_p, kind)
                    slow = moving_average(closes, slow_p, kind)
                    decider = make_macross_decider(
                        closes, atrs, fast, slow,
                        atr_mult=qh.atr_stop_mult, allow_short=qh.ma_cross_allow_short)
                    mid = len(df) // 2
                    s1 = engine.run(df.iloc[:mid], asset, tf, decider=decider).metrics
                    s2 = engine.run(df.iloc[mid:], asset, tf, decider=decider).metrics
                    pares.append((f"{kind.upper()}{fast_p}/{slow_p}@{tf}", asset,
                                 s1.sharpe, s2.sharpe, s1.n_trades, s2.n_trades))
    pares.sort(key=lambda r: -r[2])  # orden SOLO por train
    header = ["config", "activo", "Sh train", "Sh test", "n1", "n2"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for cfg_name, asset, s1, s2, n1, n2 in pares[:10]:
        print(f"| {cfg_name} | {asset} | {s1:+.2f} | {s2:+.2f} | {n1} | {n2} |")
    pos = sum(1 for r in pares if r[3] > 0)
    print(f"Test positivo: {pos}/{len(pares)} pares (referencia de ruido)")
    return pares


def run_funding_extremo_split(cfg, engine):
    """Funding-extremo: NUNCA tuvo split OOS (solo full-sample). Se construye
    aquí con el mismo protocolo (mitades) y por activo."""
    qh = cfg.quant_hypotheses
    print("\n### FUNDING EXTREMO — split por mitades, por activo ###")
    filas = []
    for asset in ASSETS:
        df = resample(load_parquet(asset, "1h"), "4h").reset_index(drop=True)
        closes = df["close"].to_numpy(dtype=float)
        atrs = atr(df, cfg.risk.atr_period).to_numpy()
        funding_ann = annualize_funding_pct(align_funding_to_bars(df, load_funding(asset)))
        trend_ma = daily_ma_on_bars(df, qh.funding_trend_ma_days)
        decider = make_funding_decider(
            closes, atrs, funding_ann, trend_ma,
            neg_thr=qh.funding_extreme_neg_ann_pct, pos_thr=qh.funding_extreme_pos_ann_pct,
            normal_low=qh.funding_normal_low_ann_pct, normal_high=qh.funding_normal_high_ann_pct,
            atr_mult=qh.atr_stop_mult)
        mid = len(df) // 2
        s1 = engine.run(df.iloc[:mid], asset, "4h", decider=decider).metrics
        s2 = engine.run(df.iloc[mid:], asset, "4h", decider=decider).metrics
        filas.append((asset, s1.sharpe, s2.sharpe, s1.n_trades, s2.n_trades))
    header = ["activo", "Sh train", "Sh test", "n1", "n2"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for asset, s1, s2, n1, n2 in filas:
        print(f"| {asset} | {s1:+.2f} | {s2:+.2f} | {n1} | {n2} |")
    return filas


def run_posicionamiento_completo_por_activo(cfg, engine):
    """Las señales de posicionamiento que faltaban por desglosar por activo
    (glsr_z ya se hizo en run_letwinners.py): doi24_z, tlsr_z, smart_dumb,
    imb_ma4/24/72 — momentum y contrarian, con stop ATR real (tp=None)."""
    pr = cfg.positioning_research
    split_ts = pd.Timestamp(pr.train_test_split_date)
    print("\n### POSICIONAMIENTO (resto de features) — POR ACTIVO, stop ATR real ###")

    per_symbol = {}
    for sym in cfg.market.symbols:
        px = pos_load_klines_1h(f"data/klines_taker/{sym}.parquet")
        m = pos_load_metrics_1h(f"data/metrics/{sym}_metrics_5m.parquet",
                                px.index, pr.metrics_ffill_limit_bars)
        feats = {}
        imb = taker_imbalance(px["volume"], px["taker_buy_base"])
        for k in pr.imbalance_ma_bars_grid:
            feats[f"imb_ma{k}"] = rolling_zscore(imb.rolling(k).mean(), pr.zscore_window_bars)
        feats["doi24_z"] = rolling_zscore(m["sum_open_interest"].pct_change(24), pr.zscore_window_bars)
        feats["tlsr_z"] = rolling_zscore(m["sum_toptrader_long_short_ratio"], pr.zscore_window_bars)
        glsr_z = rolling_zscore(m["count_long_short_ratio"], pr.zscore_window_bars)
        feats["smart_dumb"] = feats["tlsr_z"] - glsr_z
        per_symbol[sym] = (px, {k: v.to_numpy(dtype=float) for k, v in feats.items()})

    pares = []
    fnames = list(next(iter(per_symbol.values()))[1].keys())
    for fname in fnames:
        for direction, dlab in ((1, "mom"), (-1, "con")):
            for th in pr.entry_threshold_grid:
                for sym, (px, feats) in per_symbol.items():
                    z = feats[fname]
                    # np.asarray: funciona tanto si la comparación de un DatetimeIndex
                    # devuelve ndarray como si devolviera Series — nunca falla por tipo.
                    mask_tr = np.asarray(px.index < split_ts)
                    mask_te = ~mask_tr
                    tr_df = px[mask_tr].reset_index()
                    te_df = px[mask_te].reset_index()
                    tr_closes = tr_df["close"].to_numpy(dtype=float)
                    tr_atrs = atr(tr_df, cfg.risk.atr_period).to_numpy()
                    tr_decider = make_zscore_decider(
                        tr_closes, tr_atrs, z[mask_tr], threshold=th,
                        direction=direction, atr_mult=pr.atr_stop_mult,
                        exit_zscore_abs=pr.exit_zscore_abs)
                    te_closes = te_df["close"].to_numpy(dtype=float)
                    te_atrs = atr(te_df, cfg.risk.atr_period).to_numpy()
                    te_decider = make_zscore_decider(
                        te_closes, te_atrs, z[mask_te], threshold=th,
                        direction=direction, atr_mult=pr.atr_stop_mult,
                        exit_zscore_abs=pr.exit_zscore_abs)
                    s1 = engine.run(tr_df, sym, "1h", decider=tr_decider).metrics
                    s2 = engine.run(te_df, sym, "1h", decider=te_decider).metrics
                    pares.append((f"{fname}|{dlab}|z{th}", sym, s1.sharpe, s2.sharpe,
                                 s1.n_trades, s2.n_trades))
    pares.sort(key=lambda r: -r[2])
    header = ["config", "activo", "Sh train", "Sh test", "n1", "n2"]
    print("| " + " | ".join(header) + " | (top 10 por train)")
    print("|" + "---|" * len(header))
    for cfg_name, sym, s1, s2, n1, n2 in pares[:10]:
        print(f"| {cfg_name} | {sym} | {s1:+.2f} | {s2:+.2f} | {n1} | {n2} |")
    pos = sum(1 for r in pares if r[3] > 0)
    print(f"Test positivo: {pos}/{len(pares)} pares (referencia de ruido)")
    return pares


def main() -> int:
    cfg = load_settings()
    engine = BacktestEngine(cfg)
    print("=" * 84)
    print("MATRIZ COMPLETA estrategia × cripto — piezas nuevas (2026-07-08)")
    print("=" * 84)
    run_ma_cross_por_activo(cfg, engine)
    run_funding_extremo_split(cfg, engine)
    run_posicionamiento_completo_por_activo(cfg, engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
