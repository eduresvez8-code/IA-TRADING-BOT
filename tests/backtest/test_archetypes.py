"""Tests de los tres arquetipos: reglas de entrada/salida sobre datos sintéticos.

Construimos series con propiedades conocidas (caída fuerte tras un tramo plano,
ruptura con/ sin volumen) para fijar el comportamiento de cada decider sin
depender de datos de mercado.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.config import QuantConfig, load_settings
from backtest.engine import BacktestEngine
from backtest.archetypes import (
    make_breakout_decider,
    make_decider,
    make_mean_reversion_decider,
)


def _ema_cfg():
    """Config con quant EMA 9/21/14 (warmup 35). El motor gatea el decider si el
    score es NaN; con el SMA 50/200 del settings.yaml enviado (warmup 214) los datos
    sintéticos cortos no calentarían. Fijar EMA lo desacopla de la config enviada."""
    s = load_settings()
    s.quant = QuantConfig(ma_type="ema", ema_fast_period=9, ema_slow_period=21,
                          rsi_period=14, ema_weight=0.6)
    return s


def make_ohlcv(closes, *, highs=None, lows=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    closes = [float(c) for c in closes]
    opens = [closes[0]] + closes[:-1]
    highs = highs or [c + 0.2 for c in closes]
    lows = lows or [c - 0.2 for c in closes]
    volumes = volumes or [100.0] * n
    return pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
    })


class TestMeanReversion:
    def test_entra_long_en_banda_inferior_con_rsi_bajo(self):
        cfg = _ema_cfg()
        closes = [100.0] * 25 + [80.0]  # plano y luego caída fuerte
        df = make_ohlcv(closes)
        decide = make_mean_reversion_decider(df, cfg, allow_short=True)
        out = decide(len(closes) - 1, None, 0.0, df["open_time"].iloc[-1])
        assert out is not None and out[0] == "enter" and out[1] == "LONG"
        assert out[4] is None                 # sin TP fijo: salida en la media
        assert out[3] < closes[-1]            # stop protector por debajo del precio

    def test_sale_en_la_media_central(self):
        cfg = _ema_cfg()
        closes = [100.0] * 25 + [110.0]       # rebote por encima de la media
        df = make_ohlcv(closes)
        decide = make_mean_reversion_decider(df, cfg, allow_short=True)
        out = decide(len(closes) - 1, "LONG", 0.0, df["open_time"].iloc[-1])
        assert out == ("exit",)

    def test_warmup_devuelve_none(self):
        cfg = _ema_cfg()
        df = make_ohlcv([100.0] * 5)          # menos velas que bb_period
        decide = make_mean_reversion_decider(df, cfg, allow_short=True)
        assert decide(4, None, 0.0, df["open_time"].iloc[-1]) is None


class TestBreakout:
    def _df_breakout(self, volume_spike: float):
        # 40 velas planas (warmup del filtro de ATR: atr(14)+media(20)) y ruptura.
        closes = [100.0] * 40 + [110.0] + [111.0] * 5
        volumes = [100.0] * 40 + [volume_spike] + [100.0] * 5
        return make_ohlcv(closes, volumes=volumes)

    def test_entra_long_al_romper_con_volumen_y_volatilidad(self):
        cfg = _ema_cfg()
        df = self._df_breakout(volume_spike=500.0)
        decider = make_breakout_decider(df, cfg, allow_short=True)
        res = BacktestEngine(cfg).run(df, "TEST", "1h", decider=decider)
        assert res.metrics.n_trades >= 1
        assert res.trades[0].side == "LONG"
        assert res.trades[0].entry_price > 100.0  # entra en la ruptura, no en el plano

    def test_ruptura_sin_volumen_no_entra(self):
        cfg = _ema_cfg()
        df = self._df_breakout(volume_spike=100.0)  # sin spike → filtro de volumen veta
        decider = make_breakout_decider(df, cfg, allow_short=True)
        res = BacktestEngine(cfg).run(df, "TEST", "1h", decider=decider)
        assert res.metrics.n_trades == 0

    def test_salida_trailing_por_canal_opuesto(self):
        cfg = _ema_cfg()
        # sube de 100 a 130 (31 velas) y luego cae por debajo del canal de salida.
        closes = [100.0 + i for i in range(31)] + [115.0]
        df = make_ohlcv(closes)
        decide = make_breakout_decider(df, cfg, allow_short=True)
        out = decide(len(closes) - 1, "LONG", 0.0, df["open_time"].iloc[-1])
        assert out == ("exit",)  # 115 rompe a la baja el mínimo del canal corto previo


class TestDispatch:
    def test_trend_usa_decider_del_motor(self):
        cfg = _ema_cfg()
        df = make_ohlcv([100.0] * 30)
        assert make_decider("trend", df, cfg, allow_short=True) is None

    def test_arquetipo_desconocido_lanza(self):
        cfg = _ema_cfg()
        df = make_ohlcv([100.0] * 30)
        with pytest.raises(ValueError):
            make_decider("momentum_lunar", df, cfg, allow_short=True)
