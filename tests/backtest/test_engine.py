"""Tests del motor de backtesting.

No verificamos un PnL exacto (depende de la estrategia), sino las INVARIANTES
que hacen honesto a un backtester:
    - los costos reducen el resultado,
    - el stop-loss acota la pérdida,
    - el take-profit cierra en ganancia,
    - la contabilidad cuadra (equity final = inicial + suma de PnL netos),
    - sin señal no hay trades.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.config import QuantConfig, load_settings
from backtest.engine import BacktestEngine


def _ema_quant() -> QuantConfig:
    """Quant EMA 9/21/14 — la señal para la que se diseñaron estos tests (datos
    cortos, warmup 35 velas). Se fija explícitamente para NO depender de lo que
    haya en settings.yaml (que ahora envía SMA 50/200, warmup 214)."""
    return QuantConfig(ma_type="ema", ema_fast_period=9, ema_slow_period=21,
                       rsi_period=14, ema_weight=0.6)


def make_ohlc(closes: list[float], spread: float = 0.5) -> pd.DataFrame:
    """DataFrame OHLCV con open_time UTC a partir de una lista de cierres."""
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    opens = np.concatenate([[c[0]], c[:-1]])  # apertura = cierre previo
    return pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
        "open": opens,
        "high": c + spread,
        "low": c - spread,
        "close": c,
        "volume": np.full(n, 1000.0),
    })


def make_ohlc_ex(opens, highs, lows, closes) -> pd.DataFrame:
    """OHLCV con CADA columna explícita.

    A diferencia de make_ohlc (que fuerza open = cierre previo y por tanto hace
    los gaps imposibles), aquí el `open` es libre: así podemos construir velas
    que ABREN ya cruzadas respecto al stop/TP y ejercitar la ejecución en GAP.
    """
    n = len(closes)
    return pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
        "open": np.asarray(opens, dtype=float),
        "high": np.asarray(highs, dtype=float),
        "low": np.asarray(lows, dtype=float),
        "close": np.asarray(closes, dtype=float),
        "volume": np.full(n, 1000.0),
    })


def _trend(start: float, step: float, n: int = 120):
    """Tendencia lineal limpia (open = cierre previo) como listas OHLC."""
    closes = [start + step * i for i in range(n)]
    opens = [closes[0]] + closes[:-1]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return opens, highs, lows, closes


def fresh_settings():
    """Settings recién cargados (mutables) para tunear costos por test, con el quant
    fijado a EMA 9/21/14 (independiente del settings.yaml enviado)."""
    s = load_settings()
    s.quant = _ema_quant()
    return s


class TestNoSignalNoTrades:
    def test_flat_market_no_trades(self):
        df = make_ohlc([100.0] * 200)
        res = BacktestEngine(fresh_settings()).run(df, "BTCUSDT", "5m")
        assert res.metrics.n_trades == 0
        assert res.final_equity == pytest.approx(res.initial_capital)
        assert res.metrics.max_drawdown == 0.0


class TestEntriesHappen:
    def test_uptrend_opens_long(self):
        df = make_ohlc([100.0 + i * 0.5 for i in range(160)])
        res = BacktestEngine(fresh_settings()).run(df, "BTCUSDT", "5m")
        assert res.metrics.n_trades >= 1
        assert any(t.side == "LONG" for t in res.trades)

    def test_downtrend_opens_short_when_allowed(self):
        cfg = fresh_settings()
        cfg.backtest.allow_short = True
        df = make_ohlc([200.0 - i * 0.5 for i in range(160)])
        res = BacktestEngine(cfg).run(df, "ETHUSDT", "5m")
        assert any(t.side == "SHORT" for t in res.trades)

    def test_long_only_skips_shorts(self):
        cfg = fresh_settings()
        cfg.backtest.allow_short = False
        df = make_ohlc([200.0 - i * 0.5 for i in range(160)])
        res = BacktestEngine(cfg).run(df, "ETHUSDT", "5m")
        assert all(t.side != "SHORT" for t in res.trades)


class TestCostsMatter:
    def test_fees_reduce_final_equity(self):
        closes = [100.0 + i * 0.5 for i in range(160)]
        df = make_ohlc(closes)

        no_cost = fresh_settings()
        no_cost.backtest.commission_pct = 0.0
        no_cost.backtest.slippage_pct = 0.0
        no_cost.backtest.slippage_atr_multiplier = 0.0  # regresión: slippage puramente fijo

        high_cost = fresh_settings()
        high_cost.backtest.commission_pct = 0.5
        high_cost.backtest.slippage_pct = 0.2
        high_cost.backtest.slippage_atr_multiplier = 0.0

        eq_free = BacktestEngine(no_cost).run(df, "BTCUSDT", "5m").final_equity
        eq_costly = BacktestEngine(high_cost).run(df, "BTCUSDT", "5m").final_equity
        assert eq_free > eq_costly


class TestStopLoss:
    def test_crash_triggers_stop_loss(self):
        # Subida sostenida (abre LONG) y luego un desplome brutal en una vela.
        closes = [100.0 + i * 0.5 for i in range(120)]
        closes.append(50.0)  # crash: el low de esta vela perfora el stop
        df = make_ohlc(closes)
        res = BacktestEngine(fresh_settings()).run(df, "BTCUSDT", "5m")
        assert any(t.exit_reason == "stop_loss" for t in res.trades)
        # El stop acota la pérdida: ningún trade pierde mucho más que el riesgo.
        sl_trades = [t for t in res.trades if t.exit_reason == "stop_loss"]
        assert all(t.net_pnl < 0 for t in sl_trades)


class TestTakeProfit:
    def test_jump_triggers_take_profit(self):
        closes = [100.0 + i * 0.5 for i in range(80)]
        closes.append(300.0)  # salto al alza: el high alcanza el take-profit
        df = make_ohlc(closes)
        res = BacktestEngine(fresh_settings()).run(df, "BTCUSDT", "5m")
        assert any(t.exit_reason == "take_profit" for t in res.trades)


class TestAccounting:
    def test_equity_equals_initial_plus_pnl(self):
        df = make_ohlc([100.0 + i * 0.5 for i in range(200)])
        res = BacktestEngine(fresh_settings()).run(df, "BTCUSDT", "5m")
        total_pnl = sum(t.net_pnl for t in res.trades)
        assert res.final_equity == pytest.approx(res.initial_capital + total_pnl)

    def test_equity_curve_length_matches_candles(self):
        df = make_ohlc([100.0 + i * 0.5 for i in range(150)])
        res = BacktestEngine(fresh_settings()).run(df, "BTCUSDT", "5m")
        assert len(res.equity_curve) == len(df)

    def test_no_position_open_at_end(self):
        # Tras el cierre forzado, la equity final es realizada (cash), no MTM.
        df = make_ohlc([100.0 + i * 0.5 for i in range(150)])
        res = BacktestEngine(fresh_settings()).run(df, "BTCUSDT", "5m")
        assert res.final_equity == pytest.approx(res.equity_curve.iloc[-1])


class TestGapExecution:
    """Caracterización: rellenar al open cuando la vela abre cruzada.

    Cada test compara dos escenarios IDÉNTICOS salvo la apertura de la vela de
    salida (gap vs toque intrabar). Como la entrada es la misma, aísla el efecto
    de la ejecución en gap.
    """

    def test_gap_down_long_fills_worse_than_intrabar_touch(self):
        """LONG, gap EN CONTRA: abrir bajo el stop pierde ≥ que tocarlo intrabar."""
        cfg = fresh_settings()
        cfg.backtest.take_profit_rr = 1000.0         # TP inalcanzable → aislar el stop
        cfg.backtest.slippage_atr_multiplier = 0.0   # aislar el gap, no el slippage dinámico
        o, h, l, c = _trend(100.0, 0.5, 120)
        # NO-GAP: la vela de salida abre arriba (cierre previo); el low perfora el stop.
        ng = make_ohlc_ex(o + [c[-1]], h + [c[-1] + 0.5], l + [10.0], c + [10.0])
        # GAP: la vela de salida ABRE ya por debajo del stop.
        g = make_ohlc_ex(o + [10.0], h + [10.5], l + [9.5], c + [10.0])
        res_ng = BacktestEngine(cfg).run(ng, "BTCUSDT", "5m")
        res_g = BacktestEngine(cfg).run(g, "BTCUSDT", "5m")
        sl_ng = [t for t in res_ng.trades if t.exit_reason == "stop_loss"]
        sl_g = [t for t in res_g.trades if t.exit_reason == "stop_loss"]
        assert sl_ng and sl_g
        assert sl_g[-1].net_pnl <= sl_ng[-1].net_pnl      # nueva pérdida ≥ anterior
        assert sl_g[-1].exit_price < sl_ng[-1].exit_price  # fill peor (más bajo)

    def test_gap_up_short_fills_worse_than_intrabar_touch(self):
        """SHORT, gap EN CONTRA: abrir sobre el stop pierde ≥ que tocarlo intrabar."""
        cfg = fresh_settings()
        cfg.backtest.allow_short = True
        cfg.backtest.take_profit_rr = 1000.0
        cfg.backtest.slippage_atr_multiplier = 0.0
        o, h, l, c = _trend(200.0, -0.5, 120)
        # NO-GAP: abre abajo (cierre previo); el high perfora el stop intrabar.
        ng = make_ohlc_ex(o + [c[-1]], h + [300.0], l + [c[-1] - 0.5], c + [300.0])
        # GAP: abre ya por encima del stop.
        g = make_ohlc_ex(o + [300.0], h + [300.5], l + [299.5], c + [300.0])
        res_ng = BacktestEngine(cfg).run(ng, "ETHUSDT", "5m")
        res_g = BacktestEngine(cfg).run(g, "ETHUSDT", "5m")
        sl_ng = [t for t in res_ng.trades if t.exit_reason == "stop_loss"]
        sl_g = [t for t in res_g.trades if t.exit_reason == "stop_loss"]
        assert sl_ng and sl_g
        assert sl_g[-1].net_pnl <= sl_ng[-1].net_pnl
        assert sl_g[-1].exit_price > sl_ng[-1].exit_price  # short: peor fill = más alto

    def test_gap_up_long_take_profit_fills_better_than_touch(self):
        """LONG, gap A FAVOR: abrir sobre el TP gana ≥ que alcanzarlo intrabar."""
        cfg = fresh_settings()
        cfg.backtest.take_profit_rr = 50.0           # TP lejano: solo lo alcanza la vela inyectada
        cfg.backtest.slippage_atr_multiplier = 0.0
        o, h, l, c = _trend(100.0, 0.5, 120)
        # NO-GAP: abre por debajo del TP; el high lo alcanza intrabar.
        ng = make_ohlc_ex(o + [c[-1]], h + [400.0], l + [c[-1] - 0.5], c + [400.0])
        # GAP a favor: abre por encima del TP.
        g = make_ohlc_ex(o + [400.0], h + [400.5], l + [399.5], c + [400.0])
        res_ng = BacktestEngine(cfg).run(ng, "BTCUSDT", "5m")
        res_g = BacktestEngine(cfg).run(g, "BTCUSDT", "5m")
        tp_ng = [t for t in res_ng.trades if t.exit_reason == "take_profit"]
        tp_g = [t for t in res_g.trades if t.exit_reason == "take_profit"]
        assert tp_ng and tp_g
        assert tp_g[-1].net_pnl >= tp_ng[-1].net_pnl       # gap a favor: ganancia ≥
        assert tp_g[-1].exit_price > tp_ng[-1].exit_price   # fill mejor (más alto)


class TestDynamicSlippage:
    def test_zero_slippage_entry_at_open_exactly(self):
        """Regresión: k=0 y fijo=0 ⇒ el fill de entrada es EXACTAMENTE el open.

        Si el término ATR se colara aun con k=0, entry_price ≠ open y este test
        lo atraparía.
        """
        df = make_ohlc([100.0 + i * 0.5 for i in range(160)])
        cfg = fresh_settings()
        cfg.backtest.slippage_pct = 0.0
        cfg.backtest.slippage_atr_multiplier = 0.0
        res = BacktestEngine(cfg).run(df, "BTCUSDT", "5m")
        t = res.trades[0]
        open_at_entry = df.loc[df["open_time"] == t.entry_time, "open"].iloc[0]
        assert t.entry_price == pytest.approx(open_at_entry)

    def test_higher_multiplier_costs_more(self):
        """k>0 añade slippage proporcional a la volatilidad → menos equity final."""
        df = make_ohlc([100.0 + i * 0.5 for i in range(160)])
        base = fresh_settings()
        base.backtest.slippage_atr_multiplier = 0.0
        dyn = fresh_settings()
        dyn.backtest.slippage_atr_multiplier = 1.0
        eq_base = BacktestEngine(base).run(df, "BTCUSDT", "5m").final_equity
        eq_dyn = BacktestEngine(dyn).run(df, "BTCUSDT", "5m").final_equity
        assert eq_dyn < eq_base
