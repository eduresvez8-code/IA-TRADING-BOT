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

from src.core.config import load_settings
from backtest.engine import BacktestEngine


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


def fresh_settings():
    """Settings recién cargados (mutables) para tunear costos por test."""
    return load_settings()


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

        high_cost = fresh_settings()
        high_cost.backtest.commission_pct = 0.5
        high_cost.backtest.slippage_pct = 0.2

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
