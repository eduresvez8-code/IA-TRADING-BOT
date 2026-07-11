"""Tests de los contratos de datos (src/core/models.py).

Regla del repo: models.py no se modifica sin actualizar este archivo en el
mismo cambio.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.core.models import Action, Candle, Decision, Order, Side, Signal

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


# ---------- Candle ----------

def test_candle_valida():
    c = Candle(symbol="SPY", timeframe="1d", open_time=NOW,
               open=100.0, high=101.0, low=99.0, close=100.5, volume=1e6)
    assert c.closed is True


def test_candle_rechaza_open_time_naive():
    with pytest.raises(ValidationError, match="timezone"):
        Candle(symbol="SPY", timeframe="1d",
               open_time=datetime(2026, 7, 11, 12, 0),  # sin tz
               open=100.0, high=101.0, low=99.0, close=100.5, volume=1e6)


def test_candle_normaliza_a_utc():
    from datetime import timedelta, timezone as tz
    bogota = tz(timedelta(hours=-5))
    c = Candle(symbol="SPY", timeframe="1d",
               open_time=datetime(2026, 7, 11, 7, 0, tzinfo=bogota),
               open=100.0, high=101.0, low=99.0, close=100.5, volume=1e6)
    assert c.open_time.tzinfo == timezone.utc
    assert c.open_time.hour == 12


# ---------- Signal ----------

def test_signal_score_fuera_de_rango_no_pasa():
    with pytest.raises(ValidationError):
        Signal(symbol="SPY", score=1.5, strategy="x", timestamp=NOW)


# ---------- Decision ----------

def test_decision_valida():
    d = Decision(symbol="SPY", action=Action.LONG, score=0.8,
                 size_factor=1.0, reason="tsmom_12m", timestamp=NOW)
    assert d.action == Action.LONG


def test_decision_size_factor_acotado():
    with pytest.raises(ValidationError):
        Decision(symbol="SPY", action=Action.LONG, score=0.8,
                 size_factor=1.5, reason="x", timestamp=NOW)


# ---------- Order ----------

def _order_kwargs(**overrides):
    base = dict(symbol="SPY", side=Side.BUY, quantity=10.0, entry_price=100.0,
                stop_loss=95.0, take_profit=None, decision_reason="test",
                created_at=NOW)
    base.update(overrides)
    return base


def test_order_valida_sin_take_profit():
    o = Order(**_order_kwargs())
    assert o.take_profit is None


def test_order_stop_de_compra_debe_proteger_abajo():
    with pytest.raises(ValidationError, match="menor que entry_price"):
        Order(**_order_kwargs(stop_loss=105.0))


def test_order_stop_de_venta_debe_proteger_arriba():
    with pytest.raises(ValidationError, match="mayor que entry_price"):
        Order(**_order_kwargs(side=Side.SELL, stop_loss=95.0))
    o = Order(**_order_kwargs(side=Side.SELL, stop_loss=105.0))
    assert o.stop_loss == 105.0


def test_order_cantidad_positiva():
    with pytest.raises(ValidationError):
        Order(**_order_kwargs(quantity=0.0))
