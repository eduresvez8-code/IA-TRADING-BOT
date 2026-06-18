"""Tests de los contratos de datos.

Estos tests protegen las invariantes del sistema: si alguien (humano o IA)
relaja una validación en models.py, esto falla y lo delata.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.core.models import (
    Candle,
    Order,
    OrderType,
    PositionSide,
    SentimentScore,
    Side,
    Signal,
    SymbolFilters,
)

NOW = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)


def make_candle(**overrides):
    base = dict(
        symbol="BTCUSDT", timeframe="5m", open_time=NOW,
        open=100.0, high=110.0, low=95.0, close=105.0, volume=1234.5,
    )
    base.update(overrides)
    return Candle(**base)


def test_candle_valida():
    c = make_candle()
    assert c.closed is True
    assert c.open_time.tzinfo is not None


def test_candle_rechaza_timestamp_naive():
    with pytest.raises(ValidationError, match="timezone"):
        make_candle(open_time=datetime(2026, 6, 12, 12, 0))  # sin tzinfo


def test_signal_score_fuera_de_rango():
    with pytest.raises(ValidationError):
        Signal(symbol="BTCUSDT", score=1.5, strategy="ema_cross", timestamp=NOW)


def make_order(**overrides):
    base = dict(
        symbol="BTCUSDT", side=Side.BUY, quantity=0.01,
        entry_price=100_000.0, stop_loss=98_500.0,
        decision_reason="test", created_at=NOW,
    )
    base.update(overrides)
    return Order(**base)


def test_orden_compra_valida():
    o = make_order()
    assert o.stop_loss < o.entry_price


def test_orden_compra_con_stop_por_encima_es_invalida():
    # Un SL por encima de la entrada en una compra no protege nada.
    with pytest.raises(ValidationError, match="stop_loss"):
        make_order(stop_loss=101_000.0)


def test_orden_venta_con_stop_por_debajo_es_invalida():
    with pytest.raises(ValidationError, match="stop_loss"):
        make_order(side=Side.SELL, stop_loss=98_500.0)


def test_orden_sin_stop_loss_no_existe():
    # El campo es obligatorio: no hay órdenes sin SL en este sistema.
    with pytest.raises(ValidationError):
        Order(symbol="BTCUSDT", side=Side.BUY, quantity=0.01,
              entry_price=100_000.0, decision_reason="test", created_at=NOW)


def test_orden_leverage_por_defecto_y_explicito():
    # Por defecto 1 (sin apalancar); en Futuros el Risk Manager lo fija.
    assert make_order().leverage == 1
    assert make_order(leverage=3).leverage == 3


def test_orden_leverage_cero_es_invalido():
    with pytest.raises(ValidationError):
        make_order(leverage=0)


def test_orden_position_side_por_defecto_es_both():
    # Sin especificar (one-way / sin hedge), el cubo es BOTH.
    assert make_order().position_side == PositionSide.BOTH


def test_apertura_long_con_buy_es_valida():
    o = make_order(side=Side.BUY, position_side=PositionSide.LONG)
    assert o.position_side == PositionSide.LONG


def test_apertura_long_con_sell_es_incoherente():
    # Abrir LONG con SELL no es una apertura válida en hedge mode.
    with pytest.raises(ValidationError):
        make_order(side=Side.SELL, position_side=PositionSide.LONG)


def test_apertura_short_con_buy_es_incoherente():
    # make_order usa side=BUY con un SL válido de compra; solo el position_side
    # SHORT lo hace incoherente (aísla el validador de apertura del de SL).
    with pytest.raises(ValidationError, match="position_side SHORT"):
        make_order(position_side=PositionSide.SHORT)


def test_order_type_y_position_side_enums():
    assert OrderType.STOP_MARKET.value == "STOP_MARKET"
    assert PositionSide.SHORT.value == "SHORT"


def test_symbol_filters_coerce_a_decimal():
    # Los strings de exchangeInfo deben quedar como Decimal exacto (no float).
    f = SymbolFilters(symbol="BTCUSDT", tick_size="0.01", step_size="0.0001",
                      min_qty="0.0001", min_notional="5")
    assert f.tick_size == Decimal("0.01")
    assert f.step_size == Decimal("0.0001")
    assert f.min_notional == Decimal("5")


def test_symbol_filters_tick_size_cero_es_invalido():
    # Un tickSize de 0 haría imposible cuantizar el precio (gt=0).
    with pytest.raises(ValidationError):
        SymbolFilters(symbol="BTCUSDT", tick_size="0", step_size="0.0001",
                      min_qty="0", min_notional="5")


def make_sentiment(**overrides):
    base = dict(news_id="n1", symbol_scope=["BTC"], score=0.5, confidence=0.8,
                analyzed_at=NOW)
    base.update(overrides)
    return SentimentScore(**base)


def test_sentiment_event_kind_por_defecto_es_none():
    # Una noticia sin clasificar (o sentimiento de mercado tipo Fear&Greed) no es
    # ni macro ni shock: el default no debe bloquear nada en la confluencia.
    assert make_sentiment().event_kind == "none"


def test_sentiment_event_kind_valores_validos():
    assert make_sentiment(event_kind="scheduled").event_kind == "scheduled"
    assert make_sentiment(event_kind="shock").event_kind == "shock"


def test_sentiment_event_kind_invalido_es_rechazado():
    # El Literal cierra el dominio: un valor libre (typo) muere en la frontera.
    with pytest.raises(ValidationError):
        make_sentiment(event_kind="hack")
