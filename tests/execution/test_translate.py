"""Tests de la traducción Order → OrderRequest (matriz side × positionSide).

Es la pieza de seguridad: si una protectora saliera con el positionSide o el
side equivocado, cerraría la pierna que no toca. Un test por celda de la matriz.
"""

from datetime import datetime, timezone

from src.core.models import Order, OrderType, PositionSide, Side
from src.execution.translate import build_close_request, build_open_requests, opposite

NOW = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)


def make_order(side: Side, position_side: PositionSide, *, tp: float | None) -> Order:
    if side == Side.BUY:  # LONG: SL por debajo, TP por encima
        sl, tpv = 925.0, (1150.0 if tp else None)
    else:                 # SHORT: SL por encima, TP por debajo
        sl, tpv = 1075.0, (850.0 if tp else None)
    return Order(symbol="BTCUSDT", side=side, quantity=1.5, entry_price=1000.0,
                 stop_loss=sl, take_profit=tpv, leverage=3, position_side=position_side,
                 decision_reason="test", created_at=NOW)


def _tagged(tag: str) -> str:
    return tag  # id determinista para auditar en los asserts


def test_opposite():
    assert opposite(Side.BUY) == Side.SELL
    assert opposite(Side.SELL) == Side.BUY


def test_apertura_long_genera_entry_sl_tp():
    order = make_order(Side.BUY, PositionSide.LONG, tp=1150.0)
    reqs = build_open_requests(order, working_type="MARK_PRICE", id_factory=_tagged)
    assert len(reqs) == 3
    entry, sl, tp = reqs

    # Entrada: BUY sobre el cubo LONG, con cantidad.
    assert (entry.side, entry.position_side, entry.type) == (
        Side.BUY, PositionSide.LONG, OrderType.MARKET)
    assert entry.quantity == 1.5 and entry.close_position is False

    # SL/TP: lado OPUESTO (SELL), mismo cubo LONG, cierran la pierna.
    assert (sl.side, sl.position_side, sl.type) == (
        Side.SELL, PositionSide.LONG, OrderType.STOP_MARKET)
    assert sl.stop_price == 925.0 and sl.close_position is True
    assert sl.working_type == "MARK_PRICE" and sl.quantity is None
    assert (tp.side, tp.position_side, tp.type) == (
        Side.SELL, PositionSide.LONG, OrderType.TAKE_PROFIT_MARKET)
    assert tp.stop_price == 1150.0 and tp.close_position is True


def test_apertura_short_es_espejo():
    order = make_order(Side.SELL, PositionSide.SHORT, tp=850.0)
    entry, sl, tp = build_open_requests(order, working_type="MARK_PRICE", id_factory=_tagged)
    assert (entry.side, entry.position_side) == (Side.SELL, PositionSide.SHORT)
    # Protectoras: BUY sobre el cubo SHORT.
    assert (sl.side, sl.position_side, sl.type) == (
        Side.BUY, PositionSide.SHORT, OrderType.STOP_MARKET)
    assert (tp.side, tp.position_side, tp.type) == (
        Side.BUY, PositionSide.SHORT, OrderType.TAKE_PROFIT_MARKET)


def test_sin_take_profit_solo_dos_ordenes():
    order = make_order(Side.BUY, PositionSide.LONG, tp=None)
    reqs = build_open_requests(order, working_type="MARK_PRICE", id_factory=_tagged)
    assert len(reqs) == 2
    assert reqs[1].type == OrderType.STOP_MARKET


def test_ids_provienen_del_factory():
    order = make_order(Side.BUY, PositionSide.LONG, tp=1150.0)
    entry, sl, tp = build_open_requests(order, working_type="MARK_PRICE", id_factory=_tagged)
    assert (entry.client_order_id, sl.client_order_id, tp.client_order_id) == (
        "entry", "sl", "tp")


def test_cierre_long_es_sell_con_cantidad():
    req = build_close_request("BTCUSDT", PositionSide.LONG, 1.5, id_factory=_tagged)
    assert req.side == Side.SELL and req.position_side == PositionSide.LONG
    assert req.type == OrderType.MARKET and req.quantity == 1.5
    assert req.close_position is False  # cierre a mercado lleva qty, no closePosition


def test_cierre_short_es_buy():
    req = build_close_request("BTCUSDT", PositionSide.SHORT, 2.0, id_factory=_tagged)
    assert req.side == Side.BUY and req.position_side == PositionSide.SHORT
    assert req.quantity == 2.0
