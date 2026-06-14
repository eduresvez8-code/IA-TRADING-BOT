"""Traducción de una Order aprobada a las órdenes concretas del exchange.

Es el corazón seguro de la ejecución en hedge mode: una apertura se descompone
en TRES órdenes que comparten el mismo `positionSide` (el cubo) pero invierten
el `side` para las protectoras.

Matriz side × positionSide (hedge mode):
    Abrir LONG  → BUY  + LONG       Cerrar LONG → SELL + LONG
    Abrir SHORT → SELL + SHORT      Cerrar SHORT→ BUY  + SHORT

Funciones puras (sin estado ni I/O): se validan con tests de tabla.
"""

from __future__ import annotations

from typing import Callable
from uuid import uuid4

from src.core.models import Order, OrderType, PositionSide, Side
from src.execution.exchange import OrderRequest


def opposite(side: Side) -> Side:
    """El lado contrario: cerrar una pierna usa el side opuesto al de apertura."""
    return Side.SELL if side == Side.BUY else Side.BUY


def _default_id(tag: str) -> str:
    # Formato compatible con newClientOrderId de Binance (^[\.A-Za-z0-9_-]{1,36}$).
    return f"bot-{tag}-{uuid4().hex[:18]}"


def build_open_requests(
    order: Order,
    *,
    working_type: str,
    id_factory: Callable[[str], str] = _default_id,
) -> list[OrderRequest]:
    """Descompone una apertura en entrada MARKET + SL + TP.

    Las protectoras usan `close_position=True` (cierran la pierna entera, sin
    arrastrar `qty` que pueda desincronizarse por fills parciales) y disparan
    sobre `working_type` (MARK_PRICE por defecto).

    El `id_factory` permite ids deterministas en tests; en producción genera
    uno único por orden (idempotencia ante reintentos).
    """
    opp = opposite(order.side)
    reqs = [
        # Entrada: abre/aumenta la pierna en su positionSide.
        OrderRequest(
            symbol=order.symbol, side=order.side, position_side=order.position_side,
            type=OrderType.MARKET, quantity=order.quantity,
            client_order_id=id_factory("entry"),
        ),
        # Stop-loss: lado opuesto, mismo positionSide, cierra la pierna.
        OrderRequest(
            symbol=order.symbol, side=opp, position_side=order.position_side,
            type=OrderType.STOP_MARKET, stop_price=order.stop_loss,
            close_position=True, working_type=working_type,
            client_order_id=id_factory("sl"),
        ),
    ]
    if order.take_profit is not None:
        reqs.append(
            OrderRequest(
                symbol=order.symbol, side=opp, position_side=order.position_side,
                type=OrderType.TAKE_PROFIT_MARKET, stop_price=order.take_profit,
                close_position=True, working_type=working_type,
                client_order_id=id_factory("tp"),
            )
        )
    return reqs


def build_close_request(
    symbol: str,
    position_side: PositionSide,
    quantity: float,
    *,
    id_factory: Callable[[str], str] = _default_id,
) -> OrderRequest:
    """Orden a mercado que cierra una pierna (salida por señal o flip).

    Cerrar es el side OPUESTO al de apertura sobre el mismo positionSide: una
    pierna LONG se abrió con BUY, se cierra con SELL; y viceversa. En hedge mode
    el cierre lleva la CANTIDAD de la pierna — no `reduceOnly` (que el modo
    rechaza) ni `closePosition` (que es solo para órdenes condicionales).
    """
    side = Side.SELL if position_side == PositionSide.LONG else Side.BUY
    return OrderRequest(
        symbol=symbol, side=side, position_side=position_side,
        type=OrderType.MARKET, quantity=quantity,
        client_order_id=id_factory("close"),
    )
