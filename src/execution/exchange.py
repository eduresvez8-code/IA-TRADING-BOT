"""Abstracción del exchange de Futuros USD-M.

Define QUÉ necesita el Executor del exchange (un `Protocol`) y los DTOs que
intercambian, sin acoplar la lógica a python-binance. Así el Executor se prueba
contra un fake en memoria (determinista, sin red) y en producción habla con el
adaptador real — sin enterarse de cuál usa.

Estos DTOs son transitorios entre el Executor y el adaptador (ambos en el
paquete `execution`), por eso son dataclasses y no contratos Pydantic de
`core/models.py`: no cruzan la frontera del pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.core.models import OrderType, PositionSide, Side, SymbolFilters


@dataclass(frozen=True)
class OrderRequest:
    """Una orden a enviar al exchange (entrada o protectora)."""

    symbol: str
    side: Side
    position_side: PositionSide
    type: OrderType
    quantity: float | None = None       # None cuando close_position=True
    stop_price: float | None = None     # disparo de STOP_MARKET / TAKE_PROFIT_MARKET
    close_position: bool = False         # cierra la pierna entera (protectoras)
    working_type: str = "MARK_PRICE"     # precio de disparo del stop
    client_order_id: str | None = None   # idempotencia ante reintentos


@dataclass(frozen=True)
class OrderResult:
    """Respuesta del exchange a una orden enviada."""

    order_id: str
    symbol: str
    status: str                          # NEW (resting) | FILLED | ...
    side: Side
    position_side: PositionSide
    type: OrderType
    executed_qty: float
    avg_price: float                     # 0.0 si aún no ha llenado (protectoras)
    client_order_id: str | None = None


@dataclass(frozen=True)
class ExchangePosition:
    """Una pierna abierta tal como la reporta el exchange."""

    symbol: str
    position_side: PositionSide
    qty: float                           # tamaño de la pierna (>0)
    entry_price: float
    initial_margin: float
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class AccountSnapshot:
    """Foto de la cuenta de futuros: saldos + piernas abiertas."""

    wallet_balance: float                # colateral total SIN PnL no realizado
    available_balance: float             # margen libre para abrir
    positions: list[ExchangePosition] = field(default_factory=list)


@runtime_checkable
class FuturesExchange(Protocol):
    """Operaciones de Futuros USD-M que el Executor necesita."""

    async def get_position_mode(self) -> bool:
        """True si la cuenta está en hedge mode (dualSidePosition)."""
        ...

    async def set_position_mode(self, dual: bool) -> None:
        """Impone hedge (True) u one-way (False). Falla con posiciones abiertas."""
        ...

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        """Filtros de microestructura del par (de exchangeInfo)."""
        ...

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Fija el apalancamiento del símbolo."""
        ...

    async def get_account(self) -> AccountSnapshot:
        """Saldos y posiciones actuales."""
        ...

    async def place_order(self, req: OrderRequest) -> OrderResult:
        """Envía una orden y devuelve su resultado."""
        ...

    async def cancel_all(self, symbol: str) -> None:
        """Cancela todas las órdenes abiertas (resting) de un símbolo."""
        ...
