"""Exchange de Futuros USD-M falso, en memoria y determinista.

Implementa el Protocol `FuturesExchange` para probar el Executor y correr el
demo sin red. Simula lo esencial para ejercitar la lógica:
    - el modo de posición (y que NO se puede cambiar con posiciones abiertas),
    - fills inmediatos de las MARKET al precio de marca configurado,
    - reserva/liberación de margen inicial (= nocional / leverage),
    - el rechazo del positionSide incoherente con el modo.

NO pretende reproducir PnL, funding ni liquidación: solo el contrato necesario.
"""

from __future__ import annotations

from src.core.models import OrderType, PositionSide, Side, SymbolFilters
from src.execution.exchange import (
    AccountSnapshot,
    ExchangePosition,
    OrderRequest,
    OrderResult,
)


class FakeExchangeError(Exception):
    """Equivalente a un rechazo del exchange (p. ej. -4068, -4061)."""


class FakeFuturesExchange:
    """Exchange en memoria para tests y demo."""

    def __init__(
        self,
        *,
        wallet_balance: float = 10_000.0,
        filters: dict[str, SymbolFilters] | None = None,
        prices: dict[str, float] | None = None,
        dual_mode: bool = False,
    ):
        self.wallet_balance = wallet_balance
        self.available_balance = wallet_balance
        self.dual_mode = dual_mode
        self.filters = filters or {}
        self.prices = prices or {}
        self.leverage: dict[str, int] = {}
        # piernas abiertas, indexadas por (symbol, positionSide)
        self.positions: dict[tuple[str, PositionSide], ExchangePosition] = {}
        # órdenes condicionales en reposo (SL/TP), por símbolo
        self.resting: dict[str, list[OrderResult]] = {}
        self.sent: list[OrderRequest] = []   # historial para auditar en tests
        # piernas que existen pero que get_account aún NO reporta: simula la
        # latencia entre el fill (motor de matching) y el endpoint de cuenta.
        self.hidden: set[tuple[str, PositionSide]] = set()
        self._oid = 0

    def _next_oid(self) -> str:
        self._oid += 1
        return str(self._oid)

    # ---------- modo de posición ----------

    async def get_position_mode(self) -> bool:
        return self.dual_mode

    async def set_position_mode(self, dual: bool) -> None:
        if dual != self.dual_mode and self.positions:
            # Binance rechaza el cambio con posiciones abiertas (error -4068).
            raise FakeExchangeError("no se puede cambiar el modo con posiciones abiertas")
        self.dual_mode = dual

    # ---------- metadatos del símbolo ----------

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        return self.filters[symbol]

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        self.leverage[symbol] = leverage

    # ---------- cuenta ----------

    async def get_account(self) -> AccountSnapshot:
        # Las piernas "hidden" existen pero el endpoint de cuenta aún no las
        # reporta (latencia de visibilidad del fill).
        visible = [p for k, p in self.positions.items() if k not in self.hidden]
        return AccountSnapshot(
            wallet_balance=self.wallet_balance,
            available_balance=self.available_balance,
            positions=visible,
        )

    async def get_open_orders(self, symbol: str) -> list[OrderResult]:
        return list(self.resting.get(symbol, []))

    # ---------- órdenes ----------

    async def place_order(self, req: OrderRequest) -> OrderResult:
        self.sent.append(req)
        # El positionSide debe ser coherente con el modo (Binance: -4061).
        if self.dual_mode and req.position_side == PositionSide.BOTH:
            raise FakeExchangeError("hedge mode exige positionSide LONG/SHORT")
        if not self.dual_mode and req.position_side != PositionSide.BOTH:
            raise FakeExchangeError("one-way mode exige positionSide BOTH")

        price = self.prices.get(req.symbol, 0.0)
        lev = self.leverage.get(req.symbol, 1)
        key = (req.symbol, req.position_side)

        if req.type in (OrderType.MARKET, OrderType.LIMIT):
            # En hedge mode, el side determina abrir vs reducir esa pierna:
            # LONG se abre con BUY (cierra con SELL); SHORT se abre con SELL.
            opening_side = Side.SELL if req.position_side == PositionSide.SHORT else Side.BUY

            # Para LIMIT-IOC: verificar si el límite cubre el precio actual.
            # BUY: queremos pagar ≤ limit → fill si limit ≥ price (mercado).
            # SELL: queremos recibir ≥ limit → fill si limit ≤ price (mercado).
            if req.type == OrderType.LIMIT and req.time_in_force == "IOC":
                workable = (
                    (req.side == Side.BUY and req.price is not None and req.price >= price) or
                    (req.side == Side.SELL and req.price is not None and req.price <= price)
                )
                if not workable:
                    return OrderResult(
                        order_id=self._next_oid(), symbol=req.symbol, status="EXPIRED",
                        side=req.side, position_side=req.position_side, type=req.type,
                        executed_qty=0.0, avg_price=0.0, client_order_id=req.client_order_id,
                    )

            if req.side == opening_side:
                return self._open_or_add(req, price, lev, key)
            return self._reduce_leg(req, price, key)

        # STOP_MARKET / TAKE_PROFIT_MARKET: quedan en reposo (no llenan ahora).
        res = OrderResult(
            order_id=self._next_oid(), symbol=req.symbol, status="NEW",
            side=req.side, position_side=req.position_side, type=req.type,
            executed_qty=0.0, avg_price=0.0, client_order_id=req.client_order_id,
        )
        self.resting.setdefault(req.symbol, []).append(res)
        return res

    def _open_or_add(self, req, price, lev, key) -> OrderResult:
        qty = req.quantity or 0.0
        margin = qty * price / lev
        existing = self.positions.get(key)
        if existing is None:
            self.positions[key] = ExchangePosition(
                symbol=req.symbol, position_side=req.position_side,
                qty=qty, entry_price=price, initial_margin=margin,
            )
        else:
            new_qty = existing.qty + qty
            self.positions[key] = ExchangePosition(
                symbol=req.symbol, position_side=req.position_side, qty=new_qty,
                entry_price=price, initial_margin=existing.initial_margin + margin,
            )
        self.available_balance -= margin
        return OrderResult(
            order_id=self._next_oid(), symbol=req.symbol, status="FILLED",
            side=req.side, position_side=req.position_side, type=req.type,
            executed_qty=qty, avg_price=price, client_order_id=req.client_order_id,
        )

    def _reduce_leg(self, req, price, key) -> OrderResult:
        pos = self.positions.get(key)
        if pos is None:  # nada que reducir
            return OrderResult(
                order_id=self._next_oid(), symbol=req.symbol, status="FILLED",
                side=req.side, position_side=req.position_side, type=req.type,
                executed_qty=0.0, avg_price=price, client_order_id=req.client_order_id,
            )
        close_qty = min(req.quantity if req.quantity is not None else pos.qty, pos.qty)
        frac = close_qty / pos.qty if pos.qty else 1.0
        freed = pos.initial_margin * frac
        remaining = pos.qty - close_qty
        if remaining <= 1e-12:
            self.positions.pop(key, None)
            self.resting.pop(req.symbol, None)  # protectoras ya sin pierna que cuidar
        else:
            self.positions[key] = ExchangePosition(
                symbol=req.symbol, position_side=req.position_side, qty=remaining,
                entry_price=pos.entry_price, initial_margin=pos.initial_margin - freed,
            )
        self.available_balance += freed  # libera el margen proporcional
        return OrderResult(
            order_id=self._next_oid(), symbol=req.symbol, status="FILLED",
            side=req.side, position_side=req.position_side, type=req.type,
            executed_qty=close_qty, avg_price=price, client_order_id=req.client_order_id,
        )

    async def cancel_all(self, symbol: str) -> None:
        self.resting.pop(symbol, None)
