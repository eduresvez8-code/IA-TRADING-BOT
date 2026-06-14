"""Adaptador real del Protocol `FuturesExchange` sobre python-binance (USD-M).

Capa fina de traducción: cada método mapea una operación del Executor a la
llamada de python-binance y devuelve nuestros DTOs. Toda llamada pasa por
`retry_with_backoff` (reintentos solo ante 429/418, respetando Retry-After).

⚠️ Requiere validación en la TESTNET de Binance Futuros: la lógica del Executor
está cubierta por tests contra el fake, pero este mapeo concreto (nombres de
campos de la API) solo se confirma contra el exchange real. Crear el cliente
con `AsyncClient.create(..., testnet=True)`.
"""

from __future__ import annotations

from binance import AsyncClient
from binance.exceptions import BinanceAPIException

from src.core.models import OrderType, PositionSide, Side, SymbolFilters
from src.data.binance_client import retry_with_backoff
from src.execution.exchange import (
    AccountSnapshot,
    ExchangePosition,
    OrderRequest,
    OrderResult,
)

# Código de Binance: "No need to change position side." al fijar un modo ya activo.
_NO_NEED_TO_CHANGE = -4059


class BinanceFuturesExchange:
    """Implementa FuturesExchange contra la API de Futuros USD-M."""

    def __init__(self, client: AsyncClient):
        self.client = client
        self._filters_cache: dict[str, SymbolFilters] = {}

    @classmethod
    async def connect(cls, api_key: str, api_secret: str,
                      testnet: bool = True) -> "BinanceFuturesExchange":
        client = await AsyncClient.create(api_key, api_secret, testnet=testnet)
        return cls(client)

    async def close(self) -> None:
        await self.client.close_connection()

    # ---------- modo de posición ----------

    async def get_position_mode(self) -> bool:
        res = await retry_with_backoff(self.client.futures_get_position_mode)
        return bool(res["dualSidePosition"])

    async def set_position_mode(self, dual: bool) -> None:
        try:
            await retry_with_backoff(lambda: self.client.futures_change_position_mode(
                dualSidePosition="true" if dual else "false"))
        except BinanceAPIException as e:
            if e.code != _NO_NEED_TO_CHANGE:  # ya estaba en el modo pedido → ok
                raise

    # ---------- metadatos del símbolo ----------

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        if not self._filters_cache:
            info = await retry_with_backoff(self.client.futures_exchange_info)
            for s in info["symbols"]:
                self._filters_cache[s["symbol"]] = _parse_filters(s)
        return self._filters_cache[symbol]

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await retry_with_backoff(lambda: self.client.futures_change_leverage(
            symbol=symbol, leverage=leverage))

    # ---------- cuenta ----------

    async def get_account(self) -> AccountSnapshot:
        acct = await retry_with_backoff(self.client.futures_account)
        positions = []
        for p in acct.get("positions", []):
            amt = float(p["positionAmt"])
            if amt == 0.0:
                continue  # solo piernas con tamaño
            positions.append(ExchangePosition(
                symbol=p["symbol"],
                position_side=PositionSide(p["positionSide"]),
                qty=abs(amt),
                entry_price=float(p["entryPrice"]),
                initial_margin=float(p.get("positionInitialMargin", p.get("initialMargin", 0.0))),
                unrealized_pnl=float(p.get("unrealizedProfit", 0.0)),
            ))
        return AccountSnapshot(
            wallet_balance=float(acct["totalWalletBalance"]),
            available_balance=float(acct["availableBalance"]),
            positions=positions,
        )

    # ---------- órdenes ----------

    async def place_order(self, req: OrderRequest) -> OrderResult:
        params: dict = {
            "symbol": req.symbol,
            "side": req.side.value,
            "positionSide": req.position_side.value,
            "type": req.type.value,
        }
        if req.client_order_id:
            params["newClientOrderId"] = req.client_order_id
        if req.type == OrderType.MARKET:
            params["quantity"] = req.quantity
        else:
            # STOP_MARKET / TAKE_PROFIT_MARKET protectoras: disparo + cierre total.
            params["stopPrice"] = req.stop_price
            params["closePosition"] = "true"
            params["workingType"] = req.working_type

        resp = await retry_with_backoff(lambda: self.client.futures_create_order(**params))
        return OrderResult(
            order_id=str(resp["orderId"]),
            symbol=resp["symbol"],
            status=resp["status"],
            side=Side(resp["side"]),
            position_side=PositionSide(resp.get("positionSide", req.position_side.value)),
            type=req.type,
            executed_qty=float(resp.get("executedQty", 0.0)),
            avg_price=float(resp.get("avgPrice", 0.0)),
            client_order_id=resp.get("clientOrderId", req.client_order_id),
        )

    async def cancel_all(self, symbol: str) -> None:
        await retry_with_backoff(lambda: self.client.futures_cancel_all_open_orders(
            symbol=symbol))


def _parse_filters(symbol_info: dict) -> SymbolFilters:
    """Extrae tick/step/min de los filtros de exchangeInfo (Futuros USD-M).

    En futuros el filtro de mínimo es MIN_NOTIONAL con campo `notional` (string),
    a diferencia de spot (`minNotional`).
    """
    by_type = {f["filterType"]: f for f in symbol_info["filters"]}
    price = by_type.get("PRICE_FILTER", {})
    lot = by_type.get("LOT_SIZE", {})
    notional = by_type.get("MIN_NOTIONAL", {})
    return SymbolFilters(
        symbol=symbol_info["symbol"],
        tick_size=price.get("tickSize", "0.01"),
        step_size=lot.get("stepSize", "0.001"),
        min_qty=lot.get("minQty", "0"),
        min_notional=notional.get("notional", "0"),
    )
