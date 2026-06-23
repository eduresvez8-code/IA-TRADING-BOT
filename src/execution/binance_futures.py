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

import logging

from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from pydantic import ValidationError

from src.core.models import OrderType, PositionSide, Side, SymbolFilters
from src.data.binance_client import retry_with_backoff
from src.execution.exchange import (
    AccountSnapshot,
    ExchangePosition,
    OrderRequest,
    OrderResult,
)

logger = logging.getLogger(__name__)

# Código de Binance: "No need to change position side." al fijar un modo ya activo.
_NO_NEED_TO_CHANGE = -4059
# Código de Binance: "Method is not allowed currently." El testnet de Futuros tiene
# deshabilitado CAMBIAR el modo de posición → no se puede activar hedge; se opera
# en one-way (el adaptador traduce positionSide en la frontera, ver abajo).
_MODE_CHANGE_DISABLED = -4084


class BinanceFuturesExchange:
    """Implementa FuturesExchange contra la API de Futuros USD-M.

    Internamente el bot razona en piernas LONG/SHORT (hedge mode). Si la cuenta
    está en ONE-WAY (p.ej. el testnet, que no permite activar hedge), este
    adaptador traduce en la frontera: las órdenes se envían con positionSide=BOTH
    y las posiciones del exchange (positionSide=BOTH, positionAmt con signo) se
    presentan al bot como piernas LONG/SHORT según el signo. El resto del sistema
    (executor, reconciliación, política) no se entera.
    """

    def __init__(self, client: AsyncClient):
        self.client = client
        self._filters_cache: dict[str, SymbolFilters] = {}
        self._dual: bool = True  # se fija en connect() con el modo real de la cuenta

    @classmethod
    async def connect(cls, api_key: str, api_secret: str,
                      testnet: bool = True) -> "BinanceFuturesExchange":
        client = await AsyncClient.create(api_key, api_secret, testnet=testnet)
        self = cls(client)
        await self.get_position_mode()  # cachea self._dual con el modo real
        return self

    async def close(self) -> None:
        await self.client.close_connection()

    # ---------- modo de posición ----------

    async def get_position_mode(self) -> bool:
        res = await retry_with_backoff(self.client.futures_get_position_mode)
        self._dual = bool(res["dualSidePosition"])
        return self._dual

    async def set_position_mode(self, dual: bool) -> None:
        try:
            await retry_with_backoff(lambda: self.client.futures_change_position_mode(
                dualSidePosition="true" if dual else "false"))
            self._dual = dual
        except BinanceAPIException as e:
            if e.code == _NO_NEED_TO_CHANGE:       # ya estaba en el modo pedido → ok
                self._dual = dual
            elif e.code == _MODE_CHANGE_DISABLED:  # testnet no permite cambiar el modo
                logger.warning(
                    "no se pudo cambiar el modo de posición (code -4084); se opera en "
                    "el modo actual: %s. El adaptador traducirá positionSide.",
                    "hedge" if self._dual else "one-way")
            else:
                raise

    # ---------- metadatos del símbolo ----------

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        if not self._filters_cache:
            info = await retry_with_backoff(self.client.futures_exchange_info)
            for s in info["symbols"]:
                # Algunos símbolos del testnet traen filtros degenerados
                # (p.ej. tickSize='0' en pares delistados) que no validan. Se
                # ignoran: solo nos importan los símbolos que vamos a operar.
                try:
                    self._filters_cache[s["symbol"]] = _parse_filters(s)
                except ValidationError:
                    continue
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
            raw_side = p["positionSide"]
            # One-way: el exchange reporta BOTH con positionAmt CON SIGNO. Lo
            # presentamos al bot como pierna LONG/SHORT según el signo. En hedge
            # el positionSide ya viene LONG/SHORT y el amt es positivo.
            if raw_side == "BOTH":
                side = PositionSide.LONG if amt > 0 else PositionSide.SHORT
            else:
                side = PositionSide(raw_side)
            positions.append(ExchangePosition(
                symbol=p["symbol"],
                position_side=side,
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

    async def get_realized_pnl(self, since_ms: int) -> dict[str, float]:
        """PnL REALIZADO por símbolo desde `since_ms` (income history de Binance).

        Suma el `income` de los registros REALIZED_PNL (el PnL al cerrar/reducir una
        posición; positivo gana, negativo pierde). Es la fuente AUTORITATIVA del
        exchange — no una reconstrucción frágil desde las órdenes. Solo lectura,
        para el panel del dashboard; nunca opera. limit=1000 cubre de sobra una
        sesión de paper trading.
        """
        rows = await retry_with_backoff(
            lambda: self.client.futures_income_history(
                incomeType="REALIZED_PNL", startTime=since_ms, limit=1000)
        )
        agg: dict[str, float] = {}
        for r in rows:
            sym = r.get("symbol")
            if sym:
                agg[sym] = agg.get(sym, 0.0) + float(r.get("income", 0.0))
        return agg

    # ---------- órdenes ----------

    async def place_order(self, req: OrderRequest) -> OrderResult:
        # En one-way el exchange exige positionSide=BOTH; el `side` ya codifica la
        # dirección (BUY abre/cierra reduciendo el neto). En hedge va LONG/SHORT.
        position_side = "BOTH" if not self._dual else req.position_side.value
        params: dict = {
            "symbol": req.symbol,
            "side": req.side.value,
            "positionSide": position_side,
            "type": req.type.value,
        }
        if req.client_order_id:
            params["newClientOrderId"] = req.client_order_id
        if req.type == OrderType.MARKET:
            params["quantity"] = req.quantity
        elif req.type == OrderType.LIMIT:
            # LIMIT-IOC marketable: cantidad + precio límite + timeInForce.
            params["quantity"] = req.quantity
            params["price"] = req.price
            if req.time_in_force:
                params["timeInForce"] = req.time_in_force
        else:
            # STOP_MARKET / TAKE_PROFIT_MARKET protectoras: disparo + cierre total.
            params["stopPrice"] = req.stop_price
            params["closePosition"] = "true"
            params["workingType"] = req.working_type

        resp = await retry_with_backoff(lambda: self.client.futures_create_order(**params))
        # Binance enruta las órdenes condicionales con closePosition como órdenes
        # ALGO: el esquema de respuesta cambia (algoId/algoStatus en vez de
        # orderId/status). Parseamos defensivamente ambos esquemas.
        return OrderResult(
            order_id=str(resp.get("orderId") or resp.get("algoId", "")),
            symbol=resp["symbol"],
            status=resp.get("status") or resp.get("algoStatus", "NEW"),
            side=Side(resp["side"]),
            # El bot razona en su pierna pretendida (LONG/SHORT), aunque en one-way
            # el exchange devuelva BOTH. Reflejamos la intención del Order.
            position_side=req.position_side,
            type=req.type,
            executed_qty=float(resp.get("executedQty", 0.0) or 0.0),
            avg_price=float(resp.get("avgPrice", 0.0) or 0.0),
            client_order_id=(resp.get("clientOrderId") or resp.get("clientAlgoId")
                             or req.client_order_id),
        )

    async def get_open_orders(self, symbol: str) -> list[OrderResult]:
        raw = await retry_with_backoff(lambda: self.client.futures_get_open_orders(
            symbol=symbol))
        out = []
        for o in raw:
            out.append(OrderResult(
                order_id=str(o["orderId"]), symbol=o["symbol"], status=o["status"],
                side=Side(o["side"]), position_side=PositionSide(o.get("positionSide", "BOTH")),
                type=OrderType(o["type"]) if o["type"] in OrderType._value2member_map_ else OrderType.MARKET,
                executed_qty=float(o.get("executedQty", 0.0)),
                avg_price=float(o.get("avgPrice", 0.0)),
                client_order_id=o.get("clientOrderId"),
            ))
        return out

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
