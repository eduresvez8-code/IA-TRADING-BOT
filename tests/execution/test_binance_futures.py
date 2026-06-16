"""Tests del adaptador real: traducción hedge ↔ one-way en la frontera.

Sin red: un cliente python-binance falso captura los params enviados y devuelve
respuestas canónicas. Verificamos que en one-way el adaptador manda
positionSide=BOTH y reconstruye piernas LONG/SHORT por el signo del positionAmt,
y que el veto -4084 del testnet (no se puede cambiar el modo) no rompe nada.
"""

import pytest
from binance.exceptions import BinanceAPIException

from src.core.models import OrderType, PositionSide, Side
from src.execution.binance_futures import BinanceFuturesExchange
from src.execution.exchange import OrderRequest


class FakeBinanceClient:
    def __init__(self, *, dual=False, positions=None, change_error=None):
        self._dual = dual
        self._positions = positions or []
        self._change_error = change_error
        self.created = []

    async def futures_get_position_mode(self):
        return {"dualSidePosition": self._dual}

    async def futures_change_position_mode(self, dualSidePosition):
        if self._change_error is not None:
            raise self._change_error
        self._dual = dualSidePosition == "true"
        return {}

    async def futures_account(self):
        return {"totalWalletBalance": "1000", "availableBalance": "1000",
                "positions": self._positions}

    async def futures_create_order(self, **params):
        self.created.append(params)
        return {"orderId": 7, "symbol": params["symbol"], "status": "FILLED",
                "side": params["side"], "positionSide": params["positionSide"],
                "executedQty": str(params.get("quantity") or 0), "avgPrice": "100",
                "clientOrderId": params.get("newClientOrderId")}


def _api_error(code: int) -> BinanceAPIException:
    class _R:
        status_code = 400
        text = ""
    return BinanceAPIException(_R(), 400, f'{{"code": {code}, "msg": "x"}}')


def _long_entry() -> OrderRequest:
    return OrderRequest(symbol="BTCUSDT", side=Side.BUY, position_side=PositionSide.LONG,
                        type=OrderType.MARKET, quantity=0.01, client_order_id="bot-entry-1")


async def _adapter(client) -> BinanceFuturesExchange:
    ex = BinanceFuturesExchange(client)
    await ex.get_position_mode()  # cachea el modo real
    return ex


class TestPlaceOrderTranslation:
    async def test_one_way_envia_position_side_both(self):
        client = FakeBinanceClient(dual=False)
        ex = await _adapter(client)
        res = await ex.place_order(_long_entry())
        assert client.created[0]["positionSide"] == "BOTH"
        assert res.position_side == PositionSide.LONG  # el bot mantiene su intención

    async def test_hedge_envia_position_side_long(self):
        client = FakeBinanceClient(dual=True)
        ex = await _adapter(client)
        await ex.place_order(_long_entry())
        assert client.created[0]["positionSide"] == "LONG"


class TestGetAccountMapping:
    async def test_one_way_signo_negativo_es_short(self):
        client = FakeBinanceClient(dual=False, positions=[
            {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "-2.0",
             "entryPrice": "100", "initialMargin": "10", "unrealizedProfit": "0"}])
        ex = await _adapter(client)
        snap = await ex.get_account()
        assert snap.positions[0].position_side == PositionSide.SHORT
        assert snap.positions[0].qty == pytest.approx(2.0)

    async def test_one_way_signo_positivo_es_long(self):
        client = FakeBinanceClient(dual=False, positions=[
            {"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt": "3.0",
             "entryPrice": "100", "initialMargin": "10", "unrealizedProfit": "0"}])
        ex = await _adapter(client)
        snap = await ex.get_account()
        assert snap.positions[0].position_side == PositionSide.LONG
        assert snap.positions[0].qty == pytest.approx(3.0)

    async def test_hedge_passthrough(self):
        client = FakeBinanceClient(dual=True, positions=[
            {"symbol": "BTCUSDT", "positionSide": "SHORT", "positionAmt": "1.5",
             "entryPrice": "100", "initialMargin": "10", "unrealizedProfit": "0"}])
        ex = await _adapter(client)
        snap = await ex.get_account()
        assert snap.positions[0].position_side == PositionSide.SHORT


class TestPositionModeVeto:
    async def test_cambiar_a_hedge_vetado_no_rompe(self):
        # -4084: el testnet no permite cambiar el modo → set_position_mode no lanza.
        client = FakeBinanceClient(dual=False, change_error=_api_error(-4084))
        ex = await _adapter(client)
        await ex.set_position_mode(True)  # no debe lanzar
        assert ex._dual is False  # sigue en one-way (el modo real)

    async def test_error_inesperado_se_propaga(self):
        client = FakeBinanceClient(dual=False, change_error=_api_error(-1234))
        ex = await _adapter(client)
        with pytest.raises(BinanceAPIException):
            await ex.set_position_mode(True)
