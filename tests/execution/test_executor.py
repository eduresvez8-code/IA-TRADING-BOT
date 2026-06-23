"""Tests del Executor contra el exchange falso (Futuros USD-M, sin red).

Cubren: imposición de hedge mode al arrancar (y su negativa segura), apertura
(entrada + SL + TP), reconciliación, cierre/flip, snapshot del PortfolioState
(con pico y arranque de día) y el log auditado en SQLite.
"""

from datetime import datetime, timezone

import pytest

from src.core.config import load_settings
from src.core.models import Order, PositionSide, Side, SymbolFilters
from src.data.storage import Storage
from src.execution.exchange import ExchangePosition
from src.execution.executor import Executor, ExecutionStartupError
from src.execution.fake_exchange import FakeFuturesExchange

# Scope a los símbolos que el fake conoce (filtros/precios abajo): estos tests son
# del Executor sobre BTC/ETH, no del universo de producción. Desacopla de settings.yaml.
CFG = load_settings().model_copy(deep=True)
CFG.market.symbols = ["BTCUSDT", "ETHUSDT"]
NOW = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
LEV = CFG.risk.max_leverage  # 3

FILTERS = {
    "BTCUSDT": SymbolFilters(symbol="BTCUSDT", tick_size="0.1", step_size="0.001",
                             min_qty="0.001", min_notional="5"),
    "ETHUSDT": SymbolFilters(symbol="ETHUSDT", tick_size="0.01", step_size="0.001",
                             min_qty="0.001", min_notional="5"),
}


def make_fake(dual_mode: bool = False) -> FakeFuturesExchange:
    return FakeFuturesExchange(
        wallet_balance=10_000.0, filters=FILTERS,
        prices={"BTCUSDT": 1000.0, "ETHUSDT": 2000.0}, dual_mode=dual_mode,
    )


def counter_ids():
    n = 0

    def factory(tag: str) -> str:
        nonlocal n
        n += 1
        return f"{tag}-{n}"

    return factory


def long_order(qty: float = 1.0, *, tp: float | None = 1150.0) -> Order:
    return Order(symbol="BTCUSDT", side=Side.BUY, quantity=qty, entry_price=1000.0,
                 stop_loss=925.0, take_profit=tp, leverage=LEV,
                 position_side=PositionSide.LONG, decision_reason="test", created_at=NOW)


def short_order(qty: float = 1.0) -> Order:
    return Order(symbol="BTCUSDT", side=Side.SELL, quantity=qty, entry_price=1000.0,
                 stop_loss=1075.0, take_profit=850.0, leverage=LEV,
                 position_side=PositionSide.SHORT, decision_reason="test", created_at=NOW)


# ------------------------------ arranque ------------------------------

async def test_startup_impone_hedge_mode():
    ex = make_fake(dual_mode=False)
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    assert ex.dual_mode is True
    assert ex.leverage["BTCUSDT"] == LEV and ex.leverage["ETHUSDT"] == LEV
    assert "BTCUSDT" in execu.filters


async def test_startup_respeta_hedge_ya_activo():
    ex = make_fake(dual_mode=True)
    await Executor(ex, CFG, id_factory=counter_ids()).startup()
    assert ex.dual_mode is True


async def test_startup_rechaza_one_way_con_posiciones():
    # No se puede imponer hedge con posiciones abiertas, y NO se cierran a ciegas.
    ex = make_fake(dual_mode=False)
    ex.positions[("BTCUSDT", PositionSide.LONG)] = ExchangePosition(
        symbol="BTCUSDT", position_side=PositionSide.LONG, qty=1.0,
        entry_price=1000.0, initial_margin=333.0)
    with pytest.raises(ExecutionStartupError):
        await Executor(ex, CFG, id_factory=counter_ids()).startup()


# ------------------------------ apertura ------------------------------

async def test_open_position_coloca_entry_sl_tp():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    report = await execu.open_position(long_order(qty=1.0))

    assert report.ok is True
    assert report.entry.status == "FILLED" and report.entry.side == Side.BUY
    assert len(report.protective) == 2  # SL + TP
    pos = ex.positions[("BTCUSDT", PositionSide.LONG)]
    assert pos.qty == pytest.approx(1.0)
    # margen = nocional/leverage = 1000/3; el available baja en esa cuantía.
    assert ex.available_balance == pytest.approx(10_000.0 - 1000.0 / LEV)
    assert len(ex.resting["BTCUSDT"]) == 2


async def test_open_sin_take_profit_solo_un_protector():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    report = await execu.open_position(long_order(tp=None))
    assert len(report.protective) == 1


# ------------------------------ reconciliación ------------------------------

async def test_reconcile_consistente():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    await execu.open_position(long_order(qty=1.0))
    recon = await execu.reconcile([("BTCUSDT", PositionSide.LONG, 1.0)])
    assert recon.consistent is True and recon.discrepancies == []


async def test_reconcile_detecta_desincronizacion():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    await execu.open_position(long_order(qty=1.0))
    # Esperamos 5.0 pero el exchange tiene 1.0 → discrepancia → halt (cb c).
    recon = await execu.reconcile([("BTCUSDT", PositionSide.LONG, 5.0)])
    assert recon.consistent is False
    assert recon.discrepancies[0].expected_qty == 5.0
    assert recon.discrepancies[0].actual_qty == pytest.approx(1.0)


# ------------------------------ cierre / flip ------------------------------

async def test_close_position_libera_margen_y_cancela_protectoras():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    await execu.open_position(long_order(qty=1.0))
    res = await execu.close_position("BTCUSDT", PositionSide.LONG)
    assert res.status == "FILLED" and res.executed_qty == pytest.approx(1.0)
    assert ("BTCUSDT", PositionSide.LONG) not in ex.positions
    assert ex.available_balance == pytest.approx(10_000.0)  # margen liberado
    assert "BTCUSDT" not in ex.resting


async def test_close_inexistente_devuelve_none():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    assert await execu.close_position("BTCUSDT", PositionSide.LONG) is None


async def test_flip_long_a_short():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    await execu.open_position(long_order(qty=1.0))
    await execu.close_position("BTCUSDT", PositionSide.LONG)
    await execu.open_position(short_order(qty=1.0))
    assert ("BTCUSDT", PositionSide.LONG) not in ex.positions
    assert ("BTCUSDT", PositionSide.SHORT) in ex.positions


# ------------------------------ snapshot del PortfolioState ------------------------------

async def test_snapshot_portfolio_refleja_la_cuenta():
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    await execu.open_position(long_order(qty=1.0))
    state = await execu.snapshot_portfolio(now=NOW)
    assert state.wallet_balance == pytest.approx(10_000.0)
    assert state.committed_margin == pytest.approx(1000.0 / LEV)
    assert state.available_balance == pytest.approx(10_000.0 - 1000.0 / LEV)
    assert state.open_positions == 1


async def test_snapshot_lleva_pico_y_arranque_de_dia():
    ex = make_fake(dual_mode=True)
    execu = Executor(ex, CFG, id_factory=counter_ids())
    d1 = datetime(2026, 6, 14, 10, 0, tzinfo=timezone.utc)

    ex.wallet_balance = 10_000.0
    s = await execu.snapshot_portfolio(now=d1)
    assert s.peak_wallet_balance == 10_000.0 and s.day_start_wallet_balance == 10_000.0

    ex.wallet_balance = 11_000.0  # nuevo máximo el mismo día
    s = await execu.snapshot_portfolio(now=d1)
    assert s.peak_wallet_balance == 11_000.0 and s.day_start_wallet_balance == 10_000.0

    ex.wallet_balance = 9_000.0   # día siguiente → reinicia el arranque de día
    d2 = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
    s = await execu.snapshot_portfolio(now=d2)
    assert s.day_start_wallet_balance == 9_000.0
    assert s.peak_wallet_balance == 11_000.0  # el pico NO se reinicia


# ------------------------------ LIMIT-IOC (Fase 1.3) ------------------------------


async def test_open_position_market_legacy_sin_mark_price():
    # Sin mark_price → MARKET clásico; backward-compat de todas las paths existentes.
    ex = make_fake()
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    report = await execu.open_position(long_order(qty=1.0))
    assert report.ok is True
    assert report.entry.status == "FILLED"
    sent = ex.sent[0]
    from src.core.models import OrderType
    assert sent.type == OrderType.MARKET
    assert sent.price is None


async def test_open_position_limit_ioc_llena_dentro_de_banda():
    # mark=1000, cap=10bps → BUY limit = 1000 × 1.001 = 1001.0 (fake price=1000 < 1001).
    # El IOC llena; confirmed_qty refleja la cantidad real (1.0 para fill completo).
    ex = make_fake()  # prices BTCUSDT = 1000.0
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    report = await execu.open_position(long_order(qty=1.0), mark_price=1000.0)
    assert report.ok is True
    assert report.confirmed_qty == pytest.approx(1.0)
    from src.core.models import OrderType
    entry_req = next(r for r in ex.sent if r.type == OrderType.LIMIT)
    assert entry_req.time_in_force == "IOC"
    assert entry_req.price > 1000.0  # BUY limit debe estar POR ENCIMA del mark


async def test_open_position_ioc_expirado_fuera_de_banda():
    # mark=1000, cap=10bps → BUY limit = 1001.0; pero el exchange tiene price=1005
    # (fuera de la banda de 0.1%): la orden expira sin fill → ok=False.
    ex = FakeFuturesExchange(wallet_balance=10_000.0, filters=FILTERS,
                             prices={"BTCUSDT": 1005.0, "ETHUSDT": 2000.0})
    execu = Executor(ex, CFG, id_factory=counter_ids())
    await execu.startup()
    report = await execu.open_position(long_order(qty=1.0), mark_price=1000.0)
    assert report.ok is False
    assert "IOC" in report.detail
    assert report.confirmed_qty == 0.0
    assert ("BTCUSDT", PositionSide.LONG) not in ex.positions  # sin pierna abierta


# ------------------------------ log auditado ------------------------------

async def test_log_auditado_persiste_las_ordenes(tmp_path):
    ex = make_fake()
    storage = await Storage(tmp_path / "t.db", tmp_path / "c").init()
    execu = Executor(ex, CFG, storage=storage, id_factory=counter_ids())
    await execu.startup()
    await execu.open_position(long_order(qty=1.0))

    orders = await storage.get_orders()
    types = {o["type"] for o in orders}
    assert types == {"MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"}
    entry = next(o for o in orders if o["type"] == "MARKET")
    assert entry["side"] == "BUY" and entry["position_side"] == "LONG"
    assert entry["status"] == "FILLED" and entry["decision_reason"] == "test"
    await storage.close()
