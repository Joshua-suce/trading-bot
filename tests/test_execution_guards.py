import pytest

from src.audit import AuditStore
from src.execution.guards import ExecutionGuard
from src.execution.order_manager import OrderManager


class FakeExchangeClient:
    def __init__(self, *, ticker=None, market=None):
        self.created_orders = []
        self.ticker = ticker or {"last": 100.0, "bid": 99.95, "ask": 100.05}
        self.market = market or {
            "precision": {"amount": 3, "price": 2},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
        }

    async def fetch_market(self, symbol):
        return self.market

    async def fetch_ticker(self, symbol):
        return self.ticker

    async def create_order(self, symbol, order_type, side, amount, price, params):
        order = {
            "id": "order-1",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": params,
            "filled": amount,
        }
        self.created_orders.append(order)
        return order


@pytest.mark.asyncio
async def test_execution_guard_rounds_amount_and_price():
    client = FakeExchangeClient()
    guard = ExecutionGuard(max_slippage_bps=50.0, min_notional=5.0)

    prepared = await guard.prepare(
        client,
        symbol="BTCUSDT",
        order_type="limit",
        side="buy",
        quantity=0.123456,
        price=100.129,
        params={},
    )

    assert prepared.allowed is True
    assert prepared.quantity == 0.123
    assert prepared.price == 100.12


@pytest.mark.asyncio
async def test_execution_guard_supports_ccxt_tick_size_precision():
    client = FakeExchangeClient(
        ticker={"last": 60_000.0, "bid": 59_999.0, "ask": 60_001.0},
        market={
            "precision": {"amount": 0.001, "price": 0.1},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 50.0}},
        },
    )
    guard = ExecutionGuard(max_slippage_bps=50.0, min_notional=5.0)

    prepared = await guard.prepare(
        client,
        symbol="BTCUSDT",
        order_type="limit",
        side="buy",
        quantity=0.001648,
        price=60_000.19,
        params={},
    )

    assert prepared.allowed is True
    assert prepared.quantity == 0.001
    assert prepared.price == 60_000.1
    assert prepared.notional == pytest.approx(60.0001)


@pytest.mark.asyncio
async def test_execution_guard_never_increases_quantity_to_exchange_minimum():
    client = FakeExchangeClient(
        ticker={"last": 60_000.0, "bid": 59_999.0, "ask": 60_001.0},
        market={
            "precision": {"amount": 0.001, "price": 0.1},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 50.0}},
        },
    )
    guard = ExecutionGuard(max_slippage_bps=50.0, min_notional=5.0)

    prepared = await guard.prepare(
        client,
        symbol="BTCUSDT",
        order_type="market",
        side="buy",
        quantity=0.0005,
        price=None,
        params={},
    )

    assert prepared.allowed is False
    assert prepared.quantity == 0.0
    assert "amount below minimum" in prepared.reason


def test_tick_size_of_one_rounds_to_whole_units():
    assert ExecutionGuard._round_down(3.9, 1.0) == 3.0


@pytest.mark.asyncio
async def test_order_manager_blocks_small_notional(tmp_path):
    client = FakeExchangeClient()
    audit = AuditStore(str(tmp_path / "audit.db"))
    orders = OrderManager(
        client,
        audit_store=audit,
        guard=ExecutionGuard(max_slippage_bps=50.0, min_notional=50.0),
    )

    order = await orders.market_order("BTCUSDT", "buy", 0.01)

    assert order is None
    assert client.created_orders == []
    events = audit.load_recent_events(5)
    assert events[0]["event_type"] == "order_blocked"


@pytest.mark.asyncio
async def test_order_manager_blocks_entry_slippage(tmp_path):
    client = FakeExchangeClient(ticker={"last": 100.0, "bid": 99.0, "ask": 105.0})
    audit = AuditStore(str(tmp_path / "audit.db"))
    orders = OrderManager(
        client,
        audit_store=audit,
        guard=ExecutionGuard(max_slippage_bps=25.0, min_notional=5.0),
    )

    order = await orders.market_order("BTCUSDT", "buy", 1.0)

    assert order is None
    assert client.created_orders == []
    assert "slippage" in audit.load_recent_events(5)[0]["message"]


@pytest.mark.asyncio
async def test_order_manager_allows_reduce_only_exit_despite_slippage(tmp_path):
    client = FakeExchangeClient(ticker={"last": 100.0, "bid": 90.0, "ask": 110.0})
    orders = OrderManager(
        client,
        guard=ExecutionGuard(max_slippage_bps=1.0, min_notional=5.0),
    )

    order = await orders.market_order("BTCUSDT", "sell", 1.0, reduce_only=True)

    assert order is not None
    assert client.created_orders[0]["amount"] == 1.0


@pytest.mark.asyncio
async def test_order_manager_allows_reduce_only_exit_below_notional_minimum(tmp_path):
    client = FakeExchangeClient(
        market={
            "precision": {"amount": 3, "price": 2},
            "limits": {"amount": {"min": 0.001}, "cost": {"min": 50.0}},
        }
    )
    orders = OrderManager(
        client,
        guard=ExecutionGuard(max_slippage_bps=1.0, min_notional=50.0),
    )

    order = await orders.market_order(
        "BTCUSDT",
        "sell",
        0.01,
        reduce_only=True,
    )

    assert order is not None
    assert client.created_orders[0]["params"]["reduceOnly"] is True
