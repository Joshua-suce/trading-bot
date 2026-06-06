import pytest

from src.audit import AuditStore
from src.execution.guards import PreparedOrder
from src.execution.order_manager import OrderManager


class AllowGuard:
    async def prepare(
        self, client, *, symbol, order_type, side, quantity, price, params
    ):
        reference_price = price or 100.0
        return PreparedOrder(
            allowed=True,
            reason="ok",
            quantity=quantity,
            price=price,
            reference_price=reference_price,
            slippage_bps=0.0,
            notional=reference_price * quantity,
        )


class FakeRest:
    def __init__(self, *, fail_cancel_all=False):
        self.cancelled_symbols = []
        self.fail_cancel_all = fail_cancel_all

    async def cancel_all_orders(self, symbol):
        if self.fail_cancel_all:
            raise RuntimeError("cancel all rejected")
        self.cancelled_symbols.append(symbol)


class FlakyClient:
    def __init__(self, *, failures=0, cancel_failure=False, fail_cancel_all=False):
        self.failures = failures
        self.cancel_failure = cancel_failure
        self.created_orders = []
        self.cancelled_orders = []
        self.rest = FakeRest(fail_cancel_all=fail_cancel_all)

    async def create_order(self, symbol, order_type, side, amount, price, params):
        if len(self.created_orders) < self.failures:
            self.created_orders.append({"failed": True})
            raise RuntimeError("temporary exchange failure")
        order = {
            "id": f"order-{len(self.created_orders) + 1}",
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

    async def cancel_order(self, order_id, symbol, params=None):
        if self.cancel_failure:
            raise RuntimeError("cancel rejected")
        self.cancelled_orders.append((symbol, order_id))

    async def cancel_all_orders(self, symbol, conditional=False):
        if self.rest.fail_cancel_all:
            raise RuntimeError("cancel all rejected")
        self.rest.cancelled_symbols.append((symbol, conditional))


@pytest.mark.asyncio
async def test_market_order_retries_then_audits_success(tmp_path, monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("src.execution.order_manager.asyncio.sleep", no_sleep)
    audit = AuditStore(str(tmp_path / "audit.db"))
    client = FlakyClient(failures=2)
    manager = OrderManager(client, audit_store=audit, guard=AllowGuard())

    order = await manager.market_order("BTCUSDT", "buy", 0.25)

    assert order is not None
    assert order["id"] == "order-3"
    assert len(client.created_orders) == 3
    event = audit.load_recent_events(1)[0]
    assert event["event_type"] == "order_placed"
    assert event["payload"]["client_order_id"].startswith("tb_ord_")


@pytest.mark.asyncio
async def test_market_order_audits_permanent_failure(tmp_path, monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("src.execution.order_manager.asyncio.sleep", no_sleep)
    audit = AuditStore(str(tmp_path / "audit.db"))
    client = FlakyClient(failures=5)
    manager = OrderManager(client, audit_store=audit, guard=AllowGuard())

    order = await manager.market_order("BTCUSDT", "buy", 0.25)

    assert order is None
    event = audit.load_recent_events(1)[0]
    assert event["event_type"] == "order_failed"
    assert event["severity"] == "error"
    assert "temporary exchange failure" in event["payload"]["error"]
    assert "temporary exchange failure" in manager.failure_reason("BTCUSDT", "fallback")


@pytest.mark.asyncio
async def test_invalid_quantity_does_not_call_exchange(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    client = FlakyClient()
    manager = OrderManager(client, audit_store=audit, guard=AllowGuard())

    order = await manager.limit_order("BTCUSDT", "buy", 0, 100.0)

    assert order is None
    assert client.created_orders == []
    assert audit.load_recent_events(1) == []


@pytest.mark.asyncio
async def test_cancel_paths_audit_success_and_failure(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    client = FlakyClient()
    manager = OrderManager(client, audit_store=audit, guard=AllowGuard())

    await manager.cancel_all_orders("BTCUSDT")
    await manager.cancel_order("BTCUSDT", "order-1")

    events = audit.load_recent_events(3)
    assert {event["event_type"] for event in events} == {
        "orders_cancelled",
        "order_cancelled",
    }
    assert client.rest.cancelled_symbols == [
        ("BTCUSDT", False),
        ("BTCUSDT", True),
    ]
    assert client.cancelled_orders == [("BTCUSDT", "order-1")]

    failing_client = FlakyClient(cancel_failure=True, fail_cancel_all=True)
    failing_manager = OrderManager(
        failing_client, audit_store=audit, guard=AllowGuard()
    )
    await failing_manager.cancel_all_orders("ETHUSDT")
    await failing_manager.cancel_order("ETHUSDT", "order-2")

    failed_events = audit.load_recent_events(3)
    assert [event["event_type"] for event in failed_events] == [
        "order_cancel_failed",
        "order_cancel_failed",
        "order_cancel_failed",
    ]
    assert all(event["severity"] == "error" for event in failed_events)
