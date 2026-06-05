import sqlite3
from contextlib import closing
from datetime import datetime

import pytest

from src.audit import AuditStore
from src.exchange.account import AccountInfo
from src.execution.position_manager import PositionManager
from src.risk.portfolio import PortfolioManager, TradeRecord
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager


class FakeOrderManager:
    def __init__(self, *, stop_ok=True, tp_ok=True, flatten_ok=True):
        self.stop_ok = stop_ok
        self.tp_ok = tp_ok
        self.flatten_ok = flatten_ok
        self.market_orders = []
        self.cancelled_all = []

    async def market_order(self, symbol, side, quantity, reduce_only=False):
        self.market_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "reduce_only": reduce_only,
            }
        )
        if reduce_only and not self.flatten_ok:
            return None
        return {
            "id": f"m-{len(self.market_orders)}",
            "filled": quantity,
            "average": 100.0,
        }

    async def stop_loss_order(self, symbol, side, quantity, stop_price, price=None):
        if not self.stop_ok:
            return None
        return {"id": "sl-1"}

    async def take_profit_order(self, symbol, side, quantity, price):
        if not self.tp_ok:
            return None
        return {"id": "tp-1"}

    async def cancel_all_orders(self, symbol):
        self.cancelled_all.append(symbol)

    async def cancel_order(self, symbol, order_id):
        return None


class FakeClient:
    def __init__(self, positions=None, open_orders=None):
        self.positions = positions or []
        self.open_orders = open_orders or {}

    async def fetch_positions(self):
        return self.positions

    async def fetch_open_orders(self, symbol):
        return self.open_orders.get(symbol, [])


class FakeAlerter:
    def __init__(self):
        self.failed = []
        self.errors = []

    async def trade_failed_alert(self, mode, symbol, reason):
        self.failed.append({"mode": mode, "symbol": symbol, "reason": reason})

    async def trade_opened_alert(self, *args):
        return None

    async def trade_completed_alert(self, *args):
        return None

    async def error_alert(self, error):
        self.errors.append(error)


def build_manager(tmp_path, order_mgr, client=None, paper=False):
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountInfo(
            total_equity=10_000.0,
            wallet_balance=10_000.0,
            available_balance=10_000.0,
            unrealized_pnl=0.0,
            margin_ratio=0.0,
        )
    )
    audit = AuditStore(str(tmp_path / "audit.db"))
    return PositionManager(
        client=client or FakeClient(),
        order_mgr=order_mgr,
        pos_sizer=PositionSizer(portfolio),
        sl_mgr=StopLossManager(),
        portfolio=portfolio,
        paper=paper,
        alerter=FakeAlerter(),
        mode="live" if not paper else "paper",
        audit_store=audit,
    )


@pytest.mark.asyncio
async def test_live_entry_flattens_and_fails_when_stop_loss_is_not_confirmed(tmp_path):
    orders = FakeOrderManager(stop_ok=False)
    manager = build_manager(tmp_path, orders)

    opened = await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)

    assert opened is False
    assert "BTCUSDT" not in manager.open_trades
    assert orders.cancelled_all == ["BTCUSDT"]
    assert orders.market_orders[-1]["reduce_only"] is True
    assert manager.alerter.failed[0]["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_failed_emergency_flatten_activates_kill_switch(tmp_path):
    orders = FakeOrderManager(stop_ok=False, flatten_ok=False)
    manager = build_manager(tmp_path, orders)

    opened = await manager.enter_short("ETHUSDT", price=100.0, atr=2.0)

    assert opened is False
    allowed, reason = manager.audit_store.trading_allowed()
    assert allowed is False
    assert "emergency stop" in reason


@pytest.mark.asyncio
async def test_reconciliation_blocks_unmanaged_live_position(tmp_path):
    orders = FakeOrderManager()
    client = FakeClient(positions=[{"symbol": "BTCUSDT", "contracts": 0.25}])
    manager = build_manager(tmp_path, orders, client=client)

    reconciled = await manager.reconcile_exchange_state()

    assert reconciled is False
    allowed, reason = manager.audit_store.trading_allowed()
    assert allowed is False
    assert "emergency stop" in reason
    assert manager.alerter.errors


def test_audit_store_persists_trade_events(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    audit.record_event("test_event", "hello", symbol="BTCUSDT")

    with closing(sqlite3.connect(tmp_path / "audit.db")) as conn:
        count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    assert count == 1


def test_position_manager_restores_open_trades_from_audit(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.25,
        timestamp=datetime.now(),
    )
    audit.record_open_trade(
        trade,
        mode="live",
        correlation_id="corr-1",
        stop_loss=95.0,
        take_profit=110.0,
        stop_order_id="sl-1",
        take_profit_order_id="tp-1",
    )
    manager = build_manager(tmp_path, FakeOrderManager())
    manager.audit_store = audit

    restored = manager.restore_open_trades_from_audit()

    assert restored == 1
    assert manager.open_trades["BTCUSDT"].quantity == 1.25
    assert manager.active_stops["BTCUSDT"] == "sl-1"
    assert manager.active_tps["BTCUSDT"] == "tp-1"


@pytest.mark.asyncio
async def test_reconciliation_blocks_missing_protective_order(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(),
    )
    audit.record_open_trade(
        trade,
        mode="live",
        correlation_id="corr-2",
        stop_loss=95.0,
        take_profit=110.0,
        stop_order_id="sl-2",
        take_profit_order_id="tp-2",
    )
    client = FakeClient(
        positions=[{"symbol": "BTCUSDT", "contracts": 1.0}],
        open_orders={"BTCUSDT": [{"id": "tp-2"}]},
    )
    manager = build_manager(tmp_path, FakeOrderManager(), client=client)
    manager.audit_store = audit
    manager.restore_open_trades_from_audit()

    reconciled = await manager.reconcile_exchange_state()

    assert reconciled is False
    allowed, reason = manager.audit_store.trading_allowed()
    assert allowed is False
    assert "emergency stop" in reason
