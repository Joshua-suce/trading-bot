from datetime import datetime, timezone

import pytest

from src.audit import AuditStore
from src.exchange.account import AccountInfo
from src.execution.position_manager import PositionManager
from src.risk.portfolio import PortfolioManager
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager


class FakeAlerter:
    def __init__(self):
        self.opened = []
        self.completed = []
        self.failed = []

    async def trade_opened_alert(
        self, mode, symbol, side, price, quantity, stop_loss, take_profit
    ):
        self.opened.append(
            {
                "mode": mode,
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        )

    async def trade_completed_alert(self, mode, symbol, side, exit_price, pnl, reason):
        self.completed.append(
            {
                "mode": mode,
                "symbol": symbol,
                "side": side,
                "exit_price": exit_price,
                "pnl": pnl,
                "reason": reason,
            }
        )

    async def trade_failed_alert(self, mode, symbol, reason):
        self.failed.append({"mode": mode, "symbol": symbol, "reason": reason})


class FakeOrderManager:
    def __init__(self, *, null_exit_price=False):
        self.counter = 0
        self.null_exit_price = null_exit_price
        self.protective_quantities = []
        self.stop_prices = []
        self.target_prices = []

    async def market_order(self, symbol, side, quantity, reduce_only=False):
        self.counter += 1
        return {
            "id": f"market-{self.counter}",
            "filled": quantity,
            "average": (
                None
                if reduce_only and self.null_exit_price
                else (101.0 if reduce_only else 100.0)
            ),
            "price": None,
        }

    async def stop_loss_order(self, symbol, side, quantity, stop_price, price=None):
        self.protective_quantities.append(quantity)
        self.stop_prices.append(stop_price)
        return {"id": "stop-1"}

    async def take_profit_order(self, symbol, side, quantity, price):
        self.protective_quantities.append(quantity)
        self.target_prices.append(price)
        return {"id": "target-1"}

    async def cancel_order(self, symbol, order_id, conditional=False):
        return None

    async def cancel_all_orders(self, symbol):
        return None


class FakeClient:
    async def fetch_positions(self):
        return []

    async def fetch_open_orders(self, symbol, conditional=False):
        return []

    async def fetch_ticker(self, symbol):
        return {"last": 100.0, "bid": 100.0, "ask": 100.0}

    async def fetch_order(self, order_id, symbol, conditional=False):
        raise LookupError(order_id)

    async def fetch_my_trades(self, symbol, order_id=None, limit=100):
        return []

    async def fetch_orders(self, symbol, conditional=False, limit=50):
        return []


def build_manager(alerter, tmp_path, *, with_account=True):
    portfolio = PortfolioManager()
    if with_account:
        portfolio.update_account(
            AccountInfo(
                total_equity=5000.0,
                wallet_balance=5000.0,
                available_balance=5000.0,
                unrealized_pnl=0.0,
                margin_ratio=0.0,
            )
        )
    return PositionManager(
        client=FakeClient(),
        order_mgr=FakeOrderManager(),
        pos_sizer=PositionSizer(portfolio),
        sl_mgr=StopLossManager(),
        portfolio=portfolio,
        alerter=alerter,
        mode="trade",
        audit_store=AuditStore(str(tmp_path / "audit.db")),
    )


@pytest.mark.asyncio
async def test_entry_records_actual_fill_and_protects_filled_quantity(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    async def partially_filled_order(symbol, side, quantity, reduce_only=False):
        return {"id": "market-fill", "filled": 0.75, "average": 100.0}

    manager.orders.market_order = partially_filled_order

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert manager.open_trades["BTCUSDT"].quantity == 0.75
    assert manager.orders.protective_quantities == [0.75, 0.75]
    assert alerter.opened[0]["quantity"] == 0.75


@pytest.mark.asyncio
async def test_protective_levels_are_recalculated_from_actual_fill(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    async def favorable_short_fill(symbol, side, quantity, reduce_only=False):
        return {"id": "market-fill", "filled": quantity, "average": 100.2}

    manager.orders.market_order = favorable_short_fill

    assert await manager.enter_short("ETHUSDT", price=100.0, atr=2.0)
    assert manager.open_trades["ETHUSDT"].entry_price == 100.2
    assert manager.orders.stop_prices == [102.2]
    assert manager.orders.target_prices == [96.2]


@pytest.mark.asyncio
async def test_adverse_fill_slippage_is_immediately_flattened(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(
        position_manager_module.settings, "max_entry_slippage_bps", 25.0
    )
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)
    calls = []

    async def slipped_fill(symbol, side, quantity, reduce_only=False):
        calls.append((side, reduce_only))
        return {
            "id": f"market-{len(calls)}",
            "filled": quantity,
            "average": 101.0,
        }

    manager.orders.market_order = slipped_fill

    assert not await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert calls == [("buy", False), ("sell", True)]
    assert manager.open_trades == {}
    assert manager.orders.protective_quantities == []
    assert "slippage above limit" in alerter.failed[-1]["reason"]
    assert manager._reentry_cooldown_reason("BTCUSDT")


@pytest.mark.asyncio
async def test_signal_price_drift_blocks_before_market_order(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(
        position_manager_module.settings, "max_entry_slippage_bps", 25.0
    )
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    async def stale_quote(symbol):
        return {"last": 101.0, "bid": 101.0, "ask": 101.0}

    manager.client.fetch_ticker = stale_quote

    assert not await manager.enter_short("BTCUSDT", price=100.0, atr=2.0)
    assert manager.orders.counter == 0
    event = manager.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "trade_blocked_price_drift"
    assert "signal price drift above limit" in alerter.failed[-1]["reason"]


@pytest.mark.asyncio
async def test_entry_fill_price_is_refetched_before_protection(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    async def fill_without_price(symbol, side, quantity, reduce_only=False):
        return {
            "id": "market-fill",
            "filled": quantity,
            "average": None,
            "price": None,
        }

    async def resolved_fill(order_id, symbol, conditional=False):
        return {"id": order_id, "filled": 1.0, "average": 100.2}

    manager.orders.market_order = fill_without_price
    manager.client.fetch_order = resolved_fill

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert manager.open_trades["BTCUSDT"].entry_price == 100.2
    assert manager.orders.stop_prices == [98.2]
    assert manager.orders.target_prices == [104.2]


@pytest.mark.asyncio
async def test_entry_fill_price_uses_weighted_trade_executions(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    async def fill_without_price(symbol, side, quantity, reduce_only=False):
        return {
            "id": "market-fill",
            "filled": quantity,
            "average": None,
            "price": None,
        }

    async def execution_fills(symbol, order_id=None, limit=100):
        return [
            {
                "order": order_id,
                "amount": 0.4,
                "price": 100.0,
            },
            {
                "order": order_id,
                "amount": 0.6,
                "price": 100.2,
            },
        ]

    manager.orders.market_order = fill_without_price
    manager.client.fetch_my_trades = execution_fills

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert manager.open_trades["BTCUSDT"].entry_price == pytest.approx(100.12)
    assert manager.orders.stop_prices == [pytest.approx(98.12)]
    assert manager.orders.target_prices == [pytest.approx(104.12)]


@pytest.mark.asyncio
async def test_entry_fill_price_retries_delayed_trade_executions(monkeypatch, tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)
    calls = 0

    async def fill_without_price(symbol, side, quantity, reduce_only=False):
        return {
            "id": "market-fill",
            "filled": quantity,
            "average": None,
            "price": None,
        }

    async def delayed_execution(symbol, order_id=None, limit=100):
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        return [{"order": order_id, "amount": 1.0, "price": 100.1}]

    async def no_sleep(seconds):
        return None

    manager.orders.market_order = fill_without_price
    manager.client.fetch_my_trades = delayed_execution
    monkeypatch.setattr("src.execution.position_manager.asyncio.sleep", no_sleep)

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert calls == 2
    assert manager.open_trades["BTCUSDT"].entry_price == 100.1


@pytest.mark.asyncio
async def test_missing_entry_fill_price_is_immediately_flattened(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)
    calls = []

    async def fill_without_price(symbol, side, quantity, reduce_only=False):
        calls.append((side, reduce_only))
        return {
            "id": "market-fill",
            "filled": quantity,
            "average": 100.0 if reduce_only else None,
            "price": None,
        }

    manager.orders.market_order = fill_without_price

    assert not await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert calls == [("buy", False), ("sell", True)]
    assert manager.orders.protective_quantities == []
    assert "fill price unavailable" in alerter.failed[-1]["reason"]


@pytest.mark.asyncio
async def test_reentry_cooldown_blocks_same_symbol(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(
        position_manager_module.settings, "reentry_cooldown_seconds", 900
    )
    manager = build_manager(FakeAlerter(), tmp_path)
    manager.last_symbol_exit_at["BTCUSDT"] = datetime.now(timezone.utc)

    assert not await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert manager.orders.counter == 0
    event = manager.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "trade_blocked_cooldown"


@pytest.mark.asyncio
async def test_portfolio_risk_block_alert_is_deduplicated(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(
        position_manager_module.settings,
        "risk_block_alert_cooldown_seconds",
        900,
    )
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)
    manager.portfolio.consecutive_losses = manager.portfolio.max_consecutive_losses
    manager.portfolio.last_loss_at = datetime.now(timezone.utc)

    assert not await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    assert not await manager.enter_long("ETHUSDT", price=100.0, atr=2.0)

    assert len(alerter.failed) == 1
    events = manager.audit_store.load_recent_events(10)
    assert sum(event["event_type"] == "trade_blocked" for event in events) == 1


@pytest.mark.asyncio
async def test_exit_uses_ticker_when_market_fill_has_null_prices(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)
    manager.orders.null_exit_price = True

    async def exit_quote(symbol):
        return {"last": 102.0}

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    manager.client.fetch_ticker = exit_quote
    await manager.exit_position("BTCUSDT", reason="emergency")

    assert alerter.completed[0]["exit_price"] == 102.0


@pytest.mark.asyncio
async def test_trade_open_and_complete_alerts(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    opened = await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    await manager.exit_position("BTCUSDT", reason="test_exit")

    assert opened is True
    assert alerter.opened[0]["symbol"] == "BTCUSDT"
    assert alerter.opened[0]["mode"] == "trade"
    assert alerter.completed[0]["reason"] == "test_exit"
    assert alerter.completed[0]["pnl"] is not None


@pytest.mark.asyncio
async def test_trade_failure_alert_when_sizing_rejects(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path, with_account=False)

    opened = await manager.enter_short("ETHUSDT", price=3000.0, atr=50.0)

    assert opened is False
    assert alerter.failed[0]["symbol"] == "ETHUSDT"


def test_position_key_is_reconciled_per_symbol():
    assert PositionManager.position_key("BTC/USDT:USDT", "5m") == "BTCUSDT:5m"


@pytest.mark.asyncio
async def test_exit_keeps_trade_open_when_fill_is_not_confirmed(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)

    async def unfilled_exit(symbol, side, quantity, reduce_only=False):
        return {"id": "unfilled-exit", "filled": 0.0, "status": "open"}

    manager.orders.market_order = unfilled_exit
    await manager.exit_position("BTCUSDT", reason="manual")

    assert "BTCUSDT" in manager.open_trades
    assert manager.active_stops["BTCUSDT"] == "stop-1"
    assert manager.active_tps["BTCUSDT"] == "target-1"
    assert alerter.completed == []
    assert "exit order failed" in alerter.failed[-1]["reason"]


@pytest.mark.asyncio
async def test_exit_confirmation_recovers_fill_from_private_executions(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)
    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)

    async def ambiguous_exit(symbol, side, quantity, reduce_only=False):
        return {
            "id": "exit-ledger-1",
            "filled": 0.0,
            "average": None,
            "status": "open",
        }

    async def execution_fills(symbol, order_id=None, limit=100):
        if order_id != "exit-ledger-1":
            return []
        return [
            {"order": order_id, "amount": 0.4, "price": 101.0},
            {"order": order_id, "amount": 0.6, "price": 102.0},
        ]

    manager.orders.market_order = ambiguous_exit
    manager.client.fetch_my_trades = execution_fills

    await manager.exit_position("BTCUSDT", reason="manual")

    assert manager.open_trades == {}
    assert alerter.completed[0]["exit_price"] == pytest.approx(101.6)


@pytest.mark.asyncio
async def test_exposure_blocks_max_open_positions(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(position_manager_module.settings, "max_open_positions", 1)
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    opened = await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
    blocked = await manager.enter_long("ETHUSDT", price=100.0, atr=2.0)

    assert opened is True
    assert blocked is False
    assert "max open positions" in alerter.failed[-1]["reason"]


@pytest.mark.asyncio
async def test_exposure_blocks_total_notional(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(position_manager_module.settings, "max_open_positions", 10)
    monkeypatch.setattr(
        position_manager_module.settings, "max_total_open_notional_pct", 0.01
    )
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    opened = await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)

    assert opened is False
    assert "total exposure" in alerter.failed[-1]["reason"]


@pytest.mark.asyncio
async def test_exposure_blocks_symbol_notional(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(position_manager_module.settings, "max_open_positions", 10)
    monkeypatch.setattr(
        position_manager_module.settings, "max_total_open_notional_pct", 1.0
    )
    monkeypatch.setattr(
        position_manager_module.settings, "max_symbol_open_notional_pct", 0.01
    )
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    opened = await manager.enter_short("BTCUSDT", price=100.0, atr=2.0)

    assert opened is False
    assert "symbol exposure" in alerter.failed[-1]["reason"]


@pytest.mark.asyncio
async def test_allows_three_same_direction_legs_per_symbol(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(position_manager_module.settings, "max_positions_per_symbol", 3)
    monkeypatch.setattr(position_manager_module.settings, "max_open_positions", 6)
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0, timeframe="5m")
    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0, timeframe="15m")
    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0, timeframe="30m")
    assert set(manager.open_trades) == {
        "BTCUSDT:5m",
        "BTCUSDT:15m",
        "BTCUSDT:30m",
    }

    assert not await manager.enter_long("BTCUSDT", price=100.0, atr=2.0, timeframe="1h")
    assert alerter.failed == []
    event = manager.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "trade_skipped_policy"
    assert "max trade legs for BTCUSDT reached: 3" in event["message"]


@pytest.mark.asyncio
async def test_blocks_opposite_side_leg_for_open_symbol(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    assert await manager.enter_long("ETHUSDT", price=100.0, atr=2.0, timeframe="5m")
    assert not await manager.enter_short(
        "ETHUSDT", price=100.0, atr=2.0, timeframe="15m"
    )
    assert set(manager.open_trades) == {"ETHUSDT:5m"}
    assert alerter.failed == []
    event = manager.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "trade_skipped_policy"
    assert "opposite-side entry blocked" in event["message"]


@pytest.mark.asyncio
async def test_blocks_duplicate_timeframe_leg(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)

    assert await manager.enter_short("BTCUSDT", price=100.0, atr=2.0, timeframe="4h")
    assert not await manager.enter_short(
        "BTCUSDT", price=100.0, atr=2.0, timeframe="4h"
    )
    assert set(manager.open_trades) == {"BTCUSDT:4h"}
    assert alerter.failed == []
    event = manager.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "trade_skipped_policy"
    assert "trade leg already open" in event["message"]
