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
        return {"id": "stop-1"}

    async def take_profit_order(self, symbol, side, quantity, price):
        self.protective_quantities.append(quantity)
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
        return {"last": 102.0}

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
async def test_exit_uses_ticker_when_market_fill_has_null_prices(tmp_path):
    alerter = FakeAlerter()
    manager = build_manager(alerter, tmp_path)
    manager.orders.null_exit_price = True

    assert await manager.enter_long("BTCUSDT", price=100.0, atr=2.0)
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
    assert PositionManager.position_key("BTC/USDT:USDT", "5m") == "BTCUSDT"


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
