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
        self,
        mode,
        symbol,
        side,
        price,
        quantity,
        stop_loss,
        take_profit,
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


def build_paper_manager(alerter, tmp_path):
    portfolio = PortfolioManager()
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
        client=object(),
        order_mgr=object(),
        pos_sizer=PositionSizer(portfolio),
        sl_mgr=StopLossManager(),
        portfolio=portfolio,
        paper=True,
        alerter=alerter,
        mode="paper",
        audit_store=AuditStore(str(tmp_path / "audit.db")),
    )


def build_no_account_paper_manager(alerter, tmp_path):
    portfolio = PortfolioManager()
    return PositionManager(
        client=object(),
        order_mgr=object(),
        pos_sizer=PositionSizer(portfolio),
        sl_mgr=StopLossManager(),
        portfolio=portfolio,
        paper=True,
        alerter=alerter,
        mode="paper",
        audit_store=AuditStore(str(tmp_path / "audit.db")),
    )


@pytest.mark.asyncio
async def test_paper_trade_open_and_complete_alerts(tmp_path):
    alerter = FakeAlerter()
    manager = build_paper_manager(alerter, tmp_path)

    opened = await manager.enter_long("BTCUSDT", price=50000.0, atr=100.0)
    await manager.exit_position("BTCUSDT", reason="test_exit")

    assert opened is True
    assert alerter.opened[0]["symbol"] == "BTCUSDT"
    assert alerter.opened[0]["mode"] == "paper"
    assert alerter.completed[0]["reason"] == "test_exit"
    assert alerter.completed[0]["pnl"] is not None


@pytest.mark.asyncio
async def test_paper_trade_failure_alert_when_sizing_rejects(tmp_path):
    alerter = FakeAlerter()
    manager = build_no_account_paper_manager(alerter, tmp_path)

    opened = await manager.enter_short("ETHUSDT", price=3000.0, atr=50.0)

    assert opened is False
    assert alerter.failed[0]["symbol"] == "ETHUSDT"


def test_position_key_respects_symbol_timeframe_scope(monkeypatch):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(
        position_manager_module.settings, "position_scope", "symbol_timeframe"
    )

    assert PositionManager.position_key("BTC/USDT:USDT", "5m") == "BTCUSDT:5m"


@pytest.mark.asyncio
async def test_exposure_blocks_max_open_positions(monkeypatch, tmp_path):
    from src.execution import position_manager as position_manager_module

    monkeypatch.setattr(position_manager_module.settings, "max_open_positions", 1)
    alerter = FakeAlerter()
    manager = build_paper_manager(alerter, tmp_path)

    opened = await manager.enter_long("BTCUSDT", price=50000.0, atr=100.0)
    blocked = await manager.enter_long("ETHUSDT", price=3000.0, atr=50.0)

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
    manager = build_paper_manager(alerter, tmp_path)

    opened = await manager.enter_long("BTCUSDT", price=50000.0, atr=100.0)

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
    manager = build_paper_manager(alerter, tmp_path)

    opened = await manager.enter_short("BTCUSDT", price=50000.0, atr=100.0)

    assert opened is False
    assert "symbol exposure" in alerter.failed[-1]["reason"]
