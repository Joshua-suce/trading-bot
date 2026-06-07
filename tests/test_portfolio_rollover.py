from datetime import datetime

from src.exchange.account import AccountInfo
from src.risk import portfolio as portfolio_module
from src.risk.portfolio import PortfolioManager, TradeRecord


class JuneSix(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 6, 6)


class JuneSeven(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 6, 7)


def test_close_trade_initializes_new_daily_bucket_after_midnight(monkeypatch):
    monkeypatch.setattr(portfolio_module, "datetime", JuneSix)
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
    trade = TradeRecord(
        symbol="ETHUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=JuneSix.now(),
    )

    monkeypatch.setattr(portfolio_module, "datetime", JuneSeven)
    portfolio.close_trade(trade, 101.0, "take_profit")

    assert portfolio.daily_stats["2026-06-07"].trades == 1
    assert portfolio.daily_stats["2026-06-07"].total_pnl == 1.0


def test_restore_risk_state_preserves_daily_loss_and_loss_streak(monkeypatch):
    monkeypatch.setattr(portfolio_module, "datetime", JuneSeven)
    portfolio = PortfolioManager()

    portfolio.restore_risk_state(
        [
            {"pnl": -2.0, "closed_at": "2026-06-07T00:30:00+00:00"},
            {"pnl": -1.0, "closed_at": "2026-06-07T00:20:00+00:00"},
            {"pnl": 3.0, "closed_at": "2026-06-06T23:50:00+00:00"},
        ]
    )

    stats = portfolio.daily_stats["2026-06-07"]
    assert stats.trades == 2
    assert stats.losses == 2
    assert stats.total_pnl == -3.0
    assert portfolio.daily_pnl == -3.0
    assert portfolio.consecutive_losses == 2
