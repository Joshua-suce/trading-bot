from datetime import datetime, timezone

import pytest

from src.exchange.account import AccountInfo
from src.risk.portfolio import PortfolioManager
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager


class TestStopLoss:
    def test_long_stop_loss(self):
        sl = StopLossManager(atr_mult_sl=1.5, risk_reward_ratio=2.0)
        levels = sl.calculate(entry_price=50000, side="long", atr=1000)
        assert levels.stop_loss < 50000  # Below entry
        assert levels.take_profit > 50000  # Above entry
        assert levels.stop_loss_pct > 0
        assert levels.take_profit_pct > 0

    def test_short_stop_loss(self):
        sl = StopLossManager(atr_mult_sl=1.5, risk_reward_ratio=2.0)
        levels = sl.calculate(entry_price=50000, side="short", atr=1000)
        assert levels.stop_loss > 50000  # Above entry for short
        assert levels.take_profit < 50000

    def test_range_target_uses_mean_reversion_level_when_reward_is_valid(self):
        sl = StopLossManager()
        levels = sl.calculate(
            entry_price=100.0,
            side="long",
            atr=1.0,
            strategy="range",
            market_context={"bb_middle": 102.0, "vwap": 101.0},
        )

        assert levels.take_profit == 102.0
        assert levels.take_profit_pct == 2.0

    def test_range_target_falls_back_when_mean_target_is_too_close(self):
        sl = StopLossManager()
        levels = sl.calculate(
            entry_price=100.0,
            side="long",
            atr=1.0,
            strategy="range",
            market_context={"bb_middle": 100.5, "vwap": 100.4},
        )

        assert levels.take_profit > 101.0

    def test_trailing_stop_long(self):
        sl = StopLossManager()
        new_stop = sl.update_trailing(
            current_price=51000,
            side="long",
            entry_price=50000,
            atr=1000,
            current_stop=48500,
        )
        assert new_stop > 48500  # Trailing should move up

    def test_trailing_stop_short(self):
        sl = StopLossManager()
        new_stop = sl.update_trailing(
            current_price=49000,
            side="short",
            entry_price=50000,
            atr=1000,
            current_stop=51500,
        )
        assert new_stop < 51500  # Trailing should move down

    def test_stop_distance_is_not_tighter_than_noise_floor(self):
        levels = StopLossManager(
            min_stop_loss_pct=0.0035,
            max_stop_loss_pct=0.02,
        ).calculate(entry_price=1000.0, side="long", atr=0.1)

        assert levels.stop_loss == 996.5
        assert levels.stop_loss_pct == 0.35

    def test_stop_distance_is_capped_for_high_timeframes(self):
        levels = StopLossManager(
            min_stop_loss_pct=0.0035,
            max_stop_loss_pct=0.02,
        ).calculate(entry_price=1000.0, side="long", atr=100.0)

        assert levels.stop_loss == 980.0
        assert levels.take_profit == 1040.0
        assert levels.stop_loss_pct == 2.0


def test_exchange_leverage_does_not_multiply_stop_risk_position_size():
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
    sizer = PositionSizer(portfolio)

    unleveraged = sizer.calculate(100.0, 95.0, leverage=1)
    leveraged = sizer.calculate(100.0, 95.0, leverage=5)

    assert leveraged.quantity == unleveraged.quantity
    assert leveraged.leveraged_quantity == leveraged.quantity


def test_strategy_loss_streak_does_not_block_other_strategy(monkeypatch):
    portfolio = PortfolioManager()
    monkeypatch.setattr(portfolio, "strategy_max_consecutive_losses", 2)
    portfolio.strategy_consecutive_losses["trend"] = 2
    portfolio.strategy_last_loss_at["trend"] = datetime.now(timezone.utc)

    trend_allowed, trend_reason = portfolio.can_trade("BTCUSDT", "trend")
    scalp_allowed, scalp_reason = portfolio.can_trade("BTCUSDT", "scalp")

    assert not trend_allowed
    assert "trend loss limit" in trend_reason
    assert scalp_allowed
    assert scalp_reason == "ok"


def test_scalp_uses_tighter_stop_and_its_own_reward_ratio(monkeypatch):
    monkeypatch.setattr("src.risk.stop_loss.settings.scalp_atr_stop_multiplier", 0.8)
    monkeypatch.setattr("src.risk.stop_loss.settings.scalp_min_stop_loss_pct", 0.0015)
    monkeypatch.setattr("src.risk.stop_loss.settings.scalp_max_stop_loss_pct", 0.006)
    monkeypatch.setattr("src.risk.stop_loss.settings.scalp_risk_reward_ratio", 1.5)
    manager = StopLossManager(atr_mult_sl=1.5, risk_reward_ratio=2.0)

    normal = manager.calculate(1000.0, "long", 5.0)
    scalp = manager.calculate(1000.0, "long", 5.0, strategy="scalp")

    assert scalp.stop_loss > normal.stop_loss
    assert scalp.take_profit == 1006.0


def test_scalp_position_size_uses_smaller_risk_budget(monkeypatch):
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
    monkeypatch.setattr("src.risk.position_sizer.settings.risk_per_trade", 0.01)
    monkeypatch.setattr("src.risk.position_sizer.settings.scalp_risk_per_trade", 0.003)
    monkeypatch.setattr("src.risk.position_sizer.settings.max_position_size", 1.0)
    sizer = PositionSizer(portfolio)

    normal = sizer.calculate(100.0, 99.0)
    scalp = sizer.calculate(100.0, 99.0, strategy="scalp")

    assert normal.risk_amount == 100.0
    assert scalp.risk_amount == 30.0
    assert scalp.quantity < normal.quantity


def test_mainnet_canary_caps_risk_and_notional(monkeypatch):
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
    monkeypatch.setattr(
        "src.risk.position_sizer.settings.binance_api_url",
        "https://fapi.binance.com",
    )
    monkeypatch.setattr("src.risk.position_sizer.settings.mainnet_canary_enabled", True)
    monkeypatch.setattr("src.risk.position_sizer.settings.risk_per_trade", 0.01)
    monkeypatch.setattr(
        "src.risk.position_sizer.settings.mainnet_canary_risk_per_trade",
        0.0025,
    )
    monkeypatch.setattr("src.risk.position_sizer.settings.max_position_size", 0.02)
    monkeypatch.setattr(
        "src.risk.position_sizer.settings.mainnet_canary_max_position_size",
        0.005,
    )

    size = PositionSizer(portfolio).calculate(100.0, 90.0)

    assert size.risk_amount == 5.0
    assert size.quantity == 0.5


def test_portfolio_classifies_trade_by_net_pnl_after_fees():
    from datetime import datetime

    from src.risk.portfolio import TradeRecord

    portfolio = PortfolioManager()
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(),
        entry_fee=0.06,
    )

    portfolio.close_trade(trade, 100.1, exit_fee=0.06)

    assert trade.gross_pnl == pytest.approx(0.1)
    assert trade.pnl == pytest.approx(-0.02)
    assert portfolio.consecutive_losses == 1
