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
        assert levels.take_profit < 50000  # Below entry for short

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
