"""
Regression coverage for src/strategies/{range,breakout,scalp}.py drifting
away from the values they were extracted from in src/signals/ta_signal.py
during the strategy-redesign refactor. None of these thresholds had any
prior test coverage, which is exactly how the drift went unnoticed - each
test below pins the canonical value and, in its docstring/comment, states
what the buggy (looser) value would have wrongly accepted.
"""

import pandas as pd
import pytest

from src.strategies.breakout import BreakoutStrategy
from src.strategies.range import RangeStrategy
from src.strategies.scalp import ScalpStrategy


def _series(**values) -> pd.Series:
    return pd.Series(values)


class TestRangeStrategyAdxCeiling:
    def test_adx_above_25_rejects_range_structure(self):
        # Canonical ceiling is 25 (matches settings.strategy_range_max_adx,
        # used by StrategyQualityGate's own regime check). The drifted
        # value was 28, which would treat adx=26 as still viable here only
        # for StrategyQualityGate to immediately reject it anyway - wasting
        # a cycle and mislabeling the market state as still "rangeable".
        last = _series(bb_percent_b=0.20, rsi_14=35.0, macd_hist=0.1)
        prev = _series(macd_hist=0.0)
        candle = {"bullish_rejection": True, "bearish_rejection": False}

        _, _, score = RangeStrategy.structure_score(
            last, prev, direction=1, trend_regime=0, adx=26.0, candle=candle
        )

        assert score == 0.0

    def test_adx_at_25_still_permits_range_structure(self):
        last = _series(bb_percent_b=0.20, rsi_14=35.0, macd_hist=0.1, vol_ratio=0.80)
        prev = _series(macd_hist=0.0)
        candle = {"bullish_rejection": True, "bearish_rejection": False}

        _, _, score = RangeStrategy.structure_score(
            last, prev, direction=1, trend_regime=0, adx=25.0, candle=candle
        )

        assert score > 0.0


class TestRangeStrategyBandEdge:
    def test_percent_b_between_022_and_030_no_longer_qualifies(self):
        # Canonical band edge is 0.22/0.78 (percent_b <= 0.22 for a long).
        # The drifted value was 0.30/0.70, which would have accepted a
        # materially less oversold/overbought read as a valid range edge.
        last = _series(bb_percent_b=0.26, rsi_14=35.0, macd_hist=0.1)
        prev = _series(macd_hist=0.0)
        candle = {"bullish_rejection": True, "bearish_rejection": False}

        _, _, score = RangeStrategy.structure_score(
            last, prev, direction=1, trend_regime=0, adx=15.0, candle=candle
        )

        assert score == 0.0


class TestBreakoutCandleQuality:
    @pytest.mark.parametrize("direction_key", ["bull", "bear"])
    def test_confirmation_only_candle_does_not_trigger_breakout(self, direction_key):
        # Canonical gate is candle["bullish_breakout"]/["bearish_breakout"]
        # (body_ratio>=0.50, close_location>=0.75) - materially stricter
        # than "confirmation" (body_ratio>=0.25, close_location>=0.50),
        # which the drifted code checked instead. A candle that only
        # satisfies "confirmation" must not trigger a donchian breakout.
        bull = direction_key == "bull"
        last = _series(
            dc_upper=100.0,
            dc_lower=90.0,
            close=101.0 if bull else 89.0,
            atr=1.0,
        )
        prev = _series(dc_upper=100.0, dc_lower=90.0)
        candle = {
            "bullish_confirmation": bull,
            "bearish_confirmation": not bull,
            "bullish_breakout": False,
            "bearish_breakout": False,
        }

        signal = BreakoutStrategy.donchian_signal(
            last,
            prev,
            trend_regime=1 if bull else -1,
            adx=30.0,
            adx_strength=0.6,
            volume_ratio=2.0,
            candle=candle,
        )

        assert signal is None

    def test_real_breakout_candle_still_triggers(self):
        last = _series(dc_upper=100.0, dc_lower=90.0, close=101.0, atr=1.0)
        prev = _series(dc_upper=100.0, dc_lower=90.0)
        candle = {
            "bullish_confirmation": True,
            "bearish_confirmation": False,
            "bullish_breakout": True,
            "bearish_breakout": False,
        }

        signal = BreakoutStrategy.donchian_signal(
            last,
            prev,
            trend_regime=1,
            adx=30.0,
            adx_strength=0.6,
            volume_ratio=2.0,
            candle=candle,
        )

        assert signal is not None
        assert signal.direction == 1


class TestScalpTrendPullbackThresholds:
    def _kwargs(self, **overrides):
        base = dict(
            direction=1,
            close=100.6,
            vwap=100.6,
            ema_50=100.0,
            ema_slope=1.0,
            macd=1.0,
            previous_macd=0.0,
            rsi=55.0,
            stoch_k=60.0,
            stoch_d=50.0,
            exact_cross=False,
            volume_ratio=0.9,
            candle_confirmed=True,
        )
        base.update(overrides)
        return base

    def test_wide_pullback_with_thin_volume_no_longer_qualifies(self):
        # Canonical thresholds: pullback_near_ema<=0.004, pullback_at_vwap
        # <=0.003, extended>0.005, volume_ratio>=1.0. The drifted values
        # (0.008/0.005/0.010, volume>=0.80) doubled the distance tolerance
        # and lowered the volume floor - this exact setup (0.6% pullback,
        # 0.9 volume) satisfied the drifted thresholds but must not satisfy
        # the canonical ones.
        _, score = ScalpStrategy.score_trend_scalp(**self._kwargs())
        assert score == 0.0

    def test_tight_pullback_with_full_volume_still_qualifies(self):
        kwargs = self._kwargs(
            close=100.3, vwap=100.3, volume_ratio=1.2  # 0.3% pullback
        )
        _, score = ScalpStrategy.score_trend_scalp(**kwargs)
        assert score > 0.0
