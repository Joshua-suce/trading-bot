from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategyMath, StrategySignal


class BreakoutStrategy(StrategyMath):
    name = "breakout"

    @classmethod
    def donchian_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        *,
        trend_regime: int,
        adx: float,
        adx_strength: float,
        volume_ratio: float,
        candle: dict[str, float | bool],
    ) -> StrategySignal | None:
        if not cls.has(last, "dc_upper", "dc_lower"):
            return None

        prev_upper = prev.get("dc_upper")
        prev_lower = prev.get("dc_lower")
        close = cls.safe_float(last.get("close"))
        atr = cls.safe_float(last.get("atr"), close * 0.01)
        breakout_buffer = max(atr * 0.10, close * 0.0005)
        if (
            pd.notna(prev_upper)
            and last["close"] > prev_upper + breakout_buffer
            and trend_regime >= 0
            and (trend_regime == 1 or adx >= 24)
            and volume_ratio >= 1.35
            and candle["bullish_breakout"]
        ):
            return StrategySignal(
                1,
                0.55 + 0.25 * adx_strength,
                "donchian_breakout_bull",
                cls.name,
            )
        if (
            pd.notna(prev_lower)
            and last["close"] < prev_lower - breakout_buffer
            and trend_regime <= 0
            and (trend_regime == -1 or adx >= 24)
            and volume_ratio >= 1.35
            and candle["bearish_breakout"]
        ):
            return StrategySignal(
                -1,
                0.55 + 0.25 * adx_strength,
                "donchian_breakout_bear",
                cls.name,
            )
        return None

    @staticmethod
    def claims_source(source: str) -> bool:
        return source.startswith("donchian_breakout_")
