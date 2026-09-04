from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategyMath, StrategySignal


class TrendStrategy(StrategyMath):
    name = "trend"

    @classmethod
    def structure_score(
        cls,
        last: pd.Series,
        prev: pd.Series,
        direction: int,
        trend_regime: int,
        adx: float,
        candle: dict[str, float | bool],
    ) -> tuple[str, int, float]:
        close = cls.safe_float(last.get("close"))
        ema_50 = cls.safe_float(last.get("ema_50"))
        ema_200 = cls.safe_float(last.get("ema_200"))
        ema_slope = cls.safe_float(last.get("ema_50_slope"))
        macd = cls.safe_float(last.get("macd_hist"))
        previous_macd = cls.safe_float(prev.get("macd_hist"), macd)
        previous_high = cls.safe_float(prev.get("high"), close)
        previous_low = cls.safe_float(prev.get("low"), close)
        rsi = cls.safe_float(last.get("rsi_14"), 50.0)
        volume_ratio = cls.safe_float(last.get("vol_ratio"), 0.0)
        plus_di = cls.safe_float(last.get("plus_di"))
        minus_di = cls.safe_float(last.get("minus_di"))

        trend_stack = (
            close > ema_50 > ema_200 if direction == 1 else close < ema_50 < ema_200
        )
        slope_aligned = ema_slope * direction > 0
        di_aligned = plus_di > minus_di if direction == 1 else minus_di > plus_di
        macd_aligned = macd * direction > 0 or (macd - previous_macd) * direction > 0
        rsi_aligned = 45 <= rsi <= 70 if direction == 1 else 30 <= rsi <= 55
        candle_aligned = bool(
            candle["bullish_confirmation"]
            if direction == 1
            else candle["bearish_confirmation"]
        )
        if trend_regime != direction or adx < 16:
            return cls.name, direction, 0.0
        if not (trend_stack and slope_aligned and di_aligned and macd_aligned):
            return cls.name, direction, 0.0
        if not candle_aligned:
            return cls.name, direction, 0.0

        continuation_break = (
            close > previous_high if direction == 1 else close < previous_low
        )
        score = sum(
            (
                0.24 if trend_stack else 0.0,
                0.14 if slope_aligned else 0.0,
                0.16 if di_aligned else 0.0,
                0.16 if macd_aligned else 0.0,
                0.10 if rsi_aligned else 0.0,
                0.08 if volume_ratio >= 1.15 else 0.0,
                0.04 if volume_ratio >= 1.25 else 0.0,
                0.08 if candle_aligned else 0.0,
                0.10 if adx >= 24 else 0.0,
                0.06 if continuation_break else 0.0,
            )
        )
        return cls.name, direction, min(score, 1.0)

    @classmethod
    def classic_signals(
        cls,
        last: pd.Series,
        prev: pd.Series,
        *,
        adx_strength: float,
        rsi: float,
        candle: dict[str, float | bool],
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        if cls.has(last, "ema_50", "ema_200") and cls.has(prev, "ema_50", "ema_200"):
            if prev["ema_50"] <= prev["ema_200"] and last["ema_50"] > last["ema_200"]:
                signals.append(StrategySignal(1, 0.75, "golden_cross", cls.name))
            elif prev["ema_50"] >= prev["ema_200"] and last["ema_50"] < last["ema_200"]:
                signals.append(StrategySignal(-1, 0.75, "death_cross", cls.name))

        if cls.has(last, "ema_50", "ema_200", "macd_hist", "ema_50_slope"):
            macd_prev = cls.safe_float(
                prev.get("macd_hist"), cls.safe_float(last["macd_hist"])
            )
            macd_rising = last["macd_hist"] > macd_prev
            macd_falling = last["macd_hist"] < macd_prev
            if (
                last["close"] > last["ema_50"] > last["ema_200"]
                and last["ema_50_slope"] > 0
                and macd_rising
                and 48 <= rsi <= 72
                and candle["bullish_confirmation"]
            ):
                signals.append(
                    StrategySignal(
                        1,
                        0.55 + 0.2 * adx_strength,
                        "ema_50_200_continuation_bull",
                        cls.name,
                    )
                )
            elif (
                last["close"] < last["ema_50"] < last["ema_200"]
                and last["ema_50_slope"] < 0
                and macd_falling
                and 28 <= rsi <= 52
                and candle["bearish_confirmation"]
            ):
                signals.append(
                    StrategySignal(
                        -1,
                        0.55 + 0.2 * adx_strength,
                        "ema_50_200_continuation_bear",
                        cls.name,
                    )
                )
        return signals

    @classmethod
    def fibonacci_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        trend_regime: int,
        adx_strength: float,
    ) -> StrategySignal | None:
        levels = ("fib_382", "fib_500", "fib_618")
        available = [
            (name, cls.safe_float(last.get(name)))
            for name in levels
            if pd.notna(last.get(name)) and cls.safe_float(last.get(name)) > 0
        ]
        if not available or trend_regime == 0:
            return None

        close = cls.safe_float(last.get("close"))
        previous_close = cls.safe_float(prev.get("close"), close)
        atr = cls.safe_float(last.get("atr"), close * 0.01)
        tolerance = max(atr * 0.25, close * 0.001)
        level_name, level = min(available, key=lambda item: abs(close - item[1]))
        previous_level = cls.safe_float(prev.get(level_name), level)

        if trend_regime == 1:
            reclaimed = previous_close <= previous_level and close > level
            held_support = (
                cls.safe_float(last.get("low"), close) <= level + tolerance
                and close >= level
                and close > previous_close
            )
            if reclaimed or held_support:
                return StrategySignal(
                    1,
                    0.5 + 0.2 * adx_strength,
                    f"{level_name}_bullish_confluence",
                    cls.name,
                )

        if trend_regime == -1:
            rejected = previous_close >= previous_level and close < level
            held_resistance = (
                cls.safe_float(last.get("high"), close) >= level - tolerance
                and close <= level
                and close < previous_close
            )
            if rejected or held_resistance:
                return StrategySignal(
                    -1,
                    0.5 + 0.2 * adx_strength,
                    f"{level_name}_bearish_confluence",
                    cls.name,
                )
        return None

    @classmethod
    def pullback_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        trend_regime: int,
        adx_strength: float,
        candle: dict[str, float | bool],
    ) -> StrategySignal | None:
        if trend_regime == 0 or not cls.has(last, "close", "ema_50", "atr"):
            return None
        close = cls.safe_float(last.get("close"))
        ema_50 = cls.safe_float(last.get("ema_50"))
        atr = cls.safe_float(last.get("atr"))
        volume_ratio = cls.safe_float(last.get("vol_ratio"), 0.0)
        previous_close = cls.safe_float(prev.get("close"), close)
        if atr <= 0 or abs(close - ema_50) / atr > 1.5 or volume_ratio < 0.85:
            return None
        if (
            trend_regime == 1
            and previous_close <= close
            and close >= ema_50
            and bool(candle["bullish_rejection"])
        ):
            return StrategySignal(
                1,
                0.58 + 0.17 * adx_strength,
                "trend_pullback_ema50_bull",
                cls.name,
            )
        if (
            trend_regime == -1
            and previous_close >= close
            and close <= ema_50
            and bool(candle["bearish_rejection"])
        ):
            return StrategySignal(
                -1,
                0.58 + 0.17 * adx_strength,
                "trend_pullback_ema50_bear",
                cls.name,
            )
        return None

    @staticmethod
    def claims_source(source: str) -> bool:
        return source.startswith(
            (
                "trend_structure_",
                "ema_50_200_continuation_",
                "trend_pullback_",
                "fib_",
            )
        ) or source in {"golden_cross", "death_cross"}
