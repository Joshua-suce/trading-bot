from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategyMath, StrategySignal


class RangeStrategy(StrategyMath):
    name = "range"

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
        if trend_regime != 0 or adx > 25:
            return cls.name, direction, 0.0
        percent_b = cls.safe_float(last.get("bb_percent_b"), 0.5)
        rsi = cls.safe_float(last.get("rsi_14"), 50.0)
        macd = cls.safe_float(last.get("macd_hist"))
        previous_macd = cls.safe_float(prev.get("macd_hist"), macd)
        volume_ratio = cls.safe_float(last.get("vol_ratio"), 0.0)
        band_edge = percent_b <= 0.22 if direction == 1 else percent_b >= 0.78
        rsi_edge = rsi <= 42 if direction == 1 else rsi >= 58
        momentum_turn = (macd - previous_macd) * direction > 0
        rejection = bool(
            candle["bullish_rejection"]
            if direction == 1
            else candle["bearish_rejection"]
        )
        if not band_edge or volume_ratio < 0.65:
            return cls.name, direction, 0.0
        score = sum(
            (
                0.30 if band_edge else 0.0,
                0.20 if rsi_edge else 0.0,
                0.18 if momentum_turn else 0.0,
                0.20 if rejection else 0.0,
                0.08 if volume_ratio >= 0.70 else 0.0,
                0.04 if adx <= 18 else 0.0,
            )
        )
        return cls.name, direction, min(score, 1.0)

    @classmethod
    def oscillator_signals(
        cls,
        last: pd.Series,
        *,
        strong_trend: bool,
        trend_direction: int,
        candle: dict[str, float | bool],
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        if cls.has(last, "bb_upper", "bb_lower", "bb_percent_b"):
            if (
                last["close"] <= last["bb_lower"]
                and (not strong_trend or trend_direction >= 0)
                and candle["bullish_rejection"]
            ):
                signals.append(StrategySignal(1, 0.45, "bb_lower_bounce", cls.name))
            elif (
                last["close"] >= last["bb_upper"]
                and (not strong_trend or trend_direction <= 0)
                and candle["bearish_rejection"]
            ):
                signals.append(StrategySignal(-1, 0.45, "bb_upper_reject", cls.name))
            elif last["bb_percent_b"] < 0.2 and not strong_trend:
                signals.append(StrategySignal(1, 0.25, "bb_oversold", cls.name))
            elif last["bb_percent_b"] > 0.8 and not strong_trend:
                signals.append(StrategySignal(-1, 0.25, "bb_overbought", cls.name))
        return signals

    @staticmethod
    def claims_source(source: str) -> bool:
        return source.startswith("range_structure_") or source in {
            "bb_oversold",
            "bb_overbought",
        }
