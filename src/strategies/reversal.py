from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategyMath, StrategySignal


class ReversalStrategy(StrategyMath):
    name = "reversal"

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
        if trend_regime != -direction or adx < 22:
            return cls.name, direction, 0.0
        rsi = cls.safe_float(last.get("rsi_14"), 50.0)
        macd = cls.safe_float(last.get("macd_hist"))
        previous_macd = cls.safe_float(prev.get("macd_hist"), macd)
        plus_di = cls.safe_float(last.get("plus_di"))
        minus_di = cls.safe_float(last.get("minus_di"))
        volume_ratio = cls.safe_float(last.get("vol_ratio"), 0.0)
        exhausted = rsi <= 34 if direction == 1 else rsi >= 66
        momentum_turn = (macd - previous_macd) * direction > 0
        di_turn = plus_di > minus_di if direction == 1 else minus_di > plus_di
        rejection = bool(
            candle["bullish_rejection"]
            if direction == 1
            else candle["bearish_rejection"]
        )
        if not exhausted or not rejection or volume_ratio < 0.70:
            return cls.name, direction, 0.0
        score = sum(
            (
                0.26 if exhausted else 0.0,
                0.22 if momentum_turn else 0.0,
                0.16 if di_turn else 0.0,
                0.24 if rejection else 0.0,
                0.08 if volume_ratio >= 0.80 else 0.0,
                0.04 if volume_ratio >= 1.10 else 0.0,
            )
        )
        return cls.name, direction, min(score, 1.0)

    @classmethod
    def oscillator_signals(
        cls,
        last: pd.Series,
        prev: pd.Series,
        *,
        rsi: float,
        strong_trend: bool,
        trend_direction: int,
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        if pd.notna(last.get("rsi_14")):
            if rsi < 30:
                strength = 0.55 if trend_direction >= 0 or not strong_trend else 0.25
                signals.append(StrategySignal(1, strength, "rsi_oversold", cls.name))
            elif rsi > 70:
                strength = 0.55 if trend_direction <= 0 or not strong_trend else 0.25
                signals.append(StrategySignal(-1, strength, "rsi_overbought", cls.name))
            elif rsi < 40 and not strong_trend:
                signals.append(StrategySignal(1, 0.3, "rsi_bullish", cls.name))
            elif rsi > 60 and not strong_trend:
                signals.append(StrategySignal(-1, 0.3, "rsi_bearish", cls.name))

        if cls.has(last, "macd_hist") and cls.has(prev, "macd_hist"):
            if last["macd_hist"] > 0 and prev["macd_hist"] <= 0:
                signals.append(StrategySignal(1, 0.55, "macd_bull_cross", cls.name))
            elif last["macd_hist"] < 0 and prev["macd_hist"] >= 0:
                signals.append(StrategySignal(-1, 0.55, "macd_bear_cross", cls.name))
        return signals

    @staticmethod
    def claims_source(source: str) -> bool:
        return source.startswith("reversal_structure_") or source in {
            "rsi_oversold",
            "rsi_overbought",
            "bb_lower_bounce",
            "bb_upper_reject",
            "macd_bull_cross",
            "macd_bear_cross",
        }
