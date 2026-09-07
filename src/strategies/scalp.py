from __future__ import annotations

import pandas as pd

from src.strategies.base import StrategyMath, StrategySignal


class ScalpStrategy(StrategyMath):
    name = "scalp"
    supported_timeframes = frozenset({1, 3})

    @classmethod
    def generate(
        cls,
        last: pd.Series,
        prev: pd.Series,
        *,
        adx: float,
        candle: dict[str, float | bool],
        higher_trend_bias: int = 0,
    ) -> StrategySignal | None:
        required = (
            "close",
            "ema_50",
            "vwap",
            "macd_hist",
            "rsi_14",
            "stoch_k",
            "stoch_d",
            "vol_ratio",
            "bb_percent_b",
        )
        if not cls.has(last, *required):
            return None

        close = cls.safe_float(last.get("close"))
        ema_50 = cls.safe_float(last.get("ema_50"))
        ema_slope = cls.safe_float(last.get("ema_50_slope"))
        vwap = cls.safe_float(last.get("vwap"))
        macd = cls.safe_float(last.get("macd_hist"))
        previous_macd = cls.safe_float(prev.get("macd_hist"), macd)
        rsi = cls.safe_float(last.get("rsi_14"), 50.0)
        stoch_k = cls.safe_float(last.get("stoch_k"), 50.0)
        stoch_d = cls.safe_float(last.get("stoch_d"), 50.0)
        previous_k = cls.safe_float(prev.get("stoch_k"), stoch_k)
        previous_d = cls.safe_float(prev.get("stoch_d"), stoch_d)
        volume_ratio = cls.safe_float(last.get("vol_ratio"), 0.0)
        percent_b = cls.safe_float(last.get("bb_percent_b"), 0.5)
        bullish_cross = previous_k <= previous_d and stoch_k > stoch_d
        bearish_cross = previous_k >= previous_d and stoch_k < stoch_d

        if adx >= 16 and volume_ratio >= 0.80:
            trend_signal = cls._trend_pullback_signal(
                close=close,
                vwap=vwap,
                ema_50=ema_50,
                ema_slope=ema_slope,
                macd=macd,
                previous_macd=previous_macd,
                rsi=rsi,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                bullish_cross=bullish_cross,
                bearish_cross=bearish_cross,
                volume_ratio=volume_ratio,
                candle=candle,
                higher_trend_bias=higher_trend_bias,
            )
            if trend_signal:
                return trend_signal

        if adx <= 20 and volume_ratio >= 0.75:
            range_signal = cls._range_reversion_signal(
                percent_b=percent_b,
                rsi=rsi,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                bullish_cross=bullish_cross,
                bearish_cross=bearish_cross,
                volume_ratio=volume_ratio,
                candle=candle,
            )
            if range_signal:
                return range_signal

        return None

    @classmethod
    def _trend_pullback_signal(
        cls,
        *,
        close: float,
        vwap: float,
        ema_50: float,
        ema_slope: float,
        macd: float,
        previous_macd: float,
        rsi: float,
        stoch_k: float,
        stoch_d: float,
        bullish_cross: bool,
        bearish_cross: bool,
        volume_ratio: float,
        candle: dict[str, float | bool],
        higher_trend_bias: int,
    ) -> StrategySignal | None:
        candidates = (
            cls.score_trend_scalp(
                direction=1,
                close=close,
                vwap=vwap,
                ema_50=ema_50,
                ema_slope=ema_slope,
                macd=macd,
                previous_macd=previous_macd,
                rsi=rsi,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                exact_cross=bullish_cross,
                volume_ratio=volume_ratio,
                candle_confirmed=bool(candle["bullish_confirmation"]),
                higher_trend_bias=higher_trend_bias,
            ),
            cls.score_trend_scalp(
                direction=-1,
                close=close,
                vwap=vwap,
                ema_50=ema_50,
                ema_slope=ema_slope,
                macd=macd,
                previous_macd=previous_macd,
                rsi=rsi,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                exact_cross=bearish_cross,
                volume_ratio=volume_ratio,
                candle_confirmed=bool(candle["bearish_confirmation"]),
                higher_trend_bias=higher_trend_bias,
            ),
        )
        direction, score = max(candidates, key=lambda candidate: candidate[1])
        if score < 0.50:
            return None

        side = "bull" if direction == 1 else "bear"
        return StrategySignal(
            direction,
            cls.trend_scalp_confidence(score),
            f"scalp_pullback_{side}+scalp_score_{score:.2f}",
            cls.name,
        )

    @classmethod
    def _range_reversion_signal(
        cls,
        *,
        percent_b: float,
        rsi: float,
        stoch_k: float,
        stoch_d: float,
        bullish_cross: bool,
        bearish_cross: bool,
        volume_ratio: float,
        candle: dict[str, float | bool],
    ) -> StrategySignal | None:
        candidates = (
            cls.score_range_scalp(
                direction=1,
                percent_b=percent_b,
                rsi=rsi,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                exact_cross=bullish_cross,
                volume_ratio=volume_ratio,
                rejection=bool(candle["bullish_rejection"]),
            ),
            cls.score_range_scalp(
                direction=-1,
                percent_b=percent_b,
                rsi=rsi,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                exact_cross=bearish_cross,
                volume_ratio=volume_ratio,
                rejection=bool(candle["bearish_rejection"]),
            ),
        )
        direction, score = max(candidates, key=lambda candidate: candidate[1])
        if score < 0.72:
            return None

        side = "low" if direction == 1 else "high"
        strength = min(0.60 + (score - 0.72) * 2.00, 0.95)
        return StrategySignal(
            direction,
            strength,
            f"scalp_range_{side}+scalp_score_{score:.2f}",
            cls.name,
        )

    @staticmethod
    def trend_scalp_confidence(score: float) -> float:
        return min(0.55 + max(score - 0.65, 0.0) * 2.00, 0.95)

    @staticmethod
    def score_trend_scalp(
        *,
        direction: int,
        close: float,
        vwap: float,
        ema_50: float,
        ema_slope: float,
        macd: float,
        previous_macd: float,
        rsi: float,
        stoch_k: float,
        stoch_d: float,
        exact_cross: bool,
        volume_ratio: float,
        candle_confirmed: bool,
        higher_trend_bias: int = 0,
    ) -> tuple[int, float]:
        if higher_trend_bias not in (0, direction):
            return direction, 0.0

        close_vs_ema = (close - ema_50) / ema_50 if ema_50 > 0 else 0.0
        close_vs_vwap = (close - vwap) / vwap if vwap > 0 else 0.0

        slope_aligned = ema_slope * direction > 0
        macd_aligned = macd * direction >= 0
        macd_improving = (macd - previous_macd) * direction > 0
        rsi_aligned = 45 <= rsi <= 68 if direction == 1 else 32 <= rsi <= 55
        stoch_turning = (
            stoch_k > stoch_d and stoch_k <= 75
            if direction == 1
            else stoch_k < stoch_d and stoch_k >= 25
        )
        rsi_reversal = rsi <= 48 if direction == 1 else rsi >= 52

        pullback_near_ema = 0 <= close_vs_ema * direction <= 0.004
        pullback_at_vwap = 0 <= close_vs_vwap * direction <= 0.003
        extended = close_vs_vwap * direction > 0.005

        structure_confirmed = (
            pullback_near_ema or (pullback_at_vwap and (slope_aligned or macd_aligned))
        ) and not extended
        momentum_confirmed = stoch_turning or macd_improving

        if not (
            structure_confirmed
            and slope_aligned
            and momentum_confirmed
            and candle_confirmed
            and volume_ratio >= 1.0
        ):
            return direction, 0.0

        score = sum(
            (
                0.22 if pullback_near_ema else 0.10 if pullback_at_vwap else 0.0,
                0.14 if slope_aligned else 0.0,
                0.12 if macd_aligned else 0.0,
                0.08 if macd_improving else 0.0,
                0.10 if rsi_aligned else 0.0,
                0.10 if rsi_reversal or exact_cross else 0.0,
                0.12 if stoch_turning else 0.0,
                0.06 if volume_ratio >= 1.0 else 0.0,
                0.06 if candle_confirmed else 0.0,
            )
        )
        return direction, min(score, 1.0)

    @staticmethod
    def score_range_scalp(
        *,
        direction: int,
        percent_b: float,
        rsi: float,
        stoch_k: float,
        stoch_d: float,
        exact_cross: bool,
        volume_ratio: float,
        rejection: bool,
    ) -> tuple[int, float]:
        edge_confirmed = percent_b <= 0.20 if direction == 1 else percent_b >= 0.80
        rsi_confirmed = rsi <= 42 if direction == 1 else rsi >= 58
        stoch_aligned = stoch_k > stoch_d if direction == 1 else stoch_k < stoch_d
        if not edge_confirmed or not rejection or not (rsi_confirmed or stoch_aligned):
            return direction, 0.0

        score = sum(
            (
                0.27 if edge_confirmed else 0.0,
                0.17 if rsi_confirmed else 0.0,
                0.16 if stoch_aligned else 0.0,
                0.08 if exact_cross else 0.0,
                0.10 if volume_ratio >= 0.75 else 0.0,
                0.04 if volume_ratio >= 1.10 else 0.0,
                0.26 if rejection else 0.0,
            )
        )
        return direction, min(score, 1.0)
