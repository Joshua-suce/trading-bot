from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TASignal:
    direction: int
    strength: float
    source: str
    strategy: str = "trend"


class TechnicalSignal:
    def generate(  # noqa: C901
        self, df: pd.DataFrame, higher_trend_bias: int = 0
    ) -> TASignal:
        if df.empty:
            return TASignal(0, 0.0, "neutral")

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        signals: list[TASignal] = []

        adx = self._safe_float(last.get("adx"), 0.0)
        adx_strength = min(adx / 50, 1.0)
        strong_trend = adx >= 25
        trend_direction = self._trend_direction(last)
        trend_regime = int(last.get("trend_regime", 0) or 0)
        rsi = self._safe_float(last.get("rsi_14"), 50.0)
        volume_ratio = self._safe_float(last.get("vol_ratio"), 1.0)
        vol_conf = min(volume_ratio / 2, 1.0)
        candle = self._candle_context(last)
        timeframe_minutes = self._timeframe_minutes(df)
        if timeframe_minutes in {1, 3}:
            scalp_signal = self._scalp_signal(
                last, prev, adx, candle, higher_trend_bias=higher_trend_bias
            )
            return scalp_signal or TASignal(0, 0.0, "scalp_no_setup", "scalp")

        structure_signal = self._swing_structure_signal(
            last,
            prev,
            trend_regime,
            adx,
            candle,
        )
        if structure_signal:
            signals.append(structure_signal)

        if self._has(last, "ema_50", "ema_200") and self._has(
            prev, "ema_50", "ema_200"
        ):
            if prev["ema_50"] <= prev["ema_200"] and last["ema_50"] > last["ema_200"]:
                signals.append(TASignal(1, 0.75, "golden_cross"))
            elif prev["ema_50"] >= prev["ema_200"] and last["ema_50"] < last["ema_200"]:
                signals.append(TASignal(-1, 0.75, "death_cross"))

        if self._has(last, "ema_50", "ema_200", "macd_hist", "ema_50_slope"):
            macd_prev = self._safe_float(
                prev.get("macd_hist"), self._safe_float(last["macd_hist"])
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
                    TASignal(
                        1,
                        0.55 + 0.2 * adx_strength,
                        "ema_50_200_continuation_bull",
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
                    TASignal(
                        -1,
                        0.55 + 0.2 * adx_strength,
                        "ema_50_200_continuation_bear",
                    )
                )

        if pd.notna(last.get("rsi_14")):
            if rsi < 30:
                strength = 0.55 if trend_direction >= 0 or not strong_trend else 0.25
                signals.append(TASignal(1, strength, "rsi_oversold"))
            elif rsi > 70:
                strength = 0.55 if trend_direction <= 0 or not strong_trend else 0.25
                signals.append(TASignal(-1, strength, "rsi_overbought"))
            elif rsi < 40 and not strong_trend:
                signals.append(TASignal(1, 0.3, "rsi_bullish"))
            elif rsi > 60 and not strong_trend:
                signals.append(TASignal(-1, 0.3, "rsi_bearish"))

        if self._has(last, "macd_hist") and self._has(prev, "macd_hist"):
            if last["macd_hist"] > 0 and prev["macd_hist"] <= 0:
                signals.append(TASignal(1, 0.55, "macd_bull_cross"))
            elif last["macd_hist"] < 0 and prev["macd_hist"] >= 0:
                signals.append(TASignal(-1, 0.55, "macd_bear_cross"))

        if self._has(last, "bb_upper", "bb_lower", "bb_percent_b"):
            if (
                last["close"] <= last["bb_lower"]
                and (not strong_trend or trend_direction >= 0)
                and candle["bullish_rejection"]
            ):
                signals.append(TASignal(1, 0.45, "bb_lower_bounce"))
            elif (
                last["close"] >= last["bb_upper"]
                and (not strong_trend or trend_direction <= 0)
                and candle["bearish_rejection"]
            ):
                signals.append(TASignal(-1, 0.45, "bb_upper_reject"))
            elif last["bb_percent_b"] < 0.2 and not strong_trend:
                signals.append(TASignal(1, 0.25, "bb_oversold"))
            elif last["bb_percent_b"] > 0.8 and not strong_trend:
                signals.append(TASignal(-1, 0.25, "bb_overbought"))

        fib_signal = self._fibonacci_signal(last, prev, trend_regime, adx_strength)
        if fib_signal:
            signals.append(fib_signal)

        pullback_signal = self._trend_pullback_signal(
            last,
            prev,
            trend_regime,
            adx_strength,
            candle,
        )
        if pullback_signal:
            signals.append(pullback_signal)

        if len(df) > 1 and self._has(last, "dc_upper", "dc_lower"):
            prev_upper = prev.get("dc_upper")
            prev_lower = prev.get("dc_lower")
            close = self._safe_float(last.get("close"))
            atr = self._safe_float(last.get("atr"), close * 0.01)
            breakout_buffer = max(atr * 0.10, close * 0.0005)
            if (
                pd.notna(prev_upper)
                and last["close"] > prev_upper + breakout_buffer
                and trend_regime >= 0
                and (trend_regime == 1 or adx >= 24)
                and volume_ratio >= 1.35
                and candle["bullish_breakout"]
            ):
                signals.append(
                    TASignal(1, 0.55 + 0.25 * adx_strength, "donchian_breakout_bull")
                )
            elif (
                pd.notna(prev_lower)
                and last["close"] < prev_lower - breakout_buffer
                and trend_regime <= 0
                and (trend_regime == -1 or adx >= 24)
                and volume_ratio >= 1.35
                and candle["bearish_breakout"]
            ):
                signals.append(
                    TASignal(-1, 0.55 + 0.25 * adx_strength, "donchian_breakout_bear")
                )

        if not signals:
            return TASignal(0, 0.0, "neutral")

        signals = self._strongest_family_votes(signals)
        weights = np.array([s.strength for s in signals])
        directions = np.array([s.direction for s in signals])
        total_w = weights.sum()
        if total_w == 0:
            return TASignal(0, 0.0, "neutral")

        weighted_dir = (weights * directions).sum() / total_w
        net_conviction = abs(weighted_dir)
        avg_strength = weights.mean()
        proposed_direction = 1 if weighted_dir > 0 else -1 if weighted_dir < 0 else 0

        if adx_strength < 0.3:
            weighted_dir *= 0.7
        elif trend_direction != 0 and trend_direction != proposed_direction:
            weighted_dir *= 0.3

        threshold = 0.18 if trend_regime == proposed_direction else 0.28
        direction = (
            1 if weighted_dir > threshold else -1 if weighted_dir < -threshold else 0
        )
        adx_factor = 0.5 + 0.5 * adx_strength
        vol_factor = 0.5 + 0.5 * vol_conf
        strength = (
            0.35 * avg_strength
            + 0.25 * net_conviction
            + 0.20 * adx_factor
            + 0.20 * vol_factor
        )
        if direction == 0:
            strength = 0.0

        matching = [signal for signal in signals if signal.direction == direction]
        strategy = self._strategy_for(
            matching,
            direction,
            trend_regime,
            adx,
        )
        source = "+".join(signal.source for signal in matching[:3])
        independent_families = {
            self._signal_family(signal.source) for signal in matching
        }
        if (
            direction != 0
            and strategy in {"trend", "range", "reversal"}
            and not self._has_standalone_structure(matching)
            and len(independent_families) < 1
        ):
            return TASignal(
                direction=0,
                strength=0.0,
                source=source or "insufficient_family_confirmation",
                strategy=strategy,
            )
        return TASignal(
            direction=direction,
            strength=min(strength, 1.0),
            source=source or f"ta_{len(signals)}signals",
            strategy=strategy,
        )

    @staticmethod
    def _has_standalone_structure(signals: list[TASignal]) -> bool:
        return any(
            signal.source.startswith(
                (
                    "trend_structure_",
                    "range_structure_",
                    "reversal_structure_",
                    "donchian_breakout_",
                    "trend_pullback_",
                    "fib_",
                )
            )
            and signal.strength >= 0.45
            for signal in signals
        )

    @classmethod
    def _strongest_family_votes(cls, signals: list[TASignal]) -> list[TASignal]:
        strongest: dict[tuple[int, str], TASignal] = {}
        for signal in signals:
            key = (signal.direction, cls._signal_family(signal.source))
            current = strongest.get(key)
            if current is None or signal.strength > current.strength:
                strongest[key] = signal
        return list(strongest.values())

    @staticmethod
    def _signal_family(source: str) -> str:
        if source.startswith(("golden_cross", "death_cross", "ema_")):
            return "trend"
        if source.startswith("fib_") or source.startswith("trend_pullback"):
            return "pullback"
        if source.startswith(("trend_structure_", "range_structure_")):
            return "structure"
        if source.startswith("reversal_structure_"):
            return "reversal"
        if source.startswith("rsi_"):
            return "rsi"
        if source.startswith("macd_"):
            return "macd"
        if source.startswith("bb_"):
            return "bands"
        if source.startswith("donchian_"):
            return "breakout"
        return source

    @staticmethod
    def _strategy_for(
        signals: list[TASignal],
        direction: int,
        trend_regime: int,
        adx: float,
    ) -> str:
        sources = {signal.source for signal in signals}
        if any(source.startswith("donchian_breakout") for source in sources):
            return "breakout"
        if any(source.startswith("trend_structure") for source in sources):
            return "trend"
        if any(source.startswith("range_structure") for source in sources):
            return "range"
        if any(source.startswith("reversal_structure") for source in sources):
            return "reversal"

        reversal_sources = {
            "rsi_oversold",
            "rsi_overbought",
            "macd_bull_cross",
            "macd_bear_cross",
            "bb_lower_bounce",
            "bb_upper_reject",
        }
        reversal_votes = len(sources & reversal_sources)
        if trend_regime == -direction:
            return "reversal" if reversal_votes >= 2 else "countertrend"

        range_sources = {
            "rsi_oversold",
            "rsi_overbought",
            "rsi_bullish",
            "rsi_bearish",
            "bb_lower_bounce",
            "bb_upper_reject",
            "bb_oversold",
            "bb_overbought",
        }
        if trend_regime == 0 and adx < 22 and sources & range_sources:
            return "range"
        if trend_regime == direction:
            return "trend"
        return "transition"

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        if pd.isna(value):
            return default
        return float(value)

    @staticmethod
    def _has(row: pd.Series, *columns: str) -> bool:
        return all(column in row and pd.notna(row.get(column)) for column in columns)

    @staticmethod
    def _trend_direction(row: pd.Series) -> int:
        if pd.notna(row.get("plus_di")) and pd.notna(row.get("minus_di")):
            if row["plus_di"] > row["minus_di"]:
                return 1
            if row["minus_di"] > row["plus_di"]:
                return -1
        return int(row.get("trend_regime", 0) or 0)

    @classmethod
    def _swing_structure_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        trend_regime: int,
        adx: float,
        candle: dict[str, float | bool],
    ) -> TASignal | None:
        if not cls._has(
            last,
            "close",
            "ema_50",
            "ema_200",
            "ema_50_slope",
            "macd_hist",
            "rsi_14",
            "vol_ratio",
            "bb_percent_b",
            "plus_di",
            "minus_di",
        ):
            return None

        candidates = [
            cls._score_trend_structure(last, prev, 1, trend_regime, adx, candle),
            cls._score_trend_structure(last, prev, -1, trend_regime, adx, candle),
            cls._score_range_structure(last, prev, 1, trend_regime, adx, candle),
            cls._score_range_structure(last, prev, -1, trend_regime, adx, candle),
            cls._score_reversal_structure(last, prev, 1, trend_regime, adx, candle),
            cls._score_reversal_structure(last, prev, -1, trend_regime, adx, candle),
        ]
        strategy, direction, score = max(candidates, key=lambda item: item[2])
        thresholds = {
            "trend": 0.72,
            "range": 0.60,
            "reversal": 0.65,
        }
        if score < thresholds[strategy]:
            return None
        strength = min(0.42 + score * 0.36, 0.78)
        side = "bull" if direction == 1 else "bear"
        return TASignal(
            direction,
            strength,
            f"{strategy}_structure_{side}+structure_score_{score:.2f}",
            strategy,
        )

    @classmethod
    def _score_trend_structure(
        cls,
        last: pd.Series,
        prev: pd.Series,
        direction: int,
        trend_regime: int,
        adx: float,
        candle: dict[str, float | bool],
    ) -> tuple[str, int, float]:
        close = cls._safe_float(last.get("close"))
        ema_50 = cls._safe_float(last.get("ema_50"))
        ema_200 = cls._safe_float(last.get("ema_200"))
        ema_slope = cls._safe_float(last.get("ema_50_slope"))
        macd = cls._safe_float(last.get("macd_hist"))
        previous_macd = cls._safe_float(prev.get("macd_hist"), macd)
        previous_high = cls._safe_float(prev.get("high"), close)
        previous_low = cls._safe_float(prev.get("low"), close)
        rsi = cls._safe_float(last.get("rsi_14"), 50.0)
        volume_ratio = cls._safe_float(last.get("vol_ratio"), 0.0)
        plus_di = cls._safe_float(last.get("plus_di"))
        minus_di = cls._safe_float(last.get("minus_di"))
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
            return "trend", direction, 0.0
        if not (trend_stack and slope_aligned and di_aligned and macd_aligned):
            return "trend", direction, 0.0
        if not candle_aligned:
            return "trend", direction, 0.0
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
        return "trend", direction, min(score, 1.0)

    @classmethod
    def _score_range_structure(
        cls,
        last: pd.Series,
        prev: pd.Series,
        direction: int,
        trend_regime: int,
        adx: float,
        candle: dict[str, float | bool],
    ) -> tuple[str, int, float]:
        if trend_regime != 0 or adx > 25:
            return "range", direction, 0.0
        percent_b = cls._safe_float(last.get("bb_percent_b"), 0.5)
        rsi = cls._safe_float(last.get("rsi_14"), 50.0)
        macd = cls._safe_float(last.get("macd_hist"))
        previous_macd = cls._safe_float(prev.get("macd_hist"), macd)
        volume_ratio = cls._safe_float(last.get("vol_ratio"), 0.0)
        band_edge = percent_b <= 0.22 if direction == 1 else percent_b >= 0.78
        rsi_edge = rsi <= 42 if direction == 1 else rsi >= 58
        momentum_turn = (macd - previous_macd) * direction > 0
        rejection = bool(
            candle["bullish_rejection"]
            if direction == 1
            else candle["bearish_rejection"]
        )
        if not band_edge or volume_ratio < 0.65:
            return "range", direction, 0.0
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
        return "range", direction, min(score, 1.0)

    @classmethod
    def _score_reversal_structure(
        cls,
        last: pd.Series,
        prev: pd.Series,
        direction: int,
        trend_regime: int,
        adx: float,
        candle: dict[str, float | bool],
    ) -> tuple[str, int, float]:
        if trend_regime != -direction or adx < 22:
            return "reversal", direction, 0.0
        rsi = cls._safe_float(last.get("rsi_14"), 50.0)
        macd = cls._safe_float(last.get("macd_hist"))
        previous_macd = cls._safe_float(prev.get("macd_hist"), macd)
        plus_di = cls._safe_float(last.get("plus_di"))
        minus_di = cls._safe_float(last.get("minus_di"))
        volume_ratio = cls._safe_float(last.get("vol_ratio"), 0.0)
        exhausted = rsi <= 34 if direction == 1 else rsi >= 66
        momentum_turn = (macd - previous_macd) * direction > 0
        di_turn = plus_di > minus_di if direction == 1 else minus_di > plus_di
        rejection = bool(
            candle["bullish_rejection"]
            if direction == 1
            else candle["bearish_rejection"]
        )
        if not exhausted or not rejection or volume_ratio < 0.70:
            return "reversal", direction, 0.0
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
        return "reversal", direction, min(score, 1.0)

    @classmethod
    def _fibonacci_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        trend_regime: int,
        adx_strength: float,
    ) -> TASignal | None:
        levels = ("fib_382", "fib_500", "fib_618")
        available = [
            (name, cls._safe_float(last.get(name)))
            for name in levels
            if pd.notna(last.get(name)) and cls._safe_float(last.get(name)) > 0
        ]
        if not available or trend_regime == 0:
            return None

        close = cls._safe_float(last.get("close"))
        previous_close = cls._safe_float(prev.get("close"), close)
        atr = cls._safe_float(last.get("atr"), close * 0.01)
        tolerance = max(atr * 0.25, close * 0.001)
        level_name, level = min(available, key=lambda item: abs(close - item[1]))
        previous_level = cls._safe_float(prev.get(level_name), level)

        if trend_regime == 1:
            reclaimed = previous_close <= previous_level and close > level
            held_support = (
                cls._safe_float(last.get("low"), close) <= level + tolerance
                and close >= level
                and close > previous_close
            )
            if reclaimed or held_support:
                return TASignal(
                    1,
                    0.5 + 0.2 * adx_strength,
                    f"{level_name}_bullish_confluence",
                )

        if trend_regime == -1:
            rejected = previous_close >= previous_level and close < level
            held_resistance = (
                cls._safe_float(last.get("high"), close) >= level - tolerance
                and close <= level
                and close < previous_close
            )
            if rejected or held_resistance:
                return TASignal(
                    -1,
                    0.5 + 0.2 * adx_strength,
                    f"{level_name}_bearish_confluence",
                )
        return None

    @classmethod
    def _trend_pullback_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        trend_regime: int,
        adx_strength: float,
        candle: dict[str, float | bool],
    ) -> TASignal | None:
        if trend_regime == 0 or not cls._has(last, "close", "ema_50", "atr"):
            return None
        close = cls._safe_float(last.get("close"))
        ema_50 = cls._safe_float(last.get("ema_50"))
        atr = cls._safe_float(last.get("atr"))
        volume_ratio = cls._safe_float(last.get("vol_ratio"), 0.0)
        previous_close = cls._safe_float(prev.get("close"), close)
        if atr <= 0 or abs(close - ema_50) / atr > 1.5 or volume_ratio < 0.85:
            return None
        if (
            trend_regime == 1
            and previous_close <= close
            and close >= ema_50
            and bool(candle["bullish_rejection"])
        ):
            return TASignal(
                1,
                0.58 + 0.17 * adx_strength,
                "trend_pullback_ema50_bull",
            )
        if (
            trend_regime == -1
            and previous_close >= close
            and close <= ema_50
            and bool(candle["bearish_rejection"])
        ):
            return TASignal(
                -1,
                0.58 + 0.17 * adx_strength,
                "trend_pullback_ema50_bear",
            )
        return None

    @classmethod
    def _candle_context(cls, row: pd.Series) -> dict[str, float | bool]:
        open_price = cls._safe_float(row.get("open"))
        high = cls._safe_float(row.get("high"))
        low = cls._safe_float(row.get("low"))
        close = cls._safe_float(row.get("close"))
        candle_range = max(high - low, 0.0)
        if candle_range <= 0:
            return {
                "body_ratio": 0.0,
                "close_location": 0.5,
                "bullish_confirmation": False,
                "bearish_confirmation": False,
                "bullish_rejection": False,
                "bearish_rejection": False,
                "bullish_breakout": False,
                "bearish_breakout": False,
            }
        body = abs(close - open_price)
        body_ratio = body / candle_range
        close_location = (close - low) / candle_range
        lower_wick_ratio = (min(open_price, close) - low) / candle_range
        upper_wick_ratio = (high - max(open_price, close)) / candle_range
        bullish = close > open_price
        bearish = close < open_price
        return {
            "body_ratio": body_ratio,
            "close_location": close_location,
            "bullish_confirmation": bullish
            and body_ratio >= 0.25
            and close_location >= 0.50,
            "bearish_confirmation": bearish
            and body_ratio >= 0.25
            and close_location <= 0.50,
            "bullish_rejection": close >= open_price
            and lower_wick_ratio >= 0.25
            and close_location >= 0.55,
            "bearish_rejection": close <= open_price
            and upper_wick_ratio >= 0.25
            and close_location <= 0.45,
            "bullish_breakout": bullish
            and body_ratio >= 0.50
            and close_location >= 0.75,
            "bearish_breakout": bearish
            and body_ratio >= 0.50
            and close_location <= 0.25,
        }

    @classmethod
    def _scalp_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        adx: float,
        candle: dict[str, float | bool],
        higher_trend_bias: int = 0,
    ) -> TASignal | None:
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
        if not cls._has(last, *required):
            return None
        close = cls._safe_float(last.get("close"))
        ema_50 = cls._safe_float(last.get("ema_50"))
        ema_slope = cls._safe_float(last.get("ema_50_slope"))
        vwap = cls._safe_float(last.get("vwap"))
        macd = cls._safe_float(last.get("macd_hist"))
        previous_macd = cls._safe_float(prev.get("macd_hist"), macd)
        rsi = cls._safe_float(last.get("rsi_14"), 50.0)
        stoch_k = cls._safe_float(last.get("stoch_k"), 50.0)
        stoch_d = cls._safe_float(last.get("stoch_d"), 50.0)
        previous_k = cls._safe_float(prev.get("stoch_k"), stoch_k)
        previous_d = cls._safe_float(prev.get("stoch_d"), stoch_d)
        volume_ratio = cls._safe_float(last.get("vol_ratio"), 0.0)
        percent_b = cls._safe_float(last.get("bb_percent_b"), 0.5)
        bullish_cross = previous_k <= previous_d and stoch_k > stoch_d
        bearish_cross = previous_k >= previous_d and stoch_k < stoch_d

        if adx >= 16 and volume_ratio >= 0.80:
            trend_candidates = (
                cls._score_trend_scalp(
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
                cls._score_trend_scalp(
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
            direction, score = max(trend_candidates, key=lambda candidate: candidate[1])
            if score >= 0.65:
                side = "bull" if direction == 1 else "bear"
                strength = cls._trend_scalp_confidence(score)
                return TASignal(
                    direction,
                    strength,
                    f"scalp_pullback_{side}+scalp_score_{score:.2f}",
                    "scalp",
                )

        if adx <= 20 and volume_ratio >= 0.75:
            range_candidates = (
                cls._score_range_scalp(
                    direction=1,
                    percent_b=percent_b,
                    rsi=rsi,
                    stoch_k=stoch_k,
                    stoch_d=stoch_d,
                    exact_cross=bullish_cross,
                    volume_ratio=volume_ratio,
                    rejection=bool(candle["bullish_rejection"]),
                ),
                cls._score_range_scalp(
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
            direction, score = max(range_candidates, key=lambda candidate: candidate[1])
            if score >= 0.72:
                side = "low" if direction == 1 else "high"
                strength = min(0.60 + (score - 0.72) * 2.00, 0.95)
                return TASignal(
                    direction,
                    strength,
                    f"scalp_range_{side}+scalp_score_{score:.2f}",
                    "scalp",
                )
        return None

    @staticmethod
    def _trend_scalp_confidence(score: float) -> float:
        return min(0.55 + max(score - 0.65, 0.0) * 2.00, 0.95)

    @staticmethod
    def _score_trend_scalp(
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
        ema_slope_val = ema_slope

        slope_aligned = ema_slope_val * direction > 0
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
    def _score_range_scalp(
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

    @staticmethod
    def _timeframe_minutes(df: pd.DataFrame) -> int | None:
        if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
            return None
        seconds = (df.index[-1] - df.index[-2]).total_seconds()
        if seconds <= 0:
            return None
        return round(seconds / 60)
