from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TASignal:
    direction: int
    strength: float
    source: str


class TechnicalSignal:
    def generate(self, df: pd.DataFrame) -> TASignal:  # noqa: C901
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
        vol_conf = min(self._safe_float(last.get("vol_ratio"), 1.0) / 2, 1.0)

        if self._has(last, "ema_9", "ema_21") and self._has(prev, "ema_9", "ema_21"):
            if prev["ema_9"] <= prev["ema_21"] and last["ema_9"] > last["ema_21"]:
                signals.append(TASignal(1, 0.6, "ema_cross_bull"))
            elif prev["ema_9"] >= prev["ema_21"] and last["ema_9"] < last["ema_21"]:
                signals.append(TASignal(-1, 0.6, "ema_cross_bear"))

        if self._has(last, "ema_9", "ema_21", "macd_hist", "ema_21_slope"):
            macd_prev = self._safe_float(
                prev.get("macd_hist"), self._safe_float(last["macd_hist"])
            )
            macd_rising = last["macd_hist"] > macd_prev
            macd_falling = last["macd_hist"] < macd_prev
            if (
                last["ema_9"] > last["ema_21"]
                and last["ema_21_slope"] > 0
                and macd_rising
                and 48 <= rsi <= 72
            ):
                signals.append(
                    TASignal(1, 0.5 + 0.2 * adx_strength, "trend_continuation_bull")
                )
            elif (
                last["ema_9"] < last["ema_21"]
                and last["ema_21_slope"] < 0
                and macd_falling
                and 28 <= rsi <= 52
            ):
                signals.append(
                    TASignal(-1, 0.5 + 0.2 * adx_strength, "trend_continuation_bear")
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
            if last["close"] <= last["bb_lower"] and (
                not strong_trend or trend_direction >= 0
            ):
                signals.append(TASignal(1, 0.45, "bb_lower_bounce"))
            elif last["close"] >= last["bb_upper"] and (
                not strong_trend or trend_direction <= 0
            ):
                signals.append(TASignal(-1, 0.45, "bb_upper_reject"))
            elif last["bb_percent_b"] < 0.2 and not strong_trend:
                signals.append(TASignal(1, 0.25, "bb_oversold"))
            elif last["bb_percent_b"] > 0.8 and not strong_trend:
                signals.append(TASignal(-1, 0.25, "bb_overbought"))

        if len(df) > 1 and self._has(last, "dc_upper", "dc_lower"):
            prev_upper = prev.get("dc_upper")
            prev_lower = prev.get("dc_lower")
            if (
                pd.notna(prev_upper)
                and last["close"] > prev_upper
                and trend_regime >= 0
                and vol_conf >= 0.45
            ):
                signals.append(
                    TASignal(1, 0.55 + 0.25 * adx_strength, "donchian_breakout_bull")
                )
            elif (
                pd.notna(prev_lower)
                and last["close"] < prev_lower
                and trend_regime <= 0
                and vol_conf >= 0.45
            ):
                signals.append(
                    TASignal(-1, 0.55 + 0.25 * adx_strength, "donchian_breakout_bear")
                )

        if not signals:
            return TASignal(0, 0.0, "neutral")

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
        strength = (
            avg_strength
            * net_conviction
            * (0.6 + 0.4 * adx_strength)
            * (0.7 + 0.3 * vol_conf)
        )
        if direction == 0:
            strength = 0.0

        return TASignal(
            direction=direction,
            strength=min(strength, 1.0),
            source=f"ta_{len(signals)}signals",
        )

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
