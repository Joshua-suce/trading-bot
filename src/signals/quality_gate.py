from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyQuality:
    accepted: bool
    score: float
    reason: str
    metrics: dict[str, float | int | str]


class StrategyQualityGate:
    def __init__(
        self,
        *,
        min_score: float = 0.68,
        min_adx: float = 18.0,
        min_volume_ratio: float = 0.70,
        min_atr_pct: float = 0.0005,
        max_atr_pct: float = 0.025,
        max_ema_extension_atr: float = 2.0,
        range_max_adx: float = 25.0,
        reversal_min_score: float = 0.75,
        breakout_min_volume_ratio: float = 1.0,
        breakout_max_extension_atr: float = 3.0,
        breakout_min_body_ratio: float = 0.50,
        breakout_min_close_location: float = 0.75,
        rejection_min_wick_ratio: float = 0.25,
    ) -> None:
        self.min_score = min_score
        self.min_adx = min_adx
        self.min_volume_ratio = min_volume_ratio
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.max_ema_extension_atr = max_ema_extension_atr
        self.range_max_adx = range_max_adx
        self.reversal_min_score = reversal_min_score
        self.breakout_min_volume_ratio = breakout_min_volume_ratio
        self.breakout_max_extension_atr = breakout_max_extension_atr
        self.breakout_min_body_ratio = breakout_min_body_ratio
        self.breakout_min_close_location = breakout_min_close_location
        self.rejection_min_wick_ratio = rejection_min_wick_ratio

    def evaluate(  # noqa: C901
        self,
        df: pd.DataFrame,
        direction: int,
        *,
        timeframe: str,
        higher_timeframe_regime: int | None = None,
        strategy: str = "trend",
        signal_source: str = "",
    ) -> StrategyQuality:
        if df.empty or direction not in {-1, 1}:
            return StrategyQuality(False, 0.0, "invalid signal context", {})

        row = df.iloc[-1]
        close = self._number(row.get("close"))
        ema_50 = self._number(row.get("ema_50"))
        ema_200 = self._number(row.get("ema_200"))
        ema_50_slope = self._number(row.get("ema_50_slope"))
        adx = self._number(row.get("adx"))
        plus_di = self._number(row.get("plus_di"))
        minus_di = self._number(row.get("minus_di"))
        rsi = self._number(row.get("rsi_14"), 50.0)
        atr = self._number(row.get("atr"))
        atr_pct = self._number(
            row.get("atr_pct"),
            atr / close if close > 0 else 0.0,
        )
        volume_ratio = self._number(row.get("vol_ratio"), 1.0)
        trend_regime = int(self._number(row.get("trend_regime")))
        macd_raw = row.get("macd_hist")
        macd_hist = self._number(macd_raw)
        candle = self._candle_context(row, direction)

        metrics: dict[str, float | int | str] = {
            "timeframe": timeframe,
            "strategy": strategy,
            "signal_source": signal_source,
            "trend_regime": trend_regime,
            "higher_timeframe_regime": higher_timeframe_regime or 0,
            "adx": adx,
            "rsi": rsi,
            "atr_pct": atr_pct,
            "volume_ratio": volume_ratio,
            "candle_body_ratio": candle["body_ratio"],
            "candle_close_location": candle["close_location"],
            "candle_direction_aligned": int(candle["direction_aligned"]),
            "candle_rejection_confirmed": int(candle["rejection_confirmed"]),
        }
        if close <= 0 or atr <= 0:
            return StrategyQuality(False, 0.0, "invalid price or ATR", metrics)
        extension_atr = abs(close - ema_50) / atr
        metrics["extension_atr"] = extension_atr
        if not self.min_atr_pct <= atr_pct <= self.max_atr_pct:
            return StrategyQuality(
                False,
                0.0,
                "volatility outside configured regime",
                metrics,
            )
        if volume_ratio < self.min_volume_ratio:
            return StrategyQuality(False, 0.0, "volume participation too low", metrics)
        if strategy == "countertrend":
            return self._evaluate_countertrend(row, direction, metrics)
        if strategy == "scalp":
            return self._evaluate_scalp(
                row,
                direction,
                timeframe,
                metrics,
                higher_timeframe_regime=higher_timeframe_regime,
                signal_source=signal_source,
            )
        if strategy == "range":
            return self._evaluate_range(row, direction, metrics)
        if strategy == "reversal":
            return self._evaluate_reversal(row, direction, metrics)
        if strategy == "transition":
            return self._evaluate_transition(row, direction, metrics)
        if adx < self.min_adx:
            return StrategyQuality(False, 0.0, "trend strength too low", metrics)
        if trend_regime == -direction:
            return StrategyQuality(
                False,
                0.0,
                "local trend regime opposes signal",
                metrics,
            )
        if strategy == "breakout" and volume_ratio < max(
            self.min_volume_ratio,
            self.breakout_min_volume_ratio,
        ):
            return StrategyQuality(
                False,
                0.0,
                "breakout volume confirmation too low",
                metrics,
            )
        if strategy == "breakout" and extension_atr > self.breakout_max_extension_atr:
            return StrategyQuality(
                False,
                0.0,
                "breakout price extension too high",
                metrics,
            )
        if strategy == "breakout" and (
            candle["body_ratio"] < self.breakout_min_body_ratio
            or (
                direction == 1
                and candle["close_location"] < self.breakout_min_close_location
            )
            or (
                direction == -1
                and candle["close_location"] > 1.0 - self.breakout_min_close_location
            )
            or not candle["direction_aligned"]
        ):
            return StrategyQuality(
                False,
                0.0,
                "breakout candle structure too weak",
                metrics,
            )
        if timeframe in {"1m", "3m", "5m", "15m", "30m"}:
            if higher_timeframe_regime is None:
                return StrategyQuality(
                    False,
                    0.0,
                    "waiting for 1h trend context",
                    metrics,
                )
            higher_timeframe_opposes = higher_timeframe_regime == -direction
            if higher_timeframe_opposes:
                return StrategyQuality(
                    False,
                    0.0,
                    "1h trend regime disagrees",
                    metrics,
                )
            if strategy in {"trend", "transition"} and higher_timeframe_regime == 0:
                return StrategyQuality(
                    False,
                    0.0,
                    "1h trend regime is not aligned",
                    metrics,
                )

        macd_valid = pd.notna(macd_raw)
        trend_aligned = (
            close > ema_50 > ema_200 if direction == 1 else close < ema_50 < ema_200
        )
        slope_aligned = ema_50_slope * direction > 0
        di_aligned = plus_di > minus_di if direction == 1 else minus_di > plus_di
        momentum_aligned = macd_valid and macd_hist * direction > 0
        rsi_aligned = 45 <= rsi <= 68 if direction == 1 else 32 <= rsi <= 55
        not_extended = extension_atr <= self.max_ema_extension_atr
        candle_aligned = bool(candle["direction_aligned"])

        checks = {
            "trend_alignment": trend_aligned,
            "ema_slope": slope_aligned,
            "directional_movement": di_aligned,
            "macd_momentum": momentum_aligned,
            "rsi_location": rsi_aligned,
            "not_extended": not_extended,
            "candle_alignment": candle_aligned,
        }
        weights = {
            "trend_alignment": 0.22,
            "ema_slope": 0.13,
            "directional_movement": 0.18,
            "macd_momentum": 0.14,
            "rsi_location": 0.13,
            "not_extended": 0.10,
            "candle_alignment": 0.10,
        }
        score = sum(weights[name] for name, passed in checks.items() if passed)
        metrics.update(
            {
                "passed_checks": sum(checks.values()),
            }
        )
        failed = [name for name, passed in checks.items() if not passed]
        if score < self.min_score:
            return StrategyQuality(
                False,
                round(score, 4),
                f"quality score below threshold; failed={','.join(failed)}",
                metrics,
            )
        return StrategyQuality(
            True,
            round(score, 4),
            f"{strategy} quality confirmed",
            metrics,
        )

    def _evaluate_range(
        self,
        row: pd.Series,
        direction: int,
        metrics: dict[str, float | int | str],
    ) -> StrategyQuality:
        adx = self._number(row.get("adx"))
        trend_regime = int(self._number(row.get("trend_regime")))
        rsi = self._number(row.get("rsi_14"), 50.0)
        percent_b = self._number(row.get("bb_percent_b"), 0.5)
        macd_raw = row.get("macd_hist")
        macd_hist = self._number(macd_raw)
        if trend_regime != 0 or adx > self.range_max_adx:
            return StrategyQuality(False, 0.0, "market is not range-bound", metrics)

        band_edge = percent_b <= 0.25 if direction == 1 else percent_b >= 0.75
        rsi_edge = rsi <= 42 if direction == 1 else rsi >= 58
        macd_valid = pd.notna(macd_raw)
        momentum_turn = macd_valid and macd_hist * direction >= 0
        candle = self._candle_context(row, direction)
        rejection = bool(candle["rejection_confirmed"])
        if not rejection:
            metrics.update(
                {
                    "bb_percent_b": percent_b,
                    "range_band_edge": int(band_edge),
                    "range_rsi_edge": int(rsi_edge),
                    "range_momentum_turn": int(momentum_turn),
                    "range_rejection_candle": 0,
                }
            )
            return StrategyQuality(
                False,
                0.0,
                "range rejection candle not confirmed",
                metrics,
            )
        score = (
            0.35 * band_edge + 0.25 * rsi_edge + 0.15 * momentum_turn + 0.25 * rejection
        )
        metrics.update(
            {
                "bb_percent_b": percent_b,
                "range_band_edge": int(band_edge),
                "range_rsi_edge": int(rsi_edge),
                "range_momentum_turn": int(momentum_turn),
                "range_rejection_candle": int(rejection),
            }
        )
        if score < self.min_score:
            return StrategyQuality(
                False,
                round(score, 4),
                "range mean-reversion quality below threshold",
                metrics,
            )
        return StrategyQuality(
            True,
            round(score, 4),
            "range mean-reversion quality confirmed",
            metrics,
        )

    def _evaluate_reversal(
        self,
        row: pd.Series,
        direction: int,
        metrics: dict[str, float | int | str],
    ) -> StrategyQuality:
        trend_regime = int(self._number(row.get("trend_regime")))
        adx = self._number(row.get("adx"))
        rsi = self._number(row.get("rsi_14"), 50.0)
        plus_di = self._number(row.get("plus_di"))
        minus_di = self._number(row.get("minus_di"))
        macd_raw = row.get("macd_hist")
        macd_hist = self._number(macd_raw)
        volume_ratio = self._number(row.get("vol_ratio"), 1.0)
        if trend_regime != -direction:
            return StrategyQuality(
                False,
                0.0,
                "no established trend to reverse",
                metrics,
            )
        if adx < self.min_adx:
            return StrategyQuality(
                False,
                0.0,
                "reversal trend strength too low",
                metrics,
            )

        exhausted = rsi <= 35 if direction == 1 else rsi >= 65
        momentum_turn = pd.notna(macd_raw) and macd_hist * direction > 0
        di_turn = plus_di > minus_di if direction == 1 else minus_di > plus_di
        volume_confirmed = volume_ratio >= 1.0
        candle = self._candle_context(row, direction)
        reversal_candle = bool(candle["rejection_confirmed"])
        if not reversal_candle:
            metrics["reversal_candle_confirmed"] = 0
            return StrategyQuality(
                False,
                0.0,
                "reversal candle not confirmed",
                metrics,
            )
        score = (
            0.30 * exhausted
            + 0.25 * momentum_turn
            + 0.15 * di_turn
            + 0.10 * volume_confirmed
            + 0.20 * reversal_candle
        )
        metrics.update(
            {
                "reversal_exhausted": int(exhausted),
                "reversal_momentum_turn": int(momentum_turn),
                "reversal_di_turn": int(di_turn),
                "reversal_volume_confirmed": int(volume_confirmed),
                "reversal_candle_confirmed": int(reversal_candle),
            }
        )
        threshold = self.reversal_min_score
        if score < threshold:
            return StrategyQuality(
                False,
                round(score, 4),
                "reversal confirmation below threshold",
                metrics,
            )
        return StrategyQuality(
            True,
            round(score, 4),
            "reversal exhaustion and momentum shift confirmed",
            metrics,
        )

    def _evaluate_transition(
        self,
        row: pd.Series,
        direction: int,
        metrics: dict[str, float | int | str],
    ) -> StrategyQuality:
        adx = self._number(row.get("adx"))
        trend_regime = int(self._number(row.get("trend_regime")))
        rsi = self._number(row.get("rsi_14"), 50.0)
        ema_50_slope = self._number(row.get("ema_50_slope"))
        close = self._number(row.get("close"))
        ema_50 = self._number(row.get("ema_50"))
        macd_raw = row.get("macd_hist")
        macd_hist = self._number(macd_raw)
        plus_di = self._number(row.get("plus_di"))
        minus_di = self._number(row.get("minus_di"))
        atr = self._number(row.get("atr"))
        extension_atr = abs(close - ema_50) / atr if atr > 0 else 0.0
        macd_valid = pd.notna(macd_raw)
        if trend_regime != direction:
            return StrategyQuality(
                False,
                0.0,
                "no established trend to transition from",
                metrics,
            )
        if adx < self.min_adx:
            return StrategyQuality(
                False,
                0.0,
                "transition trend strength too low",
                metrics,
            )

        slope_weakening = abs(ema_50_slope) < 0.5
        price_near_ema = extension_atr <= 1.0
        momentum_fading = macd_valid and macd_hist * direction <= 0
        di_narrowing = plus_di > minus_di if direction == 1 else minus_di > plus_di
        di_cross = plus_di < minus_di if direction == 1 else minus_di < plus_di
        rsi_mid = 40 <= rsi <= 60
        candle = self._candle_context(row, direction)
        rejection = bool(candle["rejection_confirmed"])
        transition_setup = slope_weakening and price_near_ema
        if not transition_setup:
            return StrategyQuality(
                False,
                0.0,
                "transition conditions not met: slope or extension too strong",
                metrics,
            )
        score = (
            0.20 * slope_weakening
            + 0.15 * price_near_ema
            + 0.25 * momentum_fading
            + 0.15 * (di_narrowing or di_cross)
            + 0.10 * rsi_mid
            + 0.15 * rejection
        )
        metrics.update(
            {
                "transition_slope_weakening": int(slope_weakening),
                "transition_price_near_ema": int(price_near_ema),
                "transition_momentum_fading": int(momentum_fading),
                "transition_di_narrowing": int(di_narrowing),
                "transition_di_cross": int(di_cross),
                "transition_rsi_mid": int(rsi_mid),
                "transition_rejection": int(rejection),
                "extension_atr": extension_atr,
            }
        )
        if score < self.min_score:
            return StrategyQuality(
                False,
                round(score, 4),
                f"transition quality below threshold: score={score:.4f}",
                metrics,
            )
        return StrategyQuality(
            True,
            round(score, 4),
            "transition weakening trend confirmed",
            metrics,
        )

    def _evaluate_countertrend(
        self,
        row: pd.Series,
        direction: int,
        metrics: dict[str, float | int | str],
    ) -> StrategyQuality:
        adx = self._number(row.get("adx"))
        trend_regime = int(self._number(row.get("trend_regime")))
        rsi = self._number(row.get("rsi_14"), 50.0)
        close = self._number(row.get("close"))
        ema_50 = self._number(row.get("ema_50"))
        atr = self._number(row.get("atr"))
        volume_ratio = self._number(row.get("volume_ratio"))
        macd_raw = row.get("macd_hist")
        macd_hist = self._number(macd_raw)
        macd_valid = pd.notna(macd_raw)
        if adx > 0:
            extension_atr = abs(close - ema_50) / atr if atr > 0 else 0.0
        else:
            extension_atr = 0.0
        candle = self._candle_context(row, direction)
        rejection = bool(candle["rejection_confirmed"])
        long_wick_ratio = self._number(candle.get("wick_ratio", 0.0))
        if trend_regime == 0 or trend_regime == direction:
            return StrategyQuality(
                False,
                0.0,
                "no established trend to counter",
                metrics,
            )
        if adx < self.min_adx:
            return StrategyQuality(
                False,
                0.0,
                "countertrend trend strength too low",
                metrics,
            )
        overextended = extension_atr > 2.0
        if not overextended:
            return StrategyQuality(
                False,
                0.0,
                f"price not overextended: extension_atr={extension_atr:.2f}",
                metrics,
            )
        momentum_exhaust = macd_valid and macd_hist * direction < 0
        volume_climax = volume_ratio > 1.5
        rsi_extreme = (direction == 1 and rsi > 70) or (direction == -1 and rsi < 30)
        score = (
            0.20 * overextended
            + 0.20 * rejection
            + 0.15 * momentum_exhaust
            + 0.15 * volume_climax
            + 0.15 * rsi_extreme
            + 0.15 * min(long_wick_ratio / 0.5, 1.0)
        )
        metrics.update(
            {
                "countertrend_overextended": int(overextended),
                "countertrend_rejection": int(rejection),
                "countertrend_momentum_exhaust": int(momentum_exhaust),
                "countertrend_volume_climax": int(volume_climax),
                "countertrend_rsi_extreme": int(rsi_extreme),
                "extension_atr": extension_atr,
            }
        )
        if score < self.min_score:
            return StrategyQuality(
                False,
                round(score, 4),
                f"countertrend quality below threshold: score={score:.4f}",
                metrics,
            )
        return StrategyQuality(
            True,
            round(score, 4),
            "countertrend overextended rejection confirmed",
            metrics,
        )

    def _evaluate_scalp(
        self,
        row: pd.Series,
        direction: int,
        timeframe: str,
        metrics: dict[str, float | int | str],
        *,
        higher_timeframe_regime: int | None,
        signal_source: str,
    ) -> StrategyQuality:
        if timeframe not in {"1m", "3m"}:
            return StrategyQuality(
                False,
                0.0,
                "scalp strategy requires 1m or 3m timeframe",
                metrics,
            )
        close = self._number(row.get("close"))
        ema_50 = self._number(row.get("ema_50"))
        vwap = self._number(row.get("vwap"))
        rsi = self._number(row.get("rsi_14"), 50.0)
        stoch_k = self._number(row.get("stoch_k"), 50.0)
        stoch_d = self._number(row.get("stoch_d"), 50.0)
        macd_raw = row.get("macd_hist")
        macd_hist = self._number(macd_raw)
        volume_ratio = self._number(row.get("vol_ratio"))
        percent_b = self._number(row.get("bb_percent_b"), 0.5)
        ema_slope = self._number(row.get("ema_50_slope"))
        candle = self._candle_context(row, direction)

        is_range_scalp = signal_source.startswith("scalp_range_")
        if not is_range_scalp:
            if higher_timeframe_regime is None:
                return StrategyQuality(
                    False,
                    0.0,
                    "waiting for 1h trend context",
                    metrics,
                )
            if higher_timeframe_regime != direction:
                return StrategyQuality(
                    False,
                    0.0,
                    "1h trend regime is not aligned with trend scalp",
                    metrics,
                )
        if is_range_scalp:
            price_aligned = percent_b <= 0.25 if direction == 1 else percent_b >= 0.75
            oscillator_aligned = (
                stoch_k > stoch_d and rsi <= 45
                if direction == 1
                else stoch_k < stoch_d and rsi >= 55
            )
            momentum_aligned = (
                pd.notna(macd_raw) and macd_hist * direction >= 0
                or candle["rejection_confirmed"]
            )
            candle_confirmed = bool(candle["rejection_confirmed"])
        else:
            ema_aligned = (
                close > ema_50 if direction == 1
                else close < ema_50
            )
            vwap_aligned = (
                close > vwap if direction == 1
                else close < vwap
            )
            price_aligned = ema_aligned and vwap_aligned
            oscillator_aligned = (
                stoch_k > stoch_d and rsi <= 70
                if direction == 1
                else stoch_k < stoch_d and rsi >= 30
            )
            momentum_aligned = (
                pd.notna(macd_raw) and macd_hist * direction >= 0
                or ema_slope * direction > 0
            )
            candle_confirmed = bool(
                candle["direction_aligned"] or candle["rejection_confirmed"]
            )
        volume_confirmed = volume_ratio >= 0.85
        checks = {
            "price_alignment": price_aligned,
            "oscillator_alignment": oscillator_aligned,
            "momentum_alignment": momentum_aligned,
            "volume_confirmation": volume_confirmed,
            "candle_confirmation": candle_confirmed,
        }
        weights = {
            "price_alignment": 0.25,
            "oscillator_alignment": 0.25,
            "momentum_alignment": 0.15,
            "volume_confirmation": 0.15,
            "candle_confirmation": 0.20,
        }
        score = sum(weights[name] for name, passed in checks.items() if passed)
        metrics.update(
            {
                "scalp_price_aligned": int(price_aligned),
                "scalp_oscillator_aligned": int(oscillator_aligned),
                "scalp_momentum_aligned": int(momentum_aligned),
                "scalp_volume_confirmed": int(volume_confirmed),
                "scalp_candle_confirmed": int(candle_confirmed),
                "scalp_setup_type": "range" if is_range_scalp else "trend",
            }
        )
        required_structure = price_aligned and volume_confirmed
        if is_range_scalp:
            required_structure = required_structure and candle_confirmed
        else:
            required_structure = ema_aligned and volume_confirmed
        if not required_structure or score < max(self.min_score, 0.70):
            failed = [name for name, passed in checks.items() if not passed]
            return StrategyQuality(
                False,
                round(score, 4),
                f"scalp quality below threshold; failed={','.join(failed)}",
                metrics,
            )
        return StrategyQuality(
            True,
            round(score, 4),
            "scalp structure and momentum confirmed",
            metrics,
        )

    def _candle_context(
        self,
        row: pd.Series,
        direction: int,
    ) -> dict[str, float | bool]:
        close = self._number(row.get("close"))
        open_price = self._number(row.get("open"), close)
        high = self._number(row.get("high"), max(open_price, close))
        low = self._number(row.get("low"), min(open_price, close))
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        candle_range = max(high - low, 0.0)
        if candle_range <= 0:
            return {
                "body_ratio": 0.0,
                "close_location": 0.5,
                "direction_aligned": False,
                "rejection_confirmed": False,
            }
        body_ratio = abs(close - open_price) / candle_range
        close_location = min(max((close - low) / candle_range, 0.0), 1.0)
        lower_wick_ratio = (min(open_price, close) - low) / candle_range
        upper_wick_ratio = (high - max(open_price, close)) / candle_range
        direction_aligned = close > open_price if direction == 1 else close < open_price
        rejection_confirmed = (
            direction_aligned
            and (
                lower_wick_ratio >= self.rejection_min_wick_ratio
                if direction == 1
                else upper_wick_ratio >= self.rejection_min_wick_ratio
            )
            and (close_location >= 0.55 if direction == 1 else close_location <= 0.45)
        )
        return {
            "body_ratio": body_ratio,
            "close_location": close_location,
            "direction_aligned": direction_aligned,
            "rejection_confirmed": rejection_confirmed,
        }

    @staticmethod
    def _number(value, default: float = 0.0) -> float:
        if value is None or pd.isna(value):
            return default
        return float(value)
