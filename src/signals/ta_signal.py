from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.strategies.breakout import BreakoutStrategy
from src.strategies.countertrend import CountertrendStrategy
from src.strategies.range import RangeStrategy
from src.strategies.reversal import ReversalStrategy
from src.strategies.scalp import ScalpStrategy
from src.strategies.transition import TransitionStrategy
from src.strategies.trend import TrendStrategy


@dataclass
class TASignal:
    direction: int
    strength: float
    source: str
    strategy: str = "trend"


class TASignalList(list[TASignal]):
    def _best(self) -> TASignal | None:
        return max(self, key=lambda signal: signal.strength, default=None)

    @property
    def direction(self) -> int:
        signal = self._best()
        return signal.direction if signal else 0

    @property
    def strength(self) -> float:
        signal = self._best()
        return signal.strength if signal else 0.0

    @property
    def source(self) -> str:
        signal = self._best()
        return signal.source if signal else "neutral"

    @property
    def strategy(self) -> str:
        signal = self._best()
        return signal.strategy if signal else "trend"


class TechnicalSignal:
    def generate(  # noqa: C901
        self, df: pd.DataFrame, higher_trend_bias: int = 0
    ) -> list[TASignal]:
        if df.empty:
            return TASignalList()

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
            scalp = self._scalp_signal(
                last, prev, adx, candle, higher_trend_bias=higher_trend_bias
            )
            if scalp:
                return TASignalList([scalp])
            return TASignalList([TASignal(0, 0.0, "scalp_no_setup", "scalp")])

        structure_signal = self._swing_structure_signal(
            last,
            prev,
            trend_regime,
            adx,
            candle,
        )
        if structure_signal:
            signals.append(structure_signal)

        signals.extend(
            self._to_ta_signal(signal)
            for signal in TrendStrategy.classic_signals(
                last,
                prev,
                adx_strength=adx_strength,
                rsi=rsi,
                candle=candle,
            )
        )
        signals.extend(
            self._to_ta_signal(signal)
            for signal in ReversalStrategy.oscillator_signals(
                last,
                prev,
                rsi=rsi,
                strong_trend=strong_trend,
                trend_direction=trend_direction,
            )
        )
        signals.extend(
            self._to_ta_signal(signal)
            for signal in RangeStrategy.oscillator_signals(
                last,
                strong_trend=strong_trend,
                trend_direction=trend_direction,
                candle=candle,
            )
        )

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

        if len(df) > 1:
            breakout_signal = BreakoutStrategy.donchian_signal(
                last,
                prev,
                trend_regime=trend_regime,
                adx=adx,
                adx_strength=adx_strength,
                volume_ratio=volume_ratio,
                candle=candle,
            )
            if breakout_signal:
                signals.append(self._to_ta_signal(breakout_signal))

        if not signals:
            return TASignalList()

        signals = self._strongest_family_votes(signals)

        return TASignalList(
            self._produce_strategy_signals(
                signals,
                adx_strength,
                vol_conf,
                trend_regime,
                adx,
                trend_direction,
            )
        )

    @classmethod
    def _produce_strategy_signals(  # noqa: C901
        cls,
        signals,
        adx_strength,
        vol_conf,
        trend_regime,
        adx,
        trend_direction,
    ) -> list[TASignal]:
        strategy_map: dict[str, list[TASignal]] = {}
        for signal in signals:
            strategies = cls._signal_strategies(
                signal.source,
                signal.direction,
                trend_regime,
            )
            for strategy in strategies:
                strategy_map.setdefault(strategy, []).append(signal)

        result: list[TASignal] = []
        for strategy_name, group_signals in strategy_map.items():
            if not cls._strategy_group_actionable(strategy_name, group_signals):
                continue
            direction, strength = cls._compute_group_signal(
                group_signals, adx_strength, vol_conf, trend_direction, trend_regime
            )
            if direction != 0:
                source = "+".join(s.source for s in group_signals[:3])
                result.append(
                    TASignal(direction, min(strength, 1.0), source, strategy_name)
                )

        # Always add countertrend if not already present and regime has clear direction
        if trend_regime != 0 and not any(s.strategy == "countertrend" for s in result):
            ct_dir = -trend_regime
            ct_group = [s for s in signals if s.direction == ct_dir]
            if ct_group:
                direction, strength = cls._compute_group_signal(
                    ct_group, adx_strength, vol_conf, trend_direction, trend_regime
                )
                if direction != 0:
                    source = "+".join(s.source for s in ct_group[:3])
                    result.append(
                        TASignal(
                            direction,
                            min(
                                strength * CountertrendStrategy.confidence_multiplier(),
                                1.0,
                            ),
                            source,
                            CountertrendStrategy.name,
                        )
                    )

        families = {cls._signal_family(signal.source) for signal in signals}
        if not result and len(families) >= 2:
            direction, strength = cls._compute_group_signal(
                signals, adx_strength, vol_conf, trend_direction, trend_regime
            )
            if direction != 0:
                source = "+".join(s.source for s in signals[:3])
                result.append(
                    TASignal(
                        direction,
                        min(strength * TransitionStrategy.fallback_multiplier(), 1.0),
                        source,
                        TransitionStrategy.name,
                    )
                )

        return result

    @classmethod
    def _strategy_group_actionable(
        cls, strategy_name: str, group_signals: list[TASignal]
    ) -> bool:
        if len({cls._signal_family(signal.source) for signal in group_signals}) >= 2:
            return True
        strong_prefixes = (
            "trend_structure_",
            "range_structure_",
            "reversal_structure_",
            "donchian_breakout_",
            "trend_pullback_",
            "fib_",
        )
        return any(
            signal.source.startswith(strong_prefixes) for signal in group_signals
        )

    @staticmethod
    def _signal_strategies(source: str, direction: int, trend_regime: int) -> list[str]:
        s = []
        if TrendStrategy.claims_source(source):
            s.append("trend")
        if BreakoutStrategy.claims_source(source):
            s.append("breakout")
        if RangeStrategy.claims_source(source):
            s.append("range")
        if ReversalStrategy.claims_source(source):
            s.append("reversal")
        if CountertrendStrategy.claims_signal(direction, trend_regime):
            s.append("countertrend")
        if TransitionStrategy.claims_source(source):
            s.append("transition")
        return s

    @classmethod
    def _compute_group_signal(
        cls,
        group_signals: list[TASignal],
        adx_strength: float,
        vol_conf: float,
        trend_direction: int = 0,
        trend_regime: int = 0,
    ) -> tuple[int, float]:
        weights = np.array([s.strength for s in group_signals])
        directions = np.array([s.direction for s in group_signals])
        total_w = weights.sum()
        if total_w == 0:
            return 0, 0.0
        weighted_dir = (weights * directions).sum() / total_w
        net_conviction = abs(weighted_dir)
        avg_strength = weights.mean()
        proposed_direction = 1 if weighted_dir > 0 else -1 if weighted_dir < 0 else 0
        if adx_strength < 0.3:
            weighted_dir *= 0.7
        elif trend_direction != 0 and trend_direction != proposed_direction:
            # Signal opposes the immediate trend direction: require much more
            # conviction before treating it as actionable rather than noise.
            weighted_dir *= 0.3
        threshold = 0.18 if trend_regime == proposed_direction else 0.28
        direction = (
            1 if weighted_dir > threshold else -1 if weighted_dir < -threshold else 0
        )
        adx_factor = 0.5 + 0.5 * adx_strength
        vol_factor = 0.5 + 0.5 * vol_conf
        confidence = (
            0.35 * avg_strength
            + 0.25 * net_conviction
            + 0.20 * adx_factor
            + 0.20 * vol_factor
        )
        return direction, confidence

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

    @staticmethod
    def _to_ta_signal(signal) -> TASignal:
        return TASignal(
            signal.direction,
            signal.strength,
            signal.source,
            signal.strategy,
        )

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
            TrendStrategy.structure_score(last, prev, 1, trend_regime, adx, candle),
            TrendStrategy.structure_score(last, prev, -1, trend_regime, adx, candle),
            RangeStrategy.structure_score(last, prev, 1, trend_regime, adx, candle),
            RangeStrategy.structure_score(last, prev, -1, trend_regime, adx, candle),
            ReversalStrategy.structure_score(last, prev, 1, trend_regime, adx, candle),
            ReversalStrategy.structure_score(last, prev, -1, trend_regime, adx, candle),
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
    def _fibonacci_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        trend_regime: int,
        adx_strength: float,
    ) -> TASignal | None:
        signal = TrendStrategy.fibonacci_signal(last, prev, trend_regime, adx_strength)
        return cls._to_ta_signal(signal) if signal else None

    @classmethod
    def _trend_pullback_signal(
        cls,
        last: pd.Series,
        prev: pd.Series,
        trend_regime: int,
        adx_strength: float,
        candle: dict[str, float | bool],
    ) -> TASignal | None:
        signal = TrendStrategy.pullback_signal(
            last,
            prev,
            trend_regime,
            adx_strength,
            candle,
        )
        return cls._to_ta_signal(signal) if signal else None

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
        signal = ScalpStrategy.generate(
            last,
            prev,
            adx=adx,
            candle=candle,
            higher_trend_bias=higher_trend_bias,
        )
        return cls._to_ta_signal(signal) if signal else None

    @staticmethod
    def _trend_scalp_confidence(score: float) -> float:
        return ScalpStrategy.trend_scalp_confidence(score)

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
        return ScalpStrategy.score_trend_scalp(
            direction=direction,
            close=close,
            vwap=vwap,
            ema_50=ema_50,
            ema_slope=ema_slope,
            macd=macd,
            previous_macd=previous_macd,
            rsi=rsi,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            exact_cross=exact_cross,
            volume_ratio=volume_ratio,
            candle_confirmed=candle_confirmed,
            higher_trend_bias=higher_trend_bias,
        )

    @staticmethod
    def _timeframe_minutes(df: pd.DataFrame) -> int | None:
        if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
            return None
        seconds = (df.index[-1] - df.index[-2]).total_seconds()
        if seconds <= 0:
            return None
        return round(seconds / 60)
