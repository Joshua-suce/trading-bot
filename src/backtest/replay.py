from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.config import settings
from src.indicators.compute import compute_all_indicators
from src.signals.aggregator import FinalSignal, SignalAggregator
from src.signals.decision_policy import (
    SignalDecision,
    evaluate_signal_decision,
    strategy_quality_gate_from_settings,
)
from src.signals.invocation import generate_with_context
from src.signals.quality_gate import StrategyQualityGate


@dataclass(frozen=True)
class ReplayObservation:
    timestamp: pd.Timestamp
    signal: FinalSignal
    decision: SignalDecision
    higher_timeframe_regime: int | None


@dataclass(frozen=True)
class ReplayResult:
    observations: list[ReplayObservation]
    backtest: BacktestResult


class DecisionReplayEngine:
    def __init__(
        self,
        aggregator: SignalAggregator,
        *,
        quality_gate: StrategyQualityGate | None = None,
        backtest_engine: BacktestEngine | None = None,
        min_history: int = 200,
    ) -> None:
        self.aggregator = aggregator
        self.quality_gate = quality_gate or strategy_quality_gate_from_settings()
        self.backtest_engine = backtest_engine or BacktestEngine(
            commission=(settings.scalp_effective_round_trip_fee_bps / 2 / 10_000)
        )
        self.min_history = max(min_history, 2)

    def run(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        higher_timeframe_df: pd.DataFrame | None = None,
        strategy: str | None = None,
    ) -> ReplayResult:
        observations = self.evaluate(
            df,
            timeframe=timeframe,
            higher_timeframe_df=higher_timeframe_df,
            strategy=strategy,
        )
        accepted = {
            observation.timestamp: observation
            for observation in observations
            if observation.decision.accepted
        }

        def replay_signal(frame: pd.DataFrame) -> dict[str, float | int | str]:
            observation = accepted.get(pd.Timestamp(frame.index[-1]))
            if observation is None:
                return {"direction": 0, "confidence": 0.0, "strategy": "unknown"}
            return {
                "direction": observation.signal.direction,
                "confidence": observation.signal.confidence,
                "strategy": observation.signal.strategy,
            }

        result = self.backtest_engine.run(
            df,
            replay_signal,
            symbol=symbol,
            timeframe=timeframe,
            entry_confidence=0.0,
            use_strategy_policy=True,
        )
        return ReplayResult(observations, result)

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        timeframe: str,
        higher_timeframe_df: pd.DataFrame | None = None,
        strategy: str | None = None,
    ) -> list[ReplayObservation]:
        if len(df) <= self.min_history:
            return []
        higher_regimes = self._closed_higher_timeframe_regimes(
            df.index,
            higher_timeframe_df,
        )
        # Every column compute_all_indicators produces is causal (rolling
        # windows / ewm / shift(N>=0) only look backward - verified across
        # src/indicators/*.py; the one non-causal computation, Ichimoku's
        # chikou span, is dead code compute_all_indicators never calls).
        # That means indicators for row i are unaffected by any row after i,
        # so computing the full series once here and slicing per iteration
        # below produces identical values to recomputing from scratch on
        # every growing window, at a fraction of the cost: this used to be
        # O(n^2) (every one of ~30 indicators recomputed on an ever-larger
        # window every bar, and then recomputed AGAIN inside the aggregator
        # since it was handed the raw window, not this precomputed frame),
        # which made anything past a few hundred bars impractically slow.
        full_indicators = compute_all_indicators(df)
        observations = []
        for end in range(self.min_history, len(df)):
            window_indicators = full_indicators.iloc[: end + 1]
            timestamp = pd.Timestamp(window_indicators.index[-1])
            higher_regime = higher_regimes.get(timestamp)
            signals = generate_with_context(
                self.aggregator,
                window_indicators,
                higher_regime or 0,
            )
            if strategy is not None:
                signals = [signal for signal in signals if signal.strategy == strategy]
            if not signals:
                continue
            signal = max(signals, key=lambda s: s.confidence)
            if signal.direction == 0:
                continue
            decision = evaluate_signal_decision(
                window_indicators,
                signal,
                timeframe=timeframe,
                higher_timeframe_regime=higher_regime,
                quality_gate=self.quality_gate,
            )
            observations.append(
                ReplayObservation(timestamp, signal, decision, higher_regime)
            )
        return observations

    @staticmethod
    def _closed_higher_timeframe_regimes(
        target_index: pd.Index,
        higher_timeframe_df: pd.DataFrame | None,
    ) -> dict[pd.Timestamp, int]:
        if higher_timeframe_df is None or higher_timeframe_df.empty:
            return {}
        indicators = compute_all_indicators(higher_timeframe_df)
        regimes = indicators["trend_regime"].shift(1).dropna()
        aligned = regimes.reindex(pd.DatetimeIndex(target_index), method="ffill")
        return {
            pd.Timestamp(timestamp): int(regime)
            for timestamp, regime in aligned.dropna().items()
        }
