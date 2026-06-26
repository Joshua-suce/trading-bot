from unittest.mock import MagicMock

import pandas as pd

from src.backtest.replay import DecisionReplayEngine
from src.config import Settings
from src.signals.aggregator import FinalSignal
from src.signals.decision_policy import (
    evaluate_signal_decision,
    strategy_minimum_confidence,
)
from src.signals.quality_gate import StrategyQuality


def test_decision_policy_uses_strategy_specific_threshold():
    cfg = Settings(
        _env_file=None,
        min_confidence=0.40,
        strategy_breakout_min_confidence=0.60,
    )
    quality_gate = MagicMock()
    signal = FinalSignal(1, 0.59, "breakout", 0.0, 0.0, strategy="breakout")

    decision = evaluate_signal_decision(
        pd.DataFrame({"close": [100.0]}),
        signal,
        timeframe="1h",
        higher_timeframe_regime=None,
        quality_gate=quality_gate,
        cfg=cfg,
    )

    assert strategy_minimum_confidence("breakout", cfg=cfg) == 0.60
    assert decision.accepted is False
    assert decision.decision == "skipped"
    quality_gate.evaluate.assert_not_called()


def test_replay_uses_only_closed_higher_timeframe_regime(monkeypatch):
    target_index = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
    target = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=target_index)
    higher_index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    higher = pd.DataFrame(
        {"trend_regime": [1, -1, -1]},
        index=higher_index,
    )
    monkeypatch.setattr(
        "src.backtest.replay.compute_all_indicators",
        lambda frame: frame.copy(),
    )
    aggregator = MagicMock()
    aggregator.generate.return_value = FinalSignal(
        1,
        1.0,
        "trend",
        0.0,
        0.0,
        strategy="trend",
    )
    quality_gate = MagicMock()
    quality_gate.evaluate.return_value = StrategyQuality(True, 1.0, "ok", {})
    replay = DecisionReplayEngine(
        aggregator,
        quality_gate=quality_gate,
        min_history=2,
    )

    observations = replay.evaluate(
        target,
        timeframe="30m",
        higher_timeframe_df=higher,
    )

    assert [row.higher_timeframe_regime for row in observations] == [1, 1]
    assert all(row.decision.accepted for row in observations)


def test_replay_records_quality_rejection(monkeypatch):
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=index)
    monkeypatch.setattr(
        "src.backtest.replay.compute_all_indicators",
        lambda data: data.copy(),
    )
    aggregator = MagicMock()
    aggregator.generate.return_value = FinalSignal(
        1,
        1.0,
        "trend",
        0.0,
        0.0,
        strategy="trend",
    )
    quality_gate = MagicMock()
    quality_gate.evaluate.return_value = StrategyQuality(
        False,
        0.4,
        "quality below threshold",
        {},
    )
    replay = DecisionReplayEngine(
        aggregator,
        quality_gate=quality_gate,
        min_history=2,
    )

    observations = replay.evaluate(frame, timeframe="1h")

    assert len(observations) == 1
    assert observations[0].decision.decision == "quality_rejected"
    assert observations[0].decision.reason == "quality below threshold"
