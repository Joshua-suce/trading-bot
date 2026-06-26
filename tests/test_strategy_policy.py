import pandas as pd
import pytest

from src.config import Settings
from src.signals.aggregator import FinalSignal
from src.signals.decision_policy import evaluate_signal_decision
from src.signals.quality_gate import StrategyQualityGate
from src.strategies import StrategyRegistry


def test_strategy_policies_are_independent_and_timeframe_scoped():
    registry = StrategyRegistry(Settings(_env_file=None))

    scalp = registry.get("scalp")
    trend = registry.get("trend")
    breakout = registry.get("breakout")

    assert scalp.supports("1m")
    assert not scalp.supports("5m")
    assert trend.supports("1h")
    assert not trend.supports("1m")
    assert breakout.reward_risk_ratio > trend.reward_risk_ratio
    assert scalp.risk_fraction < trend.risk_fraction


def test_holding_period_is_expressed_in_strategy_bars():
    registry = StrategyRegistry(Settings(_env_file=None))

    assert registry.get("trend").max_hold_seconds("5m") == 12 * 5 * 60
    assert registry.get("trend").max_hold_seconds("1h") == 12 * 60 * 60
    assert registry.get("range").max_hold_seconds("15m") == 4 * 15 * 60
    assert registry.get("scalp").max_hold_seconds("1m") == 15 * 60
    assert registry.get("scalp").max_hold_seconds("3m") == 10 * 3 * 60


def test_unsupported_timeframe_fails_closed():
    policy = StrategyRegistry(Settings(_env_file=None)).get("scalp")

    with pytest.raises(ValueError, match="does not support"):
        policy.max_hold_seconds("1h")


def test_required_edge_includes_fees_and_minimum_net_edge():
    cfg = Settings(
        _env_file=None,
        binance_api_url="https://demo-fapi.binance.com",
        scalp_estimated_round_trip_fee_bps=8.0,
        strategy_min_net_edge_bps=5.0,
    )

    trend = StrategyRegistry(cfg).get("trend")

    assert trend.estimated_round_trip_fee_bps == 16.0
    assert trend.required_target_edge_bps == 21.0


def test_signal_policy_rejects_strategy_on_unsupported_timeframe():
    signal = FinalSignal(1, 0.9, "scalp_pullback_bull", 0.0, 0.0, strategy="scalp")

    decision = evaluate_signal_decision(
        indicators=pd.DataFrame(),
        signal=signal,
        timeframe="1h",
        higher_timeframe_regime=1,
        quality_gate=StrategyQualityGate(),
        cfg=Settings(_env_file=None),
    )

    assert decision.accepted is False
    assert decision.decision == "unsupported_scope"
