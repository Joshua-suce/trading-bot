from dataclasses import dataclass

import pandas as pd

from src.config import Settings, settings
from src.signals.aggregator import FinalSignal
from src.signals.quality_gate import StrategyQuality, StrategyQualityGate
from src.strategies import StrategyRegistry


@dataclass(frozen=True)
class SignalDecision:
    accepted: bool
    decision: str
    reason: str
    minimum_confidence: float
    quality: StrategyQuality | None = None


def strategy_minimum_confidence(
    strategy: str,
    *,
    cfg: Settings = settings,
) -> float:
    policy = StrategyRegistry(cfg).get(strategy)
    return max(cfg.min_confidence, policy.minimum_confidence)


def strategy_quality_gate_from_settings(
    *,
    cfg: Settings = settings,
) -> StrategyQualityGate:
    return StrategyQualityGate(
        min_score=cfg.strategy_quality_min_score,
        min_adx=cfg.strategy_min_adx,
        min_volume_ratio=cfg.strategy_min_volume_ratio,
        min_atr_pct=cfg.strategy_min_atr_pct,
        max_atr_pct=cfg.strategy_max_atr_pct,
        max_ema_extension_atr=cfg.strategy_max_ema_extension_atr,
        range_max_adx=cfg.strategy_range_max_adx,
        reversal_min_score=cfg.strategy_reversal_min_score,
        breakout_min_volume_ratio=cfg.strategy_breakout_min_volume_ratio,
        breakout_max_extension_atr=cfg.strategy_breakout_max_extension_atr,
        breakout_min_body_ratio=cfg.strategy_breakout_min_body_ratio,
        breakout_min_close_location=cfg.strategy_breakout_min_close_location,
        breakout_max_close_location=cfg.strategy_breakout_max_close_location,
        rejection_min_wick_ratio=cfg.strategy_rejection_min_wick_ratio,
    )


def filter_strategies_by_regime(
    strategies: list[str],
    market_type: str,
    allowed: list[str] | None = None,
) -> list[str]:
    if allowed is not None:
        return [s for s in strategies if s in allowed]
    return strategies


def evaluate_signal_decision(
    indicators: pd.DataFrame,
    signal: FinalSignal,
    *,
    timeframe: str,
    higher_timeframe_regime: int | None,
    quality_gate: StrategyQualityGate,
    cfg: Settings = settings,
    atr_mult_sl: float | None = None,
    reward_risk_ratio: float | None = None,
) -> SignalDecision:
    policy = StrategyRegistry(cfg).get(signal.strategy)
    minimum = strategy_minimum_confidence(signal.strategy, cfg=cfg)
    if not policy.supports(timeframe):
        return SignalDecision(
            False,
            "unsupported_scope",
            f"{policy.name} strategy does not support {timeframe}",
            minimum,
        )
    if signal.direction == 0 or signal.confidence < minimum:
        return SignalDecision(
            False,
            "skipped",
            signal.decision_reason or "entry threshold not met",
            minimum,
        )
    # Default to the strategy's own policy when a caller doesn't pass these
    # explicitly, so the reward:risk floor is enforced the same way in
    # backtesting as it is live - a caller that omitted them (e.g. replay.py)
    # previously skipped the reward:risk check entirely rather than checking
    # it against the strategy's real configured ratio.
    if atr_mult_sl is None:
        atr_mult_sl = policy.atr_stop_multiplier
    if reward_risk_ratio is None:
        reward_risk_ratio = policy.reward_risk_ratio
    quality = quality_gate.evaluate(
        indicators,
        signal.direction,
        timeframe=timeframe,
        higher_timeframe_regime=higher_timeframe_regime,
        strategy=signal.strategy,
        signal_source=signal.ta_source,
        atr_mult_sl=atr_mult_sl,
        reward_risk_ratio=reward_risk_ratio,
    )
    if not quality.accepted:
        return SignalDecision(
            False,
            "quality_rejected",
            quality.reason,
            minimum,
            quality,
        )
    return SignalDecision(
        True,
        "accepted",
        signal.decision_reason or quality.reason,
        minimum,
        quality,
    )
