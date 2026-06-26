from dataclasses import dataclass
from typing import Any

from src.config import Settings, settings


def _timeframe_seconds(timeframe: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    try:
        return int(timeframe[:-1]) * units[timeframe[-1].lower()]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Unsupported strategy timeframe: {timeframe}") from exc


@dataclass(frozen=True)
class StrategyPolicy:
    name: str
    allowed_timeframes: frozenset[str]
    minimum_confidence: float
    risk_fraction: float
    max_position_fraction: float
    atr_stop_multiplier: float
    min_stop_pct: float
    max_stop_pct: float
    reward_risk_ratio: float
    max_hold_bars: int
    estimated_round_trip_fee_bps: float
    minimum_net_edge_bps: float
    max_spread_bps: float
    walk_forward_min_trades: int
    walk_forward_min_profit_factor: float
    walk_forward_max_drawdown_pct: float
    hold_bars_by_timeframe: tuple[tuple[str, int], ...] = ()

    def supports(self, timeframe: str) -> bool:
        return timeframe in self.allowed_timeframes

    def max_hold_seconds(self, timeframe: str) -> int:
        if not self.supports(timeframe):
            raise ValueError(f"{self.name} does not support {timeframe}")
        hold_bars = dict(self.hold_bars_by_timeframe).get(
            timeframe,
            self.max_hold_bars,
        )
        return hold_bars * _timeframe_seconds(timeframe)

    @property
    def required_target_edge_bps(self) -> float:
        return self.estimated_round_trip_fee_bps + self.minimum_net_edge_bps


class StrategyRegistry:
    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self._policies = self._build(cfg)

    def get(self, strategy: str | None) -> StrategyPolicy:
        key = str(strategy or "trend").lower()
        return self._policies.get(key, self._policies["trend"])

    def all(self) -> tuple[StrategyPolicy, ...]:
        return tuple(self._policies.values())

    @staticmethod
    def _build(cfg: Settings) -> dict[str, StrategyPolicy]:
        swing = frozenset({"5m", "15m", "30m", "1h", "4h", "1d"})
        scalp = frozenset({"1m", "3m"})
        effective_fees = cfg.scalp_effective_round_trip_fee_bps
        common: dict[str, Any] = {
            "max_position_fraction": cfg.max_position_size,
            "min_stop_pct": cfg.min_stop_loss_pct,
            "max_stop_pct": cfg.max_stop_loss_pct,
            "estimated_round_trip_fee_bps": effective_fees,
            "minimum_net_edge_bps": cfg.strategy_min_net_edge_bps,
            "max_spread_bps": cfg.strategy_max_spread_bps,
            "walk_forward_min_trades": cfg.walk_forward_min_trades,
            "walk_forward_max_drawdown_pct": cfg.walk_forward_max_drawdown_pct,
        }
        return {
            "trend": StrategyPolicy(
                "trend",
                swing,
                cfg.strategy_trend_min_confidence,
                cfg.strategy_trend_risk_per_trade,
                atr_stop_multiplier=1.5,
                reward_risk_ratio=2.0,
                max_hold_bars=12,
                walk_forward_min_profit_factor=1.15,
                **common,
            ),
            "transition": StrategyPolicy(
                "transition",
                swing,
                cfg.strategy_transition_min_confidence,
                cfg.strategy_transition_risk_per_trade,
                atr_stop_multiplier=1.3,
                reward_risk_ratio=2.0,
                max_hold_bars=8,
                walk_forward_min_profit_factor=1.20,
                **common,
            ),
            "range": StrategyPolicy(
                "range",
                swing,
                cfg.strategy_range_min_confidence,
                cfg.strategy_range_risk_per_trade,
                atr_stop_multiplier=1.0,
                reward_risk_ratio=1.5,
                max_hold_bars=4,
                walk_forward_min_profit_factor=1.15,
                **common,
            ),
            "breakout": StrategyPolicy(
                "breakout",
                swing,
                cfg.strategy_breakout_min_confidence,
                cfg.strategy_breakout_risk_per_trade,
                atr_stop_multiplier=1.3,
                reward_risk_ratio=2.5,
                max_hold_bars=8,
                walk_forward_min_profit_factor=1.20,
                **common,
            ),
            "reversal": StrategyPolicy(
                "reversal",
                swing,
                cfg.strategy_reversal_min_confidence,
                cfg.strategy_reversal_risk_per_trade,
                atr_stop_multiplier=1.2,
                reward_risk_ratio=1.8,
                max_hold_bars=6,
                walk_forward_min_profit_factor=1.25,
                **common,
            ),
            "countertrend": StrategyPolicy(
                "countertrend",
                swing,
                cfg.strategy_countertrend_min_confidence,
                cfg.strategy_countertrend_risk_per_trade,
                atr_stop_multiplier=1.0,
                reward_risk_ratio=1.5,
                max_hold_bars=4,
                walk_forward_min_profit_factor=1.30,
                **common,
            ),
            "scalp": StrategyPolicy(
                "scalp",
                scalp,
                cfg.strategy_scalp_min_confidence,
                cfg.scalp_risk_per_trade,
                cfg.max_position_size,
                cfg.scalp_atr_stop_multiplier,
                cfg.scalp_min_stop_loss_pct,
                cfg.scalp_max_stop_loss_pct,
                cfg.scalp_risk_reward_ratio,
                10,
                effective_fees,
                cfg.scalp_min_net_edge_bps,
                cfg.scalp_max_spread_bps,
                cfg.walk_forward_min_trades,
                1.25,
                cfg.walk_forward_max_drawdown_pct,
                (
                    ("1m", max(cfg.scalp_max_hold_seconds_1m // 60, 1)),
                    ("3m", max(cfg.scalp_max_hold_seconds_3m // 180, 1)),
                ),
            ),
        }
