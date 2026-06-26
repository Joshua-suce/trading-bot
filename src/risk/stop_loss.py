# Stop-loss, take-profit, and trailing stop calculation based on ATR
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.config import settings
from src.strategies import StrategyRegistry


# Price levels for stop-loss, take-profit, and trailing stop
@dataclass
class StopLossLevels:
    stop_loss: float
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0


class StopLossManager:
    # Defaults: 1.5× ATR for SL, 3.0× ATR for TP, 2:1 risk-reward
    def __init__(
        self,
        atr_mult_sl: float = 1.5,
        atr_mult_tp: float = 3.0,
        risk_reward_ratio: float = 2.0,
        trailing_activation: float = 1.0,
        min_stop_loss_pct: float = 0.0035,
        max_stop_loss_pct: float = 0.02,
    ):
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp
        self.risk_reward_ratio = risk_reward_ratio
        self.trailing_activation = trailing_activation
        self.min_stop_loss_pct = min_stop_loss_pct
        self.max_stop_loss_pct = max_stop_loss_pct

    # Calculate SL/TP/trailing levels from entry price, side, and ATR
    def calculate(
        self,
        entry_price: float,
        side: str,
        atr: float,
        strategy: str | None = None,
    ) -> StopLossLevels:
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.01  # fallback 1%

        policy = StrategyRegistry(settings).get(strategy)
        scalp = strategy == "scalp"
        atr_multiplier = policy.atr_stop_multiplier if strategy else self.atr_mult_sl
        min_stop_loss_pct = policy.min_stop_pct if strategy else self.min_stop_loss_pct
        max_stop_loss_pct = policy.max_stop_pct if strategy else self.max_stop_loss_pct
        risk_reward_ratio = (
            policy.reward_risk_ratio if strategy else self.risk_reward_ratio
        )
        # Adaptive ATR floor for scalp: ensures ATR never shrinks below a level
        # that would make raw_distance land far below min_stop.  When volatility
        # expands, the stop widens organically with ATR instead of staying pinned
        # at the minimum.  For non-scalp trades we use the raw ATR as-is.
        if scalp:
            min_atr_floor_pct = min_stop_loss_pct * 0.5 / atr_multiplier
            atr = max(atr, entry_price * min_atr_floor_pct)
        raw_distance = atr * atr_multiplier
        sl_distance = min(
            max(raw_distance, entry_price * min_stop_loss_pct),
            entry_price * max_stop_loss_pct,
        )

        if side == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * risk_reward_ratio
            trailing_stop = entry_price + sl_distance * self.trailing_activation
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * risk_reward_ratio
            trailing_stop = entry_price - sl_distance * self.trailing_activation

        sl_pct = sl_distance / entry_price * 100
        tp_pct = sl_distance * risk_reward_ratio / entry_price * 100

        return StopLossLevels(
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            trailing_stop=round(trailing_stop, 2),
            stop_loss_pct=round(sl_pct, 2),
            take_profit_pct=round(tp_pct, 2),
        )

    # Update trailing stop: only move in the profitable direction
    def update_trailing(
        self,
        current_price: float,
        side: str,
        entry_price: float,
        atr: float,
        current_stop: float,
    ) -> float:
        if side == "long":
            new_stop = current_price - atr * self.atr_mult_sl
            return max(new_stop, current_stop) if current_stop else new_stop
        else:
            new_stop = current_price + atr * self.atr_mult_sl
            return min(new_stop, current_stop) if current_stop else new_stop
