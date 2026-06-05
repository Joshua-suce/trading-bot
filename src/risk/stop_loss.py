# Stop-loss, take-profit, and trailing stop calculation based on ATR
from dataclasses import dataclass
from typing import Optional


# Price levels for stop-loss, take-profit, and trailing stop
@dataclass
class StopLossLevels:
    stop_loss: float
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0


class StopLossManager:
    # Defaults: 1.5× ATR for SL, 3.0× ATR for TP (2:1 risk-reward)
    def __init__(
        self,
        atr_mult_sl: float = 1.0,
        atr_mult_tp: float = 2.0,
        risk_reward_ratio: float = 2.0,
        trailing_activation: float = 1.0,
    ):
        self.atr_mult_sl = atr_mult_sl
        self.atr_mult_tp = atr_mult_tp
        self.risk_reward_ratio = risk_reward_ratio
        self.trailing_activation = trailing_activation

    # Calculate SL/TP/trailing levels from entry price, side, and ATR
    def calculate(
        self,
        entry_price: float,
        side: str,
        atr: float,
    ) -> StopLossLevels:
        if atr <= 0:
            atr = entry_price * 0.01  # fallback 1%

        sl_distance = atr * self.atr_mult_sl

        if side == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * self.risk_reward_ratio
            trailing_stop = entry_price + sl_distance * self.trailing_activation
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * self.risk_reward_ratio
            trailing_stop = entry_price - sl_distance * self.trailing_activation

        sl_pct = sl_distance / entry_price * 100
        tp_pct = sl_distance * self.risk_reward_ratio / entry_price * 100

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
