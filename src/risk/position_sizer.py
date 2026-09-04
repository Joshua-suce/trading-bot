# Position sizing — calculates quantity based on risk-per-trade and stop distance
from dataclasses import dataclass

from loguru import logger

from src.config import settings
from src.risk.portfolio import PortfolioManager
from src.strategies import StrategyRegistry


# Result of the sizing calculation
@dataclass
class PositionSize:
    quantity: float
    leveraged_quantity: float
    risk_amount: float
    size_type: str


class PositionSizer:
    def __init__(self, portfolio: PortfolioManager):
        self.portfolio = portfolio

    # Calculate position size: risk_amount / stop_loss_distance, capped by max
    def calculate(
        self,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
        side: str = "long",
        strategy: str | None = None,
    ) -> PositionSize:
        account = self.portfolio.account
        if account is None:
            return PositionSize(0, 0, 0, "no_account")

        equity = account.total_equity
        policy = StrategyRegistry(settings).get(strategy)
        risk_fraction = policy.risk_fraction if strategy else settings.risk_per_trade
        max_position_fraction = (
            policy.max_position_fraction if strategy else settings.max_position_size
        )
        if (
            settings.binance_environment == "mainnet"
            and settings.mainnet_canary_enabled
        ):
            risk_fraction = min(risk_fraction, settings.mainnet_canary_risk_per_trade)
            max_position_fraction = min(
                max_position_fraction,
                settings.mainnet_canary_max_position_size,
            )
        risk_per_trade = risk_fraction * equity

        # Risk per unit
        if side == "long":
            risk_per_unit = entry_price - stop_loss_price
        else:
            risk_per_unit = stop_loss_price - entry_price

        if risk_per_unit <= 0:
            logger.warning(
                "Invalid stop loss (below entry for long or above for short)"
            )
            return PositionSize(0, 0, 0, "invalid_sl")

        # Quantity based on risk
        quantity = risk_per_trade / risk_per_unit
        # Constrain by max position size
        max_pos = max_position_fraction * equity / entry_price
        quantity = min(quantity, max_pos)
        quantity = max(quantity, 0)
        actual_risk_amount = quantity * risk_per_unit

        return PositionSize(
            quantity=round(quantity, 6),
            # Exchange leverage changes margin use, not stop-defined loss risk.
            leveraged_quantity=round(quantity, 6),
            risk_amount=round(actual_risk_amount, 2),
            size_type="risk_based",
        )
