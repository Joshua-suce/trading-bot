# Position sizing — calculates quantity based on risk-per-trade and stop distance
from dataclasses import dataclass

from loguru import logger

from src.config import settings
from src.risk.portfolio import PortfolioManager


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
    ) -> PositionSize:
        account = self.portfolio.account
        if account is None:
            return PositionSize(0, 0, 0, "no_account")

        equity = account.total_equity
        risk_per_trade = settings.risk_per_trade * equity

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
        leveraged_quantity = quantity * min(leverage, settings.max_leverage)

        # Constrain by max position size
        max_pos = settings.max_position_size * equity / entry_price
        quantity = min(quantity, max_pos)
        leveraged_quantity = min(
            leveraged_quantity, max_pos * min(leverage, settings.max_leverage)
        )

        quantity = max(quantity, 0)
        leveraged_quantity = max(leveraged_quantity, 0)

        return PositionSize(
            quantity=round(quantity, 6),
            leveraged_quantity=round(leveraged_quantity, 6),
            risk_amount=round(risk_per_trade, 2),
            size_type="risk_based",
        )
