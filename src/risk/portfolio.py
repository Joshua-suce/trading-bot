# Portfolio manager — tracks account, trades, daily stats, and enforces risk limits
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from src.config import settings
from src.exchange.account import AccountInfo


# Record of a single exchange trade.
@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    timestamp: datetime
    timeframe: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None


# Aggregated stats for one trading day
@dataclass
class DailyStats:
    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    peak_equity: float = 0.0


class PortfolioManager:
    def __init__(self) -> None:
        self.account: Optional[AccountInfo] = None
        self.trades: List[TradeRecord] = []
        self.daily_stats: dict[str, DailyStats] = {}
        self.peak_equity: float = 0.0
        self.current_drawdown: float = 0.0
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.max_consecutive_losses: int = 3
        self.daily_loss_limit: float = settings.daily_loss_limit
        self.max_drawdown: float = settings.max_drawdown
        self._daily_reset()

    # Ensure today's DailyStats entry exists
    def _daily_reset(self) -> None:
        today = self._today()
        if today not in self.daily_stats:
            self.daily_stats[today] = DailyStats(date=today)
            self.daily_pnl = 0.0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def restore_risk_state(self, closed_trades: list[dict]) -> None:
        today = self._today()
        stats = DailyStats(date=today)
        self.consecutive_losses = 0

        for row in closed_trades:
            pnl = float(row.get("pnl") or 0)
            closed_at = str(row.get("closed_at") or "")
            if closed_at.startswith(today):
                stats.trades += 1
                stats.total_pnl += pnl
                if pnl > 0:
                    stats.wins += 1
                else:
                    stats.losses += 1

        for row in closed_trades:
            if float(row.get("pnl") or 0) > 0:
                break
            self.consecutive_losses += 1

        self.daily_stats[today] = stats
        self.daily_pnl = stats.total_pnl

    # Update account info and recalculate peak equity / drawdown
    def update_account(self, account: AccountInfo) -> None:
        self.account = account
        if account.total_equity > self.peak_equity:
            self.peak_equity = account.total_equity
        if self.peak_equity > 0:
            self.current_drawdown = (
                self.peak_equity - account.total_equity
            ) / self.peak_equity

    # Check whether new trades are allowed (drawdown, daily loss, consecutive losses)
    def can_trade(self) -> tuple[bool, str]:
        self._daily_reset()
        today = self._today()
        stats = self.daily_stats[today]

        if self.account and self.current_drawdown >= self.max_drawdown:
            return False, f"Max drawdown reached: {self.current_drawdown:.1%}"

        if stats.total_pnl <= -self.daily_loss_limit * (
            self.account.total_equity if self.account else 1
        ):
            return False, f"Daily loss limit reached: {stats.total_pnl:.2f}"

        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Max consecutive losses: {self.consecutive_losses}"

        return True, "ok"

    # Append a trade to the history list
    def add_trade(self, trade: TradeRecord):
        if trade not in self.trades:
            self.trades.append(trade)

    # Close a trade: compute PnL, update daily stats, log the result
    def close_trade(self, trade: TradeRecord, exit_price: float, reason: str = "tp_sl"):
        self._daily_reset()
        trade.exit_price = exit_price
        trade.exit_reason = reason
        self.add_trade(trade)
        if trade.side == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
            trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
            trade.pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        self.daily_pnl += trade.pnl or 0
        today = self._today()
        stats = self.daily_stats[today]
        stats.trades += 1
        stats.total_pnl += trade.pnl or 0
        if trade.pnl and trade.pnl > 0:
            stats.wins += 1
            self.consecutive_losses = 0
        else:
            stats.losses += 1
            self.consecutive_losses += 1

        logger.info(
            f"Trade closed: {trade.symbol} {trade.side} "
            f"PnL={trade.pnl:.2f} ({trade.pnl_pct:.2%}) reason={reason}"
        )
