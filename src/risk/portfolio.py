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
    strategy: Optional[str] = None
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    gross_pnl: Optional[float] = None


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
        self.last_loss_at: Optional[datetime] = None
        self.max_consecutive_losses: int = settings.max_consecutive_losses
        self.consecutive_loss_cooldown_seconds: int = (
            settings.consecutive_loss_cooldown_seconds
        )
        self.loss_cooldown_seconds: int = settings.loss_cooldown_seconds
        self.last_symbol_loss_at: dict[str, datetime] = {}
        self.strategy_consecutive_losses: dict[str, int] = {}
        self.strategy_last_loss_at: dict[str, datetime] = {}
        self.strategy_max_consecutive_losses = settings.strategy_max_consecutive_losses
        self.strategy_loss_cooldown_seconds = settings.strategy_loss_cooldown_seconds
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
        self.last_loss_at = None
        self.strategy_consecutive_losses.clear()
        self.strategy_last_loss_at.clear()

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
            if self.last_loss_at is None and row.get("closed_at"):
                parsed = datetime.fromisoformat(str(row["closed_at"]))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                self.last_loss_at = parsed

        self._restore_strategy_loss_streaks(closed_trades)

        self.daily_stats[today] = stats
        self.daily_pnl = stats.total_pnl

    def _restore_strategy_loss_streaks(self, closed_trades: list[dict]) -> None:
        strategies = {
            str(row.get("strategy") or "trend").lower() for row in closed_trades
        }
        for strategy in strategies:
            scoped_rows = (
                row
                for row in closed_trades
                if str(row.get("strategy") or "trend").lower() == strategy
            )
            for row in scoped_rows:
                if float(row.get("pnl") or 0) > 0:
                    break
                self.strategy_consecutive_losses[strategy] = (
                    self.strategy_consecutive_losses.get(strategy, 0) + 1
                )
                if strategy not in self.strategy_last_loss_at and row.get("closed_at"):
                    parsed = datetime.fromisoformat(str(row["closed_at"]))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    self.strategy_last_loss_at[strategy] = parsed

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
    def can_trade(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> tuple[bool, str]:
        self._daily_reset()
        today = self._today()
        stats = self.daily_stats[today]

        if self.account and self.current_drawdown >= self.max_drawdown:
            return False, f"Max drawdown reached: {self.current_drawdown:.1%}"

        if stats.total_pnl <= -self.daily_loss_limit * (
            self.account.total_equity if self.account else 1
        ):
            return False, f"Daily loss limit reached: {stats.total_pnl:.2f}"

        if strategy:
            key = strategy.lower()
            losses = self.strategy_consecutive_losses.get(key, 0)
            if losses >= self.strategy_max_consecutive_losses:
                remaining = self._strategy_loss_cooldown_remaining(key)
                if remaining <= 0:
                    self.strategy_consecutive_losses[key] = 0
                    self.strategy_last_loss_at.pop(key, None)
                else:
                    return (
                        False,
                        f"{key} loss limit: {losses}; automatic retry in "
                        f"{remaining:.0f}s",
                    )
        elif self.consecutive_losses >= self.max_consecutive_losses:
            remaining = self._loss_cooldown_remaining()
            if remaining <= 0:
                self.consecutive_losses = 0
                self.last_loss_at = None
            else:
                return (
                    False,
                    f"Max consecutive losses: {self.consecutive_losses}; "
                    f"automatic retry in {remaining:.0f}s",
                )

        return True, "ok"

    # Append a trade to the history list
    def add_trade(self, trade: TradeRecord):
        if trade not in self.trades:
            self.trades.append(trade)

    # Close a trade: compute PnL, update daily stats, log the result
    def close_trade(
        self,
        trade: TradeRecord,
        exit_price: float,
        reason: str = "tp_sl",
        *,
        exit_fee: float = 0.0,
    ):
        self._daily_reset()
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.exit_fee = max(float(exit_fee), 0.0)
        self.add_trade(trade)
        direction = 1 if trade.side == "long" else -1
        trade.gross_pnl = (exit_price - trade.entry_price) * trade.quantity * direction
        trade.pnl = trade.gross_pnl - max(trade.entry_fee, 0.0) - trade.exit_fee
        entry_notional = trade.entry_price * trade.quantity
        trade.pnl_pct = trade.pnl / entry_notional if entry_notional > 0 else 0.0

        self.daily_pnl += trade.pnl or 0
        today = self._today()
        stats = self.daily_stats[today]
        stats.trades += 1
        stats.total_pnl += trade.pnl or 0
        if trade.pnl and trade.pnl > 0:
            stats.wins += 1
            self.consecutive_losses = 0
            self.last_loss_at = None
            if trade.strategy:
                key = trade.strategy.lower()
                self.strategy_consecutive_losses[key] = 0
                self.strategy_last_loss_at.pop(key, None)
        else:
            stats.losses += 1
            self.consecutive_losses += 1
            self.last_loss_at = datetime.now(timezone.utc)
            self.last_symbol_loss_at[trade.symbol] = datetime.now(timezone.utc)
            key = str(trade.strategy or "trend").lower()
            self.strategy_consecutive_losses[key] = (
                self.strategy_consecutive_losses.get(key, 0) + 1
            )
            self.strategy_last_loss_at[key] = datetime.now(timezone.utc)

        logger.info(
            f"Trade closed: {trade.symbol} {trade.side} "
            f"PnL={trade.pnl:.2f} ({trade.pnl_pct:.2%}) "
            f"gross={trade.gross_pnl:.2f} fees={trade.entry_fee + trade.exit_fee:.2f} "
            f"reason={reason}"
        )

    def _loss_cooldown_remaining(self) -> float:
        if self.last_loss_at is None:
            return float(self.consecutive_loss_cooldown_seconds)
        last_loss_at = self.last_loss_at
        if last_loss_at.tzinfo is None:
            last_loss_at = last_loss_at.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last_loss_at).total_seconds()
        return max(self.consecutive_loss_cooldown_seconds - elapsed, 0.0)

    def _strategy_loss_cooldown_remaining(self, strategy: str) -> float:
        last_loss_at = self.strategy_last_loss_at.get(strategy)
        if last_loss_at is None:
            return float(self.strategy_loss_cooldown_seconds)
        if last_loss_at.tzinfo is None:
            last_loss_at = last_loss_at.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last_loss_at).total_seconds()
        return max(self.strategy_loss_cooldown_seconds - elapsed, 0.0)
