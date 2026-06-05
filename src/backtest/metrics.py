# Performance metrics — calculates risk/return statistics from backtest results
from dataclasses import dataclass
from typing import List

import numpy as np

from src.backtest.types import BacktestTrade


# Comprehensive set of backtest performance metrics
@dataclass
class BacktestMetrics:
    total_return: float
    total_return_pct: float
    annualized_return: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_periods: float

    # Compute all metrics from the trade list and equity curve
    @staticmethod
    def calculate(
        trades: List[BacktestTrade], equity_curve: List[float], initial_capital: float
    ) -> "BacktestMetrics":
        equity = np.array(equity_curve)
        total_return = equity[-1] - initial_capital
        total_return_pct = total_return / initial_capital * 100

        # Annualized return (assuming hourly data unless specified)
        n_periods = len(equity)
        periods_per_year = 8760  # hourly
        annualized_return = (equity[-1] / initial_capital) ** (
            periods_per_year / max(n_periods, 1)
        ) - 1

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = float(np.max(drawdown))
        max_dd_pct = float(np.max(drawdown / peak * 100))

        # Trade stats
        total_trades = len(trades)
        if total_trades == 0:
            return BacktestMetrics(
                total_return=total_return,
                total_return_pct=total_return_pct,
                annualized_return=0,
                total_trades=0,
                win_rate=0,
                profit_factor=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                max_drawdown=max_dd,
                max_drawdown_pct=max_dd_pct,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                avg_holding_periods=0,
            )

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        win_rate = len(wins) / total_trades * 100
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / max(total_losses, 1)

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # Holding periods
        holding_periods = []
        for t in trades:
            if t.entry_time and t.exit_time:
                if hasattr(t.exit_time, "timestamp") and hasattr(
                    t.entry_time, "timestamp"
                ):
                    holding = (t.exit_time - t.entry_time).total_seconds() / 3600
                    holding_periods.append(holding)
        avg_holding = float(np.mean(holding_periods)) if holding_periods else 0.0

        # Sharpe (using equity curve returns)
        equity_returns = np.diff(equity) / equity[:-1]
        sharpe = 0
        if len(equity_returns) > 1 and np.std(equity_returns) > 0:
            sharpe = (
                np.mean(equity_returns)
                / np.std(equity_returns)
                * np.sqrt(periods_per_year)
            )

        # Sortino (downside deviation)
        downside = equity_returns[equity_returns < 0]
        sortino = 0
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = (
                np.mean(equity_returns) / np.std(downside) * np.sqrt(periods_per_year)
            )

        # Calmar
        calmar = annualized_return / (max_dd_pct / 100) if max_dd_pct > 0 else 0

        return BacktestMetrics(
            total_return=round(total_return, 2),
            total_return_pct=round(total_return_pct, 2),
            annualized_return=round(annualized_return, 4),
            total_trades=total_trades,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            largest_win=round(largest_win, 2),
            largest_loss=round(largest_loss, 2),
            avg_holding_periods=round(avg_holding, 2),
        )

    # Human-readable dictionary of results for logging
    def to_dict(self) -> dict:
        return {
            "Total Return": f"${self.total_return}",
            "Total Return %": f"{self.total_return_pct}%",
            "Ann. Return": f"{self.annualized_return:.2%}",
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate}%",
            "Profit Factor": self.profit_factor,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Calmar Ratio": self.calmar_ratio,
            "Max Drawdown": f"${self.max_drawdown} ({self.max_drawdown_pct}%)",
            "Avg Win": f"${self.avg_win}",
            "Avg Loss": f"${self.avg_loss}",
            "Largest Win": f"${self.largest_win}",
            "Largest Loss": f"${self.largest_loss}",
            "Avg Hold (hrs)": self.avg_holding_periods,
        }
