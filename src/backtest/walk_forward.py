from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import inf

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.metrics import BacktestMetrics
from src.backtest.types import BacktestTrade
from src.live.data_quality import timeframe_seconds
from src.strategies import StrategyRegistry


@dataclass(frozen=True)
class WalkForwardCriteria:
    min_folds: int = 3
    min_trades: int = 20
    min_profit_factor: float = 1.10
    max_drawdown_pct: float = 20.0
    min_profitable_fold_ratio: float = 0.50
    max_expectancy_cv: float = 3.0


@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: BacktestMetrics

    def to_dict(self) -> dict:
        return {**asdict(self), "metrics": asdict(self.metrics)}


@dataclass(frozen=True)
class WalkForwardReport:
    symbol: str
    timeframe: str
    folds: list[WalkForwardFold]
    aggregate_metrics: BacktestMetrics
    profitable_fold_ratio: float
    expectancy_cv: float
    approved: bool
    reason: str
    strategy: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "folds": [fold.to_dict() for fold in self.folds],
            "aggregate_metrics": asdict(self.aggregate_metrics),
            "profitable_fold_ratio": self.profitable_fold_ratio,
            "expectancy_cv": self.expectancy_cv,
            "approved": self.approved,
            "reason": self.reason,
            "strategy": self.strategy,
        }


class WalkForwardValidator:
    def __init__(
        self,
        evaluate_fold: Callable[[pd.DataFrame, pd.DataFrame], BacktestResult],
        *,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        initial_capital: float = 10_000.0,
        criteria: WalkForwardCriteria | None = None,
    ) -> None:
        if train_size < 2 or test_size < 1:
            raise ValueError("walk-forward train and test sizes are invalid")
        self.evaluate_fold = evaluate_fold
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.initial_capital = initial_capital
        self.criteria = criteria

    def run(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        strategy: str = "trend",
    ) -> WalkForwardReport:
        folds = []
        out_of_sample_trades: list[BacktestTrade] = []
        fold_start = 0
        fold_index = 0
        while fold_start + self.train_size + self.test_size <= len(df):
            train_end = fold_start + self.train_size
            test_end = train_end + self.test_size
            train = df.iloc[fold_start:train_end].copy()
            test = df.iloc[train_end:test_end].copy()
            result = self.evaluate_fold(train, test)
            trades = self._out_of_sample_trades(result.trades, test.index[0])
            metrics = self._metrics(trades, timeframe)
            folds.append(
                WalkForwardFold(
                    index=fold_index,
                    train_start=str(train.index[0]),
                    train_end=str(train.index[-1]),
                    test_start=str(test.index[0]),
                    test_end=str(test.index[-1]),
                    metrics=metrics,
                )
            )
            out_of_sample_trades.extend(trades)
            fold_start += self.step_size
            fold_index += 1

        aggregate = self._metrics(out_of_sample_trades, timeframe)
        profitable_ratio = (
            sum(fold.metrics.total_return > 0 for fold in folds) / len(folds)
            if folds
            else 0.0
        )
        expectancies = np.array(
            [fold.metrics.expectancy for fold in folds], dtype=float
        )
        mean_expectancy = float(np.mean(expectancies)) if len(expectancies) else 0.0
        expectancy_cv = (
            float(np.std(expectancies) / abs(mean_expectancy))
            if mean_expectancy != 0
            else inf
        )
        criteria = self.criteria or criteria_for_strategy(strategy)
        approved, reason = self._approval(
            folds=len(folds),
            metrics=aggregate,
            profitable_fold_ratio=profitable_ratio,
            expectancy_cv=expectancy_cv,
            criteria=criteria,
        )
        return WalkForwardReport(
            symbol=symbol.upper(),
            timeframe=timeframe,
            folds=folds,
            aggregate_metrics=aggregate,
            profitable_fold_ratio=round(profitable_ratio, 4),
            expectancy_cv=round(expectancy_cv, 4),
            approved=approved,
            reason=reason,
            strategy=strategy,
        )

    @staticmethod
    def _out_of_sample_trades(
        trades: list[BacktestTrade],
        test_start: object,
    ) -> list[BacktestTrade]:
        boundary = pd.Timestamp(test_start)
        return [trade for trade in trades if pd.Timestamp(trade.entry_time) >= boundary]

    def _metrics(
        self,
        trades: list[BacktestTrade],
        timeframe: str,
    ) -> BacktestMetrics:
        equity = [self.initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)
        periods = 365.0 * 24 * 60 * 60 / timeframe_seconds(timeframe)
        return BacktestMetrics.calculate(
            trades,
            equity,
            self.initial_capital,
            periods_per_year=periods,
        )

    def _approval(
        self,
        *,
        folds: int,
        metrics: BacktestMetrics,
        profitable_fold_ratio: float,
        expectancy_cv: float,
        criteria: WalkForwardCriteria,
    ) -> tuple[bool, str]:
        checks = [
            (folds >= criteria.min_folds, f"needs {criteria.min_folds} folds"),
            (
                metrics.total_trades >= criteria.min_trades,
                f"needs {criteria.min_trades} out-of-sample trades",
            ),
            (
                metrics.profit_factor >= criteria.min_profit_factor,
                "out-of-sample profit factor below threshold",
            ),
            (
                metrics.max_drawdown_pct <= criteria.max_drawdown_pct,
                "out-of-sample drawdown above threshold",
            ),
            (
                profitable_fold_ratio >= criteria.min_profitable_fold_ratio,
                "too few profitable folds",
            ),
            (
                metrics.expectancy > 0 and expectancy_cv <= criteria.max_expectancy_cv,
                "out-of-sample expectancy is weak or unstable",
            ),
        ]
        for passed, reason in checks:
            if not passed:
                return False, reason
        return True, "walk-forward evidence clears all acceptance criteria"


def criteria_for_strategy(strategy: str) -> WalkForwardCriteria:
    policy = StrategyRegistry().get(strategy)
    return WalkForwardCriteria(
        min_folds=3,
        min_trades=policy.walk_forward_min_trades,
        min_profit_factor=policy.walk_forward_min_profit_factor,
        max_drawdown_pct=policy.walk_forward_max_drawdown_pct,
        min_profitable_fold_ratio=0.60,
        max_expectancy_cv=2.5,
    )
