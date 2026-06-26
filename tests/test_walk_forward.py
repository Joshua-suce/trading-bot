from datetime import timedelta

import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.metrics import BacktestMetrics
from src.backtest.types import BacktestTrade
from src.backtest.walk_forward import (
    WalkForwardCriteria,
    WalkForwardValidator,
    criteria_for_strategy,
)


def _result(test: pd.DataFrame, pnl: float) -> BacktestResult:
    entry = pd.Timestamp(test.index[0]).to_pydatetime()
    trade = BacktestTrade(
        entry_time=entry,
        exit_time=entry + timedelta(minutes=1),
        side="long",
        entry_price=100.0,
        exit_price=101.0,
        quantity=1.0,
        pnl=pnl,
        pnl_pct=pnl / 100,
        exit_reason="take_profit" if pnl > 0 else "stop_loss",
        symbol="BTCUSDT",
        timeframe="1m",
        strategy="scalp",
        gross_pnl=pnl + 0.1,
        fees=0.1,
    )
    metrics = BacktestMetrics.calculate([trade], [10_000, 10_000 + pnl], 10_000)
    return BacktestResult([trade], [10_000, 10_000 + pnl], metrics, [])


def test_walk_forward_uses_disjoint_train_and_test_windows():
    frame = pd.DataFrame(
        {"close": range(50)},
        index=pd.date_range("2026-01-01", periods=50, freq="min"),
    )
    windows = []

    def evaluate(train, test):
        windows.append((train.index[-1], test.index[0]))
        return _result(test, 1.0)

    validator = WalkForwardValidator(
        evaluate,
        train_size=20,
        test_size=10,
        step_size=10,
        criteria=WalkForwardCriteria(
            min_folds=3,
            min_trades=3,
            min_profit_factor=1.0,
            min_profitable_fold_ratio=1.0,
            max_expectancy_cv=0.1,
        ),
    )

    report = validator.run(frame, symbol="BTCUSDT", timeframe="1m")

    assert len(report.folds) == 3
    assert all(train_end < test_start for train_end, test_start in windows)
    assert report.aggregate_metrics.total_trades == 3
    assert report.profitable_fold_ratio == 1.0
    assert report.approved is True


def test_walk_forward_rejects_unstable_out_of_sample_results():
    frame = pd.DataFrame(
        {"close": range(50)},
        index=pd.date_range("2026-01-01", periods=50, freq="min"),
    )
    outcomes = iter([3.0, -2.0, 0.1])

    validator = WalkForwardValidator(
        lambda train, test: _result(test, next(outcomes)),
        train_size=20,
        test_size=10,
        step_size=10,
        criteria=WalkForwardCriteria(
            min_folds=3,
            min_trades=3,
            min_profit_factor=1.0,
            min_profitable_fold_ratio=0.5,
            max_expectancy_cv=0.5,
        ),
    )

    report = validator.run(frame, symbol="BTCUSDT", timeframe="1m")

    assert report.approved is False
    assert "unstable" in report.reason


def test_strategy_specific_walk_forward_criteria_are_stricter_for_countertrend():
    trend = criteria_for_strategy("trend")
    countertrend = criteria_for_strategy("countertrend")

    assert countertrend.min_profit_factor > trend.min_profit_factor
    assert countertrend.max_drawdown_pct == trend.max_drawdown_pct
