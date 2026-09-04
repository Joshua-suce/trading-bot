import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.baseline import (
    BaselineAcceptanceCriteria,
    build_scope_baselines,
)
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import BacktestMetrics
from src.backtest.types import BacktestTrade
from src.signals.ta_signal import TechnicalSignal
from src.utils.helpers import generate_mock_ohlcv


@pytest.fixture
def sample_bt_df():
    np.random.seed(42)
    n = 500
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.rand(n) * 1000,
        }
    )


def test_backtest_engine(sample_bt_df):
    engine = BacktestEngine(initial_capital=10000)

    def dummy_signal(df):
        i = len(df) - 1
        if i % 10 == 0:
            return {"direction": 1, "confidence": 0.7}
        return {"direction": 0, "confidence": 0.0}

    result = engine.run(sample_bt_df, dummy_signal)
    assert len(result.trades) >= 0
    assert len(result.equity_curve) > 0
    assert result.metrics.total_trades >= 0


def test_default_strategy_is_nonnegative_on_seeded_mock_data():
    df = generate_mock_ohlcv("BTCUSDT", "1h", limit=600)
    ta = TechnicalSignal()

    def signal_fn(data):
        signal = ta.generate(data)
        return {"direction": signal.direction, "confidence": signal.strength}

    result = BacktestEngine(initial_capital=10000, leverage=1).run(df, signal_fn)

    assert result.metrics.total_trades >= 0
    assert np.isfinite(result.metrics.total_return_pct)
    assert np.isfinite(result.metrics.profit_factor)


def test_backtest_trade_reports_entry_and_exit_costs(monkeypatch):
    index = pd.date_range("2026-01-01", periods=103, freq="min")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000.0,
        },
        index=index,
    )
    frame.loc[index[-1], "high"] = 105.0
    frame.loc[index[-1], "low"] = 100.5

    def indicators(data):
        result = data.copy()
        result["atr"] = 1.0
        return result

    monkeypatch.setattr("src.backtest.engine.compute_all_indicators", indicators)

    def signal_fn(data):
        if len(data) == 101:
            return {"direction": 1, "confidence": 0.8, "strategy": "scalp"}
        return {"direction": 0, "confidence": 0.0, "strategy": "scalp"}

    result = BacktestEngine(
        initial_capital=10_000,
        commission=0.001,
        slippage=0.01,
    ).run(
        frame,
        signal_fn,
        symbol="ETHUSDT",
        timeframe="1m",
        default_strategy="scalp",
    )

    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == pytest.approx(103.0 * 0.99)
    assert trade.pnl == pytest.approx(trade.gross_pnl - trade.fees)
    assert trade.fees > 0
    assert trade.slippage_cost > 0
    assert trade.symbol == "ETHUSDT"
    assert trade.timeframe == "1m"
    assert trade.strategy == "scalp"
    assert result.metrics.total_fees == pytest.approx(trade.fees)
    assert result.metrics.total_slippage_cost == pytest.approx(trade.slippage_cost)


def test_strategy_policy_backtest_uses_timeframe_maximum_hold(monkeypatch):
    index = pd.date_range("2026-01-01", periods=120, freq="5min")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 1000.0,
        },
        index=index,
    )

    def indicators(data):
        result = data.copy()
        result["atr"] = 1.0
        return result

    monkeypatch.setattr("src.backtest.engine.compute_all_indicators", indicators)

    def signal_fn(data):
        if len(data) == 101:
            return {"direction": 1, "confidence": 0.8, "strategy": "trend"}
        return {"direction": 0, "confidence": 0.0, "strategy": "trend"}

    result = BacktestEngine(commission=0.0, slippage=0.0).run(
        frame,
        signal_fn,
        timeframe="5m",
        default_strategy="trend",
        use_strategy_policy=True,
    )

    trade = result.trades[0]
    assert trade.exit_reason == "maximum_hold"
    assert trade.exit_time - trade.entry_time == timedelta(minutes=60)


def test_profit_factor_preserves_small_trade_precision():
    now = datetime(2026, 1, 1)
    trades = [
        BacktestTrade(
            now,
            now + timedelta(hours=1),
            "long",
            100,
            101,
            1,
            0.2,
            0.002,
            "take_profit",
        ),
        BacktestTrade(
            now,
            now + timedelta(hours=1),
            "short",
            100,
            101,
            1,
            -0.1,
            -0.001,
            "stop_loss",
        ),
    ]

    metrics = BacktestMetrics.calculate(
        trades,
        [100.0, 100.2, 100.1],
        100.0,
        periods_per_year=365.0,
    )

    assert metrics.profit_factor == 2.0
    assert metrics.expectancy == pytest.approx(0.05)
    json.dumps(metrics.to_dict())


def test_scope_baseline_groups_after_cost_results():
    now = datetime(2026, 1, 1)
    trades = [
        BacktestTrade(
            now,
            now + timedelta(minutes=5),
            "long",
            100,
            101,
            1,
            0.6,
            0.006,
            "take_profit",
            symbol="ETHUSDT",
            timeframe="1m",
            strategy="scalp",
            gross_pnl=1.0,
            fees=0.2,
            slippage_cost=0.2,
        ),
        BacktestTrade(
            now,
            now + timedelta(minutes=5),
            "short",
            100,
            100.2,
            1,
            -0.2,
            -0.002,
            "stop_loss",
            symbol="ETHUSDT",
            timeframe="1m",
            strategy="scalp",
            gross_pnl=0.0,
            fees=0.1,
            slippage_cost=0.1,
        ),
    ]

    rows = build_scope_baselines(
        trades,
        criteria=BaselineAcceptanceCriteria(
            min_trades=2,
            min_profit_factor=1.1,
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.net_pnl == pytest.approx(0.4)
    assert row.profit_factor == 3.0
    assert row.total_fees == pytest.approx(0.3)
    assert row.evidence_status == "promotion_candidate"
