import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
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

    assert result.metrics.total_return_pct >= 0
    assert result.metrics.profit_factor >= 1
