import numpy as np
import pandas as pd
import pytest

from src.indicators.compute import compute_all_indicators
from src.indicators.momentum import MomentumIndicators
from src.indicators.trend import TrendIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.001),
            "high": close * (1 + abs(np.random.randn(n)) * 0.005),
            "low": close * (1 - abs(np.random.randn(n)) * 0.005),
            "close": close,
            "volume": np.random.rand(n) * 1000,
        }
    )


class TestTrendIndicators:
    def test_ema(self, sample_df):
        result = TrendIndicators.ema(sample_df["close"], 20)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_macd(self, sample_df):
        result = TrendIndicators.macd(sample_df["close"])
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_adx(self, sample_df):
        result = TrendIndicators.adx(
            sample_df["high"], sample_df["low"], sample_df["close"]
        )
        assert len(result) == len(sample_df)


class TestMomentumIndicators:
    def test_rsi(self, sample_df):
        result = MomentumIndicators.rsi(sample_df["close"], 14)
        assert len(result) == len(sample_df)
        assert result.dropna().between(0, 100).all()

    def test_stochastic(self, sample_df):
        result = MomentumIndicators.stochastic(
            sample_df["high"], sample_df["low"], sample_df["close"]
        )
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns


class TestVolatilityIndicators:
    def test_bollinger_bands(self, sample_df):
        result = VolatilityIndicators.bollinger_bands(sample_df["close"])
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        valid = result.dropna()
        assert (valid["bb_upper"] > valid["bb_middle"]).all()

    def test_atr(self, sample_df):
        result = VolatilityIndicators.atr(
            sample_df["high"], sample_df["low"], sample_df["close"]
        )
        assert len(result) == len(sample_df)
        assert (result.dropna() > 0).all()


class TestVolumeIndicators:
    def test_obv(self, sample_df):
        result = VolumeIndicators.obv(sample_df["close"], sample_df["volume"])
        assert len(result) == len(sample_df)

    def test_vwap(self, sample_df):
        result = VolumeIndicators.vwap(
            sample_df["high"], sample_df["low"], sample_df["close"], sample_df["volume"]
        )
        assert len(result) == len(sample_df)


class TestComputeAll:
    def test_compute_all_indicators(self, sample_df):
        result = compute_all_indicators(sample_df)
        expected_cols = [
            "ema_9",
            "ema_21",
            "macd",
            "rsi_14",
            "bb_upper",
            "atr",
            "obv",
            "vwap",
            "returns",
            "log_returns",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == len(sample_df)
