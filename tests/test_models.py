import numpy as np
import pandas as pd
import pytest

from src.indicators.compute import compute_all_indicators
from src.models.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_feature_df():
    np.random.seed(42)
    n = 200
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    raw = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.rand(n) * 1000,
        }
    )
    return compute_all_indicators(raw)


class TestFeatureEngineer:
    def test_feature_creation(self, sample_feature_df):
        fe = FeatureEngineer(lookback=50)
        result = fe.create_features(sample_feature_df)
        assert len(fe.feature_cols) > 0
        assert "return_lag_1" in result.columns
        assert "target_direction" in result.columns

    def test_feature_matrix(self, sample_feature_df):
        fe = FeatureEngineer(lookback=50)
        df_feat = fe.create_features(sample_feature_df).dropna()
        X = fe.get_feature_matrix(df_feat)
        assert X.ndim == 2
        assert X.shape[0] == len(df_feat)
        assert X.shape[1] == len(fe.feature_cols)


def test_trainer_prepare_data_with_raw_ohlcv():
    from src.models.trainer import ModelTrainer

    np.random.seed(42)
    n = 200
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    raw = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.rand(n) * 1000,
        }
    )

    trainer = ModelTrainer()
    X, y = trainer.prepare_data(raw)
    assert X.ndim == 2
    assert len(y) == X.shape[0]
    assert set(y).issubset({-1, 0, 1})
