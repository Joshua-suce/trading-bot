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

    def test_final_candle_has_no_future_direction_label(self, sample_feature_df):
        fe = FeatureEngineer(lookback=50)

        result = fe.create_features(sample_feature_df)

        assert pd.isna(result.iloc[-1]["target"])
        assert pd.isna(result.iloc[-1]["target_direction"])


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
    assert np.issubdtype(y.dtype, np.integer)


def test_xgboost_classifier_encodes_direction_labels_and_round_trips(tmp_path):
    from src.models.classifier import XGBoostClassifier

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 5))
    y = np.array([-1, 1] * 20)
    model = XGBoostClassifier(n_estimators=2, max_depth=2)
    model.metadata = {"symbol": "BTCUSDT", "timeframe": "5m"}

    model.train(X, y)
    predictions, confidence = model.predict_with_confidence(X[:4])

    assert set(predictions).issubset({-1, 1})
    assert np.all((confidence >= 0) & (confidence <= 1))

    path = tmp_path / "xgb.json"
    model.save(str(path))
    assert (tmp_path / "xgb.meta.json").exists()
    restored = XGBoostClassifier()
    restored.load(str(path))

    restored_predictions, restored_confidence = restored.predict_with_confidence(X[:4])
    assert restored.metadata == {"symbol": "BTCUSDT", "timeframe": "5m"}
    np.testing.assert_array_equal(restored_predictions, predictions)
    np.testing.assert_allclose(restored_confidence, confidence)
