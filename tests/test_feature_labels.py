import numpy as np
import pandas as pd

from src.models.feature_engineer import FeatureEngineer


def test_labels_use_horizon_and_volatility_neutral_band():
    close = np.array([100.0, 100.02, 100.03, 101.5, 101.6, 99.0, 99.1])
    df = pd.DataFrame(
        {
            "close": close,
            "atr": np.full(len(close), 1.0),
            "returns": pd.Series(close).pct_change(),
            "log_returns": np.log(pd.Series(close) / pd.Series(close).shift(1)),
            "rsi_14": np.full(len(close), 50.0),
            "macd": np.zeros(len(close)),
            "macd_signal": np.zeros(len(close)),
            "macd_hist": np.zeros(len(close)),
            "bb_lower": close - 2,
            "bb_upper": close + 2,
            "hv": np.full(len(close), 0.2),
            "obv": np.arange(1, len(close) + 1, dtype=float),
            "vwap": close,
            "vol_ratio": np.ones(len(close)),
            "ema_50": close,
            "ema_200": close,
        }
    )
    engineer = FeatureEngineer(
        prediction_horizon=2,
        label_atr_multiplier=0.5,
        label_min_return=0.001,
    )

    result = engineer.create_features(df)

    assert result.loc[0, "target_direction"] == 0
    assert result.loc[1, "target_direction"] == 1
    assert result.loc[3, "target_direction"] == -1
    assert "target_return" not in engineer.feature_cols
    assert "target_threshold" not in engineer.feature_cols
