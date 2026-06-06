# Feature engineering — transforms indicators into 50+ ML features with target labels
from typing import List

import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.feature_cols: List[str] = []

    # Create lagged returns, rolling stats, indicator derivatives, and target
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 21]:
            result[f"return_lag_{lag}"] = result["returns"].shift(lag)

        # Lagged log returns
        for lag in [1, 2, 3, 5, 10, 21]:
            result[f"log_return_lag_{lag}"] = result["log_returns"].shift(lag)

        # Rolling statistics
        for window in [5, 10, 21]:
            result[f"return_mean_{window}"] = result["returns"].rolling(window).mean()
            result[f"return_std_{window}"] = result["returns"].rolling(window).std()
            result[f"return_skew_{window}"] = result["returns"].rolling(window).skew()
            result[f"return_kurt_{window}"] = result["returns"].rolling(window).kurt()

        # RSI features
        result["rsi_ma"] = result["rsi_14"].rolling(5).mean()
        result["rsi_div"] = result["rsi_14"] - result["rsi_ma"]
        result["rsi_above_70"] = (result["rsi_14"] > 70).astype(int)
        result["rsi_below_30"] = (result["rsi_14"] < 30).astype(int)

        # MACD features
        result["macd_signal_cross"] = (
            (result["macd"] - result["macd_signal"])
            .diff()
            .apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        )
        result["macd_hist_pct"] = result["macd_hist"] / result["close"].abs()

        # BB features
        result["bb_position"] = (result["close"] - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        ).replace(0, np.nan)
        result["bb_breakout_high"] = (result["close"] > result["bb_upper"]).astype(int)
        result["bb_breakout_low"] = (result["close"] < result["bb_lower"]).astype(int)

        # Volatility features
        result["atr_pct"] = result["atr"] / result["close"] * 100
        result["hv_rank"] = result["hv"].rolling(50).rank(pct=True)
        result["volatility_ratio"] = result["atr"] / result["atr"].rolling(50).mean()

        # Volume features
        result["obv_signal"] = result["obv"] / result["obv"].rolling(21).mean()
        result["vwap_distance"] = (
            (result["close"] - result["vwap"]) / result["close"] * 100
        )
        result["volume_spike"] = (result["vol_ratio"] > 1.5).astype(int)

        # Price relative to moving averages
        for period in [50, 200]:
            col = f"ema_{period}"
            if col in result.columns:
                result[f"price_vs_{col}"] = (
                    (result["close"] - result[col]) / result[col] * 100
                )

        # Crossover signals
        result["ema_50_200_cross"] = (
            (result["ema_50"] - result["ema_200"])
            .diff()
            .apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        )

        # Fibonacci confluence features
        for level in ["fib_236", "fib_382", "fib_500", "fib_618", "fib_786"]:
            if level in result.columns:
                result[f"distance_to_{level}"] = (
                    (result["close"] - result[level]) / result["close"] * 100
                )

        # Target: next-period return direction (for classification)
        result["target"] = result["close"].shift(-1) - result["close"]
        result["target_direction"] = result["target"].apply(
            lambda x: 1 if x > 0 else (0 if x == 0 else -1)
        )

        self.feature_cols = [
            c
            for c in result.columns
            if c
            not in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "target",
                "target_direction",
            ]
        ]

        return result

    # Extract the feature matrix (numpy array) from a feature-engineered DataFrame
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_cols].fillna(0).values
