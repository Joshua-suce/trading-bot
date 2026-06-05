import pandas as pd

from src.utils.helpers import generate_mock_ohlcv


def test_generate_mock_ohlcv_shape_and_columns():
    df = generate_mock_ohlcv("BTCUSDT", "1h", limit=50)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
