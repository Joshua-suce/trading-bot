from datetime import datetime, timedelta, timezone

import pandas as pd

from src.live.data_quality import OHLCVQualityValidator, timeframe_seconds


def valid_df(periods: int = 60, freq: str = "5min") -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    index = pd.date_range(end=end, periods=periods, freq=freq)
    return pd.DataFrame(
        {
            "open": [100.0] * periods,
            "high": [102.0] * periods,
            "low": [99.0] * periods,
            "close": [101.0] * periods,
            "volume": [10.0] * periods,
        },
        index=index,
    )


def test_quality_validator_accepts_valid_ohlcv():
    result = OHLCVQualityValidator(min_candles=50).validate(valid_df(), "5m")

    assert result.valid is True


def test_quality_validator_rejects_missing_column():
    df = valid_df().drop(columns=["volume"])

    result = OHLCVQualityValidator(min_candles=50).validate(df, "5m")

    assert result.valid is False
    assert "missing columns" in result.reason


def test_quality_validator_rejects_duplicate_timestamp():
    df = valid_df()
    duplicate_index = list(df.index)
    duplicate_index[-1] = duplicate_index[-2]
    df.index = pd.DatetimeIndex(duplicate_index)

    result = OHLCVQualityValidator(min_candles=50).validate(df, "5m")

    assert result.valid is False
    assert "increasing" in result.reason or "duplicate" in result.reason


def test_quality_validator_rejects_timestamp_gap():
    df = valid_df()
    gap_index = list(df.index)
    gap_index[-2] = gap_index[-3] + timedelta(minutes=20)
    gap_index[-1] = gap_index[-2] + timedelta(minutes=5)
    df.index = pd.DatetimeIndex(gap_index)

    result = OHLCVQualityValidator(min_candles=50).validate(df, "5m")

    assert result.valid is False
    assert "gap" in result.reason


def test_timeframe_seconds():
    assert timeframe_seconds("5m") == 300
    assert timeframe_seconds("2h") == 7200
