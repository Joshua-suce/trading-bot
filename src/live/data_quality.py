from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src.config import settings

REQUIRED_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class DataQualityResult:
    valid: bool
    reason: str = "ok"


class OHLCVQualityValidator:
    def __init__(
        self,
        *,
        min_candles: int | None = None,
        max_delay_multiplier: float | None = None,
        allow_zero_volume: bool | None = None,
    ) -> None:
        self.min_candles = min_candles or settings.min_ohlcv_candles
        self.max_delay_multiplier = (
            max_delay_multiplier or settings.max_candle_delay_multiplier
        )
        self.allow_zero_volume = (
            settings.allow_zero_volume_candles
            if allow_zero_volume is None
            else allow_zero_volume
        )

    def validate(self, df: pd.DataFrame, timeframe: str) -> DataQualityResult:
        structural = self._validate_structure(df)
        if not structural.valid:
            return structural
        values = self._validate_values(df)
        if not values.valid:
            return values
        timing = self._validate_timing(df, timeframe)
        if not timing.valid:
            return timing
        return DataQualityResult(True)

    def _validate_structure(self, df: pd.DataFrame) -> DataQualityResult:
        missing = REQUIRED_OHLCV_COLUMNS - set(df.columns)
        if missing:
            return DataQualityResult(False, f"missing columns: {sorted(missing)}")
        if len(df) < self.min_candles:
            return DataQualityResult(False, f"not enough candles: {len(df)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            return DataQualityResult(False, "index must be DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            return DataQualityResult(False, "timestamps are not increasing")
        if df.index.has_duplicates:
            return DataQualityResult(False, "duplicate timestamps detected")
        return DataQualityResult(True)

    def _validate_values(self, df: pd.DataFrame) -> DataQualityResult:
        numeric = df[list(REQUIRED_OHLCV_COLUMNS)]
        if numeric.isna().any().any():
            return DataQualityResult(False, "nan values detected")
        if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
            return DataQualityResult(False, "non-positive price detected")
        if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
            return DataQualityResult(False, "high is below candle body/range")
        if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
            return DataQualityResult(False, "low is above candle body/range")
        if not self.allow_zero_volume and (numeric["volume"] <= 0).any():
            return DataQualityResult(False, "non-positive volume detected")
        return DataQualityResult(True)

    def _validate_timing(self, df: pd.DataFrame, timeframe: str) -> DataQualityResult:
        expected_seconds = timeframe_seconds(timeframe)
        gaps = df.index.to_series().diff().dropna().dt.total_seconds()
        if not gaps.empty and (gaps > expected_seconds * 1.5).any():
            return DataQualityResult(False, "timestamp gap detected")

        last_closed = df.index[-2]
        if last_closed.tzinfo is None:
            last_closed = last_closed.tz_localize(timezone.utc)
        delay = (
            datetime.now(timezone.utc) - last_closed.to_pydatetime()
        ).total_seconds()
        if delay > expected_seconds * self.max_delay_multiplier:
            return DataQualityResult(False, f"stale candles: delay={delay:.0f}s")
        return DataQualityResult(True)


def timeframe_seconds(timeframe: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86_400, "w": 604_800}
    unit = timeframe[-1]
    if unit == "M":
        return 2_592_000
    return int(timeframe[:-1]) * units[unit]
