# Volume indicators — OBV, VWAP, Volume Profile, Ease of Movement, Volume SMA
import numpy as np
import pandas as pd


class VolumeIndicators:
    # OBV (On-Balance Volume): cumulative volume adjusted by price direction
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * volume).cumsum()
        return obv

    # VWAP (Volume-Weighted Average Price): typical price weighted by volume
    @staticmethod
    def vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        tp = (high + low + close) / 3
        cum_vol_price = (tp * volume).cumsum()
        cum_vol = volume.cumsum()
        return cum_vol_price / cum_vol.replace(0, np.nan)

    # Volume Profile: total volume at each price level (binned)
    @staticmethod
    def volume_profile(
        close: pd.Series, volume: pd.Series, bins: int = 10
    ) -> pd.Series:
        price_range = close.max() - close.min()
        bin_size = price_range / bins
        if bin_size == 0:
            return pd.Series(0, index=close.index)
        bucket = ((close - close.min()) / bin_size).astype(int).clip(0, bins - 1)
        profile = pd.Series(0, index=close.index)
        for b in range(bins):
            mask = bucket == b
            profile[mask] = volume[mask].sum()
        return profile

    # Ease of Movement: relates price change to volume (high = easy move)
    @staticmethod
    def eom(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        midpoint = (high + low) / 2
        distance = midpoint - midpoint.shift(1)
        ratio = (high - low).replace(0, np.nan)
        eom = (distance / ratio) * (volume / 100_000_000)
        return eom.rolling(period).mean()

    # Simple moving average of volume
    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        return volume.rolling(window=period).mean()
