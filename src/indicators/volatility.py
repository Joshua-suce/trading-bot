# Volatility indicators — Bollinger Bands, ATR, Keltner, HV, Donchian
import numpy as np
import pandas as pd


class VolatilityIndicators:
    # Bollinger Bands: SMA ± K×StdDev — measures volatility and overbought/oversold
    @staticmethod
    def bollinger_bands(
        close: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        width = (upper - lower) / sma * 100
        percent_b = (close - lower) / (upper - lower).replace(0, np.nan)
        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": sma,
                "bb_lower": lower,
                "bb_width": width,
                "bb_percent_b": percent_b,
            }
        )

    # ATR (Average True Range): measures market volatility (max of three ranges)
    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    # Keltner Channels: EMA ± ATR multiplier — volatility-based envelope
    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_mult: float = 1.5,
    ) -> pd.DataFrame:
        ema = close.ewm(span=period, adjust=False).mean()
        atr_val = VolatilityIndicators.atr(high, low, close, period)
        upper = ema + atr_mult * atr_val
        lower = ema - atr_mult * atr_val
        return pd.DataFrame({"kc_upper": upper, "kc_middle": ema, "kc_lower": lower})

    # Historical Volatility: standard deviation of log returns, annualised
    @staticmethod
    def historical_volatility(
        close: pd.Series, period: int = 20, annualize: bool = True
    ) -> pd.Series:
        log_returns = np.log(close / close.shift(1))
        hv = log_returns.rolling(window=period).std()
        if annualize:
            hv = hv * np.sqrt(365)
        return hv

    # Donchian Channels: highest high / lowest low over N periods
    @staticmethod
    def donchian_channels(
        high: pd.Series, low: pd.Series, period: int = 20
    ) -> pd.DataFrame:
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        middle = (upper + lower) / 2
        return pd.DataFrame({"dc_upper": upper, "dc_middle": middle, "dc_lower": lower})
