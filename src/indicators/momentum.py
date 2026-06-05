# Momentum oscillators — RSI, Stochastic, CCI, Williams %R, MFI, TSI
import numpy as np
import pandas as pd


class MomentumIndicators:
    # RSI (Relative Strength Index): measures speed/change of price movements
    # Values below 30 = oversold, above 70 = overbought
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (
                avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]
            ) / period
            avg_loss.iloc[i] = (
                avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]
            ) / period
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Stochastic Oscillator: compares close to the high/low range
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (
            (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        )
        d = k.rolling(d_period).mean()
        return pd.DataFrame({"stoch_k": k, "stoch_d": d})

    # CCI (Commodity Channel Index): deviation from statistical mean
    @staticmethod
    def cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> pd.Series:
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
        return cci

    # Williams %R: overbought/oversold (inverse of Stochastic)
    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        wr = -100 * (
            (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
        )
        return wr

    # MFI (Money Flow Index): volume-weighted RSI
    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        tp = (high + low + close) / 3
        money_flow = tp * volume
        sign = tp.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        pos_mf = money_flow.where(sign > 0, 0).rolling(period).sum()
        neg_mf = money_flow.where(sign < 0, 0).rolling(period).sum()
        mfr = pos_mf / neg_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
        return mfi

    # TSI (True Strength Index): double-smoothed momentum oscillator
    @staticmethod
    def tsi(
        close: pd.Series, fast: int = 13, slow: int = 25, signal: int = 13
    ) -> pd.DataFrame:
        diff = close.diff()
        abs_diff = diff.abs()
        double_smoothed = diff.ewm(span=fast).mean().ewm(span=slow).mean()
        double_smoothed_abs = abs_diff.ewm(span=fast).mean().ewm(span=slow).mean()
        tsi = 100 * (double_smoothed / double_smoothed_abs.replace(0, np.nan))
        signal_line = tsi.ewm(span=signal).mean()
        return pd.DataFrame({"tsi": tsi, "tsi_signal": signal_line})
