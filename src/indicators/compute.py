# Aggregator — computes every indicator on a DataFrame in one call
import numpy as np
import pandas as pd

from src.indicators.momentum import MomentumIndicators
from src.indicators.trend import TrendIndicators
from src.indicators.volatility import VolatilityIndicators
from src.indicators.volume import VolumeIndicators

# Stateless singletons — reused across every compute_all_indicators call
trend = TrendIndicators()
momentum = MomentumIndicators()
volatility = VolatilityIndicators()
volume_inds = VolumeIndicators()


def _append_indicator_frame(
    result: pd.DataFrame, indicators: pd.DataFrame
) -> pd.DataFrame:
    overlapping = [column for column in indicators.columns if column in result.columns]
    if overlapping:
        result = result.drop(columns=overlapping)
    return pd.concat([result, indicators], axis=1)


# Entry point for all indicator computations
# Takes raw OHLCV, returns a copy with 30+ indicator columns appended
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    vol = result.get("volume", pd.Series(0, index=result.index))

    # Trend
    result["ema_50"] = trend.ema(close, 50)
    result["ema_200"] = trend.ema(close, 200)
    result["ema_50_slope"] = result["ema_50"].pct_change(5)
    result["ema_200_slope"] = result["ema_200"].pct_change(10)
    result["sma_20"] = trend.sma(close, 20)
    result["sma_50"] = trend.sma(close, 50)
    fib_df = trend.fibonacci_retracement(high, low)
    result = _append_indicator_frame(result, fib_df)
    macd_df = trend.macd(close)
    result = _append_indicator_frame(result, macd_df)
    result["adx"] = trend.adx(high, low, close)
    di_df = trend.directional_indicators(high, low, close)
    result = _append_indicator_frame(result, di_df)

    # Momentum
    result["rsi_14"] = momentum.rsi(close, 14)
    stoch_df = momentum.stochastic(high, low, close)
    result = _append_indicator_frame(result, stoch_df)
    result["cci_20"] = momentum.cci(high, low, close, 20)
    result["williams_r"] = momentum.williams_r(high, low, close)
    result["mfi"] = momentum.mfi(high, low, close, vol)
    tsi_df = momentum.tsi(close)
    result = _append_indicator_frame(result, tsi_df)

    # Volatility
    bb_df = volatility.bollinger_bands(close)
    result = _append_indicator_frame(result, bb_df)
    result["atr"] = volatility.atr(high, low, close)
    result["atr_pct"] = result["atr"] / close.replace(0, pd.NA)
    kc_df = volatility.keltner_channels(high, low, close)
    result = _append_indicator_frame(result, kc_df)
    result["hv"] = volatility.historical_volatility(close)
    dc_df = volatility.donchian_channels(high, low)
    result = _append_indicator_frame(result, dc_df)

    # Volume
    result["obv"] = volume_inds.obv(close, vol)
    result["vwap"] = volume_inds.vwap(high, low, close, vol)
    result["vol_sma_20"] = volume_inds.volume_sma(vol, 20)
    result["vol_ratio"] = vol / result["vol_sma_20"].replace(0, pd.NA)

    # Regime features used by signal generation
    result["trend_regime"] = 0
    result.loc[
        (close > result["ema_50"]) & (result["ema_50"] > result["ema_200"]),
        "trend_regime",
    ] = 1
    result.loc[
        (close < result["ema_50"]) & (result["ema_50"] < result["ema_200"]),
        "trend_regime",
    ] = -1

    # Price action features
    result["high_low_pct"] = (high - low) / close * 100
    result["high_close_pct"] = (high - close.shift(1)) / close.shift(1) * 100
    result["low_close_pct"] = (low - close.shift(1)) / close.shift(1) * 100
    result["returns"] = close.pct_change()
    result["log_returns"] = np.log(close / close.shift(1))

    return result
