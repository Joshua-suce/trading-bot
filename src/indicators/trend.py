# Trend-following technical indicators — EMA, SMA, MACD, ADX, PSAR, Ichimoku

import pandas as pd


class TrendIndicators:
    # Exponential Moving Average — gives more weight to recent prices
    @staticmethod
    def ema(close: pd.Series, period: int = 20) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    # Simple Moving Average — equal weight over the window
    @staticmethod
    def sma(close: pd.Series, period: int = 20) -> pd.Series:
        return close.rolling(window=period).mean()

    # MACD (Moving Average Convergence Divergence): momentum + trend
    # Returns MACD line, signal line, and histogram
    @staticmethod
    def macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_hist": histogram,
            }
        )

    # ADX (Average Directional Index): measures trend strength (not direction)
    # Uses +DI / -DI calculations internally
    @staticmethod
    def adx(
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
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA))
        adx = dx.rolling(window=period).mean()
        return adx

    # Directional Indicators (+DI / -DI) for trend direction awareness
    @staticmethod
    def directional_indicators(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.DataFrame:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period).mean()
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di})

    # Parabolic SAR — trailing-stop indicator that accelerates over time
    @staticmethod
    def psar(
        high: pd.Series,
        low: pd.Series,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2,
    ) -> pd.Series:
        length = len(high)
        psar = high.copy()
        trend = [1] * length
        af = af_start
        ep = low.iloc[0]
        for i in range(2, length):
            if trend[i - 1] == 1:
                psar.iloc[i] = psar.iloc[i - 1] + af * (ep - psar.iloc[i - 1])
                if low.iloc[i] < psar.iloc[i]:
                    trend[i] = -1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = high.iloc[i]
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_increment, af_max)
            else:
                psar.iloc[i] = psar.iloc[i - 1] + af * (ep - psar.iloc[i - 1])
                if high.iloc[i] > psar.iloc[i]:
                    trend[i] = 1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = low.iloc[i]
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_increment, af_max)
        return psar

    # Ichimoku Cloud — comprehensive indicator with support/resistance and momentum
    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
    ) -> pd.DataFrame:
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_b = (
            (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
        ).shift(kijun)
        chikou = close.shift(-kijun)
        return pd.DataFrame(
            {
                "tenkan": tenkan_sen,
                "kijun": kijun_sen,
                "senkou_a": senkou_a,
                "senkou_b": senkou_b,
                "chikou": chikou,
            }
        )
