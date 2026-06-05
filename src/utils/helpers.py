# Utility functions — rounding, time conversions, mock OHLCV generation
import time
from decimal import ROUND_DOWN, Decimal

import numpy as np
import pandas as pd


# Round a quantity down to the exchange's step size (e.g. 0.001 BTC)
def round_step_size(quantity: float, step_size: float) -> float:
    quantize = Decimal(str(step_size))
    return float(Decimal(str(quantity)).quantize(quantize, rounding=ROUND_DOWN))


# Round a price down to the exchange's tick size (e.g. 0.01 USDT)
def round_tick_size(price: float, tick_size: float) -> float:
    quantize = Decimal(str(tick_size))
    return float(Decimal(str(price)).quantize(quantize, rounding=ROUND_DOWN))


# Current Unix timestamp in milliseconds
def timestamp_ms() -> int:
    return int(time.time() * 1000)


# Convert a timeframe string ("1h", "4h", "1d") to milliseconds
def timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    match unit:
        case "m":
            return value * 60_000
        case "h":
            return value * 3_600_000
        case "d":
            return value * 86_400_000
        case "w":
            return value * 604_800_000
        case _:
            raise ValueError(f"Unknown timeframe unit: {unit}")


# Annualised volatility by timeframe for realistic mock data
_TIMEFRAME_ANN_VOL = {
    "1m": 0.80,
    "5m": 0.75,
    "15m": 0.70,
    "1h": 0.65,
    "4h": 0.60,
    "1d": 0.55,
}


# Generate OHLC from a known open, using a volatility and drift estimate
def _generate_candle(
    open_px: float,
    dt_years: float,
    ann_vol: float,
    drift: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    sigma = ann_vol * np.sqrt(dt_years)
    mu = drift * dt_years
    close = open_px * np.exp(mu + sigma * rng.normal())
    half_range = open_px * sigma * rng.uniform(0.5, 1.2)
    high = max(open_px, close) + half_range * rng.uniform(0.0, 0.8)
    low = min(open_px, close) - half_range * rng.uniform(0.0, 0.8)
    return open_px, high, low, close


# Generate a geometric Brownian motion OHLCV DataFrame for offline backtesting
def generate_mock_ohlcv(
    symbol: str, timeframe: str = "1h", limit: int = 500
) -> pd.DataFrame:
    interval_ms = timeframe_to_ms(timeframe)
    dt_years = interval_ms / (86_400_000 * 365.25)
    ann_vol = _TIMEFRAME_ANN_VOL.get(timeframe, 0.65)

    end = pd.Timestamp.now("UTC").floor("s")
    timestamps = [
        end - pd.Timedelta(milliseconds=interval_ms * (limit - i - 1))
        for i in range(limit)
    ]

    base_price = 100.0
    if symbol.upper().endswith("USDT"):
        base_price = 50_000.0
    elif symbol.upper().endswith("BTC"):
        base_price = 30_000.0

    rng = np.random.default_rng(seed=42)
    drift = rng.uniform(-0.05, 0.05)

    opens = np.empty(limit)
    highs = np.empty(limit)
    lows = np.empty(limit)
    closes = np.empty(limit)
    volumes = np.empty(limit)

    price = base_price
    for i in range(limit):
        opens[i], highs[i], lows[i], closes[i] = _generate_candle(
            price, dt_years, ann_vol, drift, rng
        )
        price = closes[i]
        volumes[i] = rng.lognormal(mean=0, sigma=0.5) * 100

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df
