from dataclasses import dataclass

import pandas as pd

from src.strategies.base import StrategyMath


@dataclass
class MarketRegime:
    trend_direction: int
    market_type: str
    strength: float
    adx: float
    atr_pct: float
    bb_width_pct: float
    volume_ratio: float


def detect_regime(df: pd.DataFrame) -> MarketRegime:  # noqa: C901
    if df.empty:
        return MarketRegime(0, "unknown", 0.0, 0.0, 0.0, 0.0, 0.0)

    last = df.iloc[-1]
    # safe_float (not bare float()) matters here: a warmup-period NaN in an
    # existing column would otherwise propagate silently into every regime
    # comparison below instead of falling back to the default.
    close = StrategyMath.safe_float(last.get("close"), 0.0)
    adx = StrategyMath.safe_float(last.get("adx"), 0.0)
    atr = StrategyMath.safe_float(last.get("atr"), 0.0)
    atr_pct = atr / close if close > 0 else 0.0
    bb_upper = StrategyMath.safe_float(last.get("bb_upper"), 0.0)
    bb_lower = StrategyMath.safe_float(last.get("bb_lower"), 0.0)
    bb_width = (
        (bb_upper - bb_lower) / close if close > 0 and bb_upper > bb_lower else 0.0
    )

    ema_50 = StrategyMath.safe_float(last.get("ema_50"), close)
    ema_200 = StrategyMath.safe_float(last.get("ema_200"), ema_50)
    ema_slope = StrategyMath.safe_float(last.get("ema_50_slope"), 0.0)
    plus_di = StrategyMath.safe_float(last.get("plus_di"), 0.0)
    minus_di = StrategyMath.safe_float(last.get("minus_di"), 0.0)
    volume_ratio = StrategyMath.safe_float(last.get("vol_ratio"), 1.0)

    if close > ema_50 > ema_200 and ema_slope > 0:
        trend_direction = 1
    elif close < ema_50 < ema_200 and ema_slope < 0:
        trend_direction = -1
    elif plus_di > minus_di and adx >= 20:
        trend_direction = 1
    elif minus_di > plus_di and adx >= 20:
        trend_direction = -1
    else:
        trend_direction = 0

    lookback = min(60, len(df) - 1)
    atr_values = df["atr"].iloc[-lookback:] if "atr" in df.columns else pd.Series([atr])
    median_atr = float(atr_values.median()) if len(atr_values) > 0 else atr
    atr_expansion = atr / median_atr if median_atr > 0 else 1.0

    if adx >= 25 and atr_expansion <= 1.3:
        market_type = "trending"
        strength = min(adx / 50, 1.0)
    elif adx >= 25 and atr_expansion > 1.3:
        market_type = "volatile"
        strength = min(atr_expansion / 2, 1.0)
    elif adx >= 20:
        if atr_expansion > 1.2:
            market_type = "volatile"
            strength = min(atr_expansion / 2, 1.0)
        elif bb_width <= 0.04:
            market_type = "squeeze"
            strength = max(0.5, 1.0 - bb_width * 10)
        else:
            market_type = "transition"
            strength = 0.5
    elif bb_width <= 0.04:
        market_type = "squeeze"
        strength = max(0.5, 1.0 - bb_width * 10)
    else:
        market_type = "ranging"
        strength = max(0.3, 1.0 - adx / 20)

    return MarketRegime(
        trend_direction=trend_direction,
        market_type=market_type,
        strength=strength,
        adx=adx,
        atr_pct=atr_pct,
        bb_width_pct=bb_width,
        volume_ratio=volume_ratio,
    )


def regime_appropriate_strategies(regime: MarketRegime) -> list[str]:
    match regime.market_type:
        case "trending":
            if regime.trend_direction != 0:
                # Pullback entries are generated under the "trend" strategy
                # (TrendStrategy.pullback_signal); there is no separate
                # "pullback" strategy name to allow-list here.
                return ["trend", "breakout", "transition"]
            return ["range", "transition"]
        case "ranging":
            return ["range", "reversal", "scalp"]
        case "volatile":
            return ["countertrend", "scalp", "breakout"]
        case "squeeze":
            return ["breakout", "range", "scalp"]
        case "transition":
            return ["transition", "trend", "breakout"]
        case _:
            return ["trend", "range"]
