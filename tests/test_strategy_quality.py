import pandas as pd

from src.signals.quality_gate import StrategyQualityGate


def quality_frame(**overrides):
    row = {
        "open": 103.5,
        "high": 106.0,
        "low": 101.0,
        "close": 105.0,
        "ema_50": 103.0,
        "ema_200": 100.0,
        "ema_50_slope": 0.01,
        "adx": 30.0,
        "plus_di": 28.0,
        "minus_di": 14.0,
        "rsi_14": 56.0,
        "atr": 2.0,
        "atr_pct": 0.019,
        "vol_ratio": 1.2,
        "trend_regime": 1,
        "macd_hist": 0.5,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_quality_gate_accepts_aligned_intraday_signal():
    result = StrategyQualityGate().evaluate(
        quality_frame(),
        1,
        timeframe="15m",
        higher_timeframe_regime=1,
    )

    assert result.accepted
    assert result.score >= 0.68


def test_quality_gate_rejects_intraday_higher_timeframe_disagreement():
    result = StrategyQualityGate().evaluate(
        quality_frame(),
        1,
        timeframe="15m",
        higher_timeframe_regime=-1,
    )

    assert not result.accepted
    assert result.reason == "1h trend regime disagrees"


def test_quality_gate_rejects_intraday_trend_without_higher_timeframe_alignment():
    result = StrategyQualityGate().evaluate(
        quality_frame(),
        1,
        timeframe="5m",
        higher_timeframe_regime=0,
        strategy="trend",
    )

    assert not result.accepted
    assert result.reason == "1h trend regime is not aligned"


def test_quality_gate_scores_emerging_trend_in_neutral_local_regime():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            close=99.0,
            ema_50=101.0,
            ema_200=100.0,
            ema_50_slope=-0.01,
            plus_di=14.0,
            minus_di=28.0,
            rsi_14=40.0,
            trend_regime=0,
            macd_hist=-0.5,
        ),
        -1,
        timeframe="1h",
    )

    assert result.accepted
    assert result.score >= 0.68


def test_quality_gate_still_rejects_opposite_local_regime():
    result = StrategyQualityGate().evaluate(
        quality_frame(trend_regime=1),
        -1,
        timeframe="1h",
    )

    assert not result.accepted
    assert result.reason == "local trend regime opposes signal"


def test_quality_gate_accepts_range_mean_reversion():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            adx=15.0,
            trend_regime=0,
            rsi_14=35.0,
            bb_percent_b=0.1,
            macd_hist=0.1,
        ),
        1,
        timeframe="5m",
        higher_timeframe_regime=-1,
        strategy="range",
    )

    assert result.accepted
    assert result.reason == "range mean-reversion quality confirmed"


def test_quality_gate_accepts_confirmed_reversal():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            open=106.0,
            high=108.0,
            low=102.0,
            close=103.0,
            trend_regime=1,
            rsi_14=72.0,
            plus_di=14.0,
            minus_di=28.0,
            macd_hist=-0.5,
            vol_ratio=1.2,
        ),
        -1,
        timeframe="1h",
        strategy="reversal",
    )

    assert result.accepted
    assert result.score == 1.0


def test_quality_gate_allows_breakout_from_neutral_higher_timeframe():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            open=101.0,
            high=106.0,
            low=100.0,
            close=105.0,
            vol_ratio=1.4,
        ),
        1,
        timeframe="15m",
        higher_timeframe_regime=0,
        strategy="breakout",
    )

    assert result.accepted
    assert result.reason == "breakout quality confirmed"


def test_quality_gate_rejects_overextended_breakout():
    result = StrategyQualityGate(breakout_max_extension_atr=3.0).evaluate(
        quality_frame(
            open=108.0,
            high=113.0,
            low=107.0,
            close=112.0,
            ema_50=103.0,
            atr=2.0,
            vol_ratio=1.4,
        ),
        1,
        timeframe="15m",
        higher_timeframe_regime=1,
        strategy="breakout",
    )

    assert not result.accepted
    assert result.reason == "breakout price extension too high"
    assert result.metrics["extension_atr"] == 4.5


def test_quality_gate_rejects_weak_breakout_candle():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            open=104.8,
            high=106.0,
            low=100.0,
            close=105.0,
            vol_ratio=1.4,
        ),
        1,
        timeframe="15m",
        higher_timeframe_regime=1,
        strategy="breakout",
    )

    assert not result.accepted
    assert result.reason == "breakout candle structure too weak"


def test_quality_gate_rejects_range_touch_without_rejection_candle():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            open=105.0,
            high=105.5,
            low=100.0,
            close=100.5,
            adx=15.0,
            trend_regime=0,
            rsi_14=35.0,
            bb_percent_b=0.1,
            macd_hist=0.1,
        ),
        1,
        timeframe="5m",
        higher_timeframe_regime=0,
        strategy="range",
    )

    assert not result.accepted
    assert result.reason == "range rejection candle not confirmed"


def test_quality_gate_rejects_unconfirmed_countertrend_signal():
    result = StrategyQualityGate().evaluate(
        quality_frame(),
        -1,
        timeframe="1h",
        strategy="countertrend",
    )

    assert not result.accepted
    assert "price not overextended" in result.reason


def test_quality_gate_accepts_countertrend_with_overextended_rejection():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            open=106.0,
            high=109.0,
            low=98.0,
            close=98.0,
            trend_regime=1,
            rsi_14=28.0,
            plus_di=28.0,
            minus_di=14.0,
            adx=30.0,
            macd_hist=0.5,
            vol_ratio=2.0,
        ),
        -1,
        timeframe="1h",
        strategy="countertrend",
    )

    assert result.accepted
    assert "overextended" in result.reason


def test_quality_gate_rejects_exhausted_entry():
    result = StrategyQualityGate(min_score=0.95).evaluate(
        quality_frame(close=112.0, rsi_14=76.0),
        1,
        timeframe="1h",
    )

    assert not result.accepted
    assert "quality score below threshold" in result.reason


def test_quality_gate_accepts_confirmed_scalp():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            close=105.0,
            ema_50=103.0,
            vwap=103.5,
            stoch_k=58.0,
            stoch_d=50.0,
            macd_hist=0.2,
            vol_ratio=1.2,
        ),
        1,
        timeframe="1m",
        higher_timeframe_regime=1,
        strategy="scalp",
        signal_source="scalp_vwap_trend_bull+scalp_score_0.82",
    )

    assert result.accepted
    assert result.reason == "scalp structure and momentum confirmed"


def test_quality_gate_accepts_confirmed_range_scalp_below_vwap():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            open=99.8,
            high=101.2,
            low=98.0,
            close=100.5,
            ema_50=102.0,
            vwap=102.2,
            rsi_14=39.0,
            stoch_k=34.0,
            stoch_d=28.0,
            macd_hist=-0.1,
            vol_ratio=0.95,
            bb_percent_b=0.12,
        ),
        1,
        timeframe="3m",
        strategy="scalp",
        signal_source="scalp_range_low+scalp_score_0.82",
    )

    assert result.accepted
    assert result.metrics["scalp_setup_type"] == "range"


def test_quality_gate_rejects_scalp_on_swing_timeframe():
    result = StrategyQualityGate().evaluate(
        quality_frame(vwap=103.5, stoch_k=58.0, stoch_d=50.0),
        1,
        timeframe="5m",
        strategy="scalp",
    )

    assert not result.accepted
    assert result.reason == "scalp strategy requires 1m or 3m timeframe"


def test_quality_gate_rejects_trend_scalp_against_one_hour_regime():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            close=105.0,
            ema_50=103.0,
            vwap=103.5,
            stoch_k=58.0,
            stoch_d=50.0,
            macd_hist=0.2,
            vol_ratio=1.2,
        ),
        1,
        timeframe="1m",
        higher_timeframe_regime=-1,
        strategy="scalp",
        signal_source="scalp_pullback_bull+scalp_score_0.88",
    )

    assert not result.accepted
    assert result.reason == "1h trend regime is not aligned with trend scalp"


def test_quality_gate_rejects_trend_scalp_in_neutral_one_hour_regime():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            close=105.0,
            ema_50=103.0,
            vwap=103.5,
            stoch_k=58.0,
            stoch_d=50.0,
            macd_hist=0.2,
            vol_ratio=1.2,
        ),
        1,
        timeframe="1m",
        higher_timeframe_regime=0,
        strategy="scalp",
        signal_source="scalp_pullback_bull+scalp_score_0.88",
    )

    assert not result.accepted
    assert result.reason == "1h trend regime is not aligned with trend scalp"


def test_quality_gate_requires_one_hour_context_for_trend_scalp():
    result = StrategyQualityGate().evaluate(
        quality_frame(
            close=105.0,
            ema_50=103.0,
            vwap=103.5,
            stoch_k=58.0,
            stoch_d=50.0,
            macd_hist=0.2,
            vol_ratio=1.2,
        ),
        1,
        timeframe="1m",
        strategy="scalp",
        signal_source="scalp_pullback_bull+scalp_score_0.88",
    )

    assert not result.accepted
    assert result.reason == "waiting for 1h trend context"
