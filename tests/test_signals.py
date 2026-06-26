import numpy as np
import pandas as pd
import pytest

from src.indicators.compute import compute_all_indicators
from src.models.ensemble import EnsembleSignal, ModelEnsemble
from src.signals.aggregator import FinalSignal, SignalAggregator
from src.signals.ta_signal import TASignal, TechnicalSignal


def test_technical_signal_handles_short_history():
    df = pd.DataFrame(
        {
            "open": [50000.0],
            "high": [50100.0],
            "low": [49950.0],
            "close": [50050.0],
            "volume": [1000.0],
        }
    )
    df = compute_all_indicators(df)
    signal = TechnicalSignal().generate(df)
    assert isinstance(signal.direction, int)
    assert 0 <= signal.strength <= 1


def test_signal_aggregator_returns_final_signal():
    df = pd.DataFrame(
        {
            "open": [50000.0, 50050.0, 50100.0],
            "high": [50100.0, 50150.0, 50180.0],
            "low": [49950.0, 50040.0, 50080.0],
            "close": [50050.0, 50100.0, 50120.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )
    aggregator = SignalAggregator(ModelEnsemble())
    signal = aggregator.generate(df)
    assert isinstance(signal, FinalSignal)
    assert signal.direction in {-1, 0, 1}
    assert 0.0 <= signal.confidence <= 1.0


def test_compute_all_indicators_is_idempotent():
    df = pd.DataFrame(
        {
            "open": np.linspace(50000, 51000, 220),
            "high": np.linspace(50100, 51100, 220),
            "low": np.linspace(49900, 50900, 220),
            "close": np.linspace(50050, 51050, 220),
            "volume": np.linspace(1000, 1500, 220),
        }
    )

    once = compute_all_indicators(df)
    twice = compute_all_indicators(once)

    assert not twice.columns.duplicated().any()
    assert "macd_hist" in twice.columns
    assert "bb_percent_b" in twice.columns


def test_technical_signal_blends_bullish_ema_and_fibonacci_confluence():
    row = {
        "open": 109.0,
        "high": 111.0,
        "low": 107.8,
        "close": 110.0,
        "volume": 1000.0,
        "ema_50": 105.0,
        "ema_200": 100.0,
        "ema_50_slope": 0.01,
        "macd_hist": 1.0,
        "adx": 30.0,
        "plus_di": 30.0,
        "minus_di": 15.0,
        "trend_regime": 1,
        "rsi_14": 55.0,
        "vol_ratio": 1.2,
        "atr": 4.0,
        "fib_382": 115.0,
        "fib_500": 110.0,
        "fib_618": 105.0,
    }
    previous = {
        **row,
        "high": 109.0,
        "low": 106.0,
        "close": 108.0,
        "macd_hist": 0.5,
        "fib_500": 110.0,
    }

    signal = TechnicalSignal().generate(pd.DataFrame([previous, row]))

    assert signal.direction == 1
    assert signal.strength > 0


def test_technical_signal_requires_buffered_breakout_close():
    previous = {
        "open": 99.0,
        "high": 100.0,
        "low": 98.0,
        "close": 99.5,
        "ema_50": 98.0,
        "ema_200": 95.0,
        "ema_50_slope": 0.01,
        "macd_hist": 0.2,
        "adx": 30.0,
        "plus_di": 30.0,
        "minus_di": 15.0,
        "trend_regime": 1,
        "rsi_14": 55.0,
        "vol_ratio": 1.2,
        "atr": 2.0,
        "dc_upper": 100.0,
        "dc_lower": 90.0,
    }
    weak_break = {
        **previous,
        "open": 99.9,
        "high": 100.3,
        "low": 99.7,
        "close": 100.05,
    }

    signal = TechnicalSignal().generate(pd.DataFrame([previous, weak_break]))

    assert "donchian_breakout" not in signal.source


def test_technical_signal_detects_confirmed_ema50_pullback():
    previous = {
        "open": 104.0,
        "high": 105.0,
        "low": 102.0,
        "close": 103.0,
        "ema_50": 102.0,
        "ema_200": 95.0,
        "ema_50_slope": 0.01,
        "macd_hist": 0.2,
        "adx": 30.0,
        "plus_di": 30.0,
        "minus_di": 15.0,
        "trend_regime": 1,
        "rsi_14": 52.0,
        "vol_ratio": 1.2,
        "atr": 4.0,
    }
    pullback = {
        **previous,
        "open": 102.0,
        "high": 105.0,
        "low": 100.0,
        "close": 104.0,
        "ema_50": 102.5,
        "macd_hist": 0.3,
    }

    signal = TechnicalSignal().generate(pd.DataFrame([previous, pullback]))

    assert signal.direction == 1
    assert "trend_pullback_ema50_bull" in signal.source


def test_correlated_rsi_votes_are_capped_to_one_family():
    signals = [
        TASignal(1, 0.55, "rsi_oversold"),
        TASignal(1, 0.30, "rsi_bullish"),
        TASignal(-1, 0.50, "macd_bear_cross"),
    ]

    capped = TechnicalSignal._strongest_family_votes(signals)

    assert len(capped) == 2
    assert any(signal.source == "rsi_oversold" for signal in capped)


def test_single_family_trend_signal_is_not_actionable():
    previous = {
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "ema_50": 98.0,
        "ema_200": 95.0,
        "ema_50_slope": 0.01,
        "macd_hist": 0.1,
        "adx": 30.0,
        "plus_di": 30.0,
        "minus_di": 15.0,
        "trend_regime": 1,
        "rsi_14": 55.0,
        "vol_ratio": 1.2,
        "atr": 2.0,
    }
    current = {
        **previous,
        "open": 101.0,
        "high": 104.0,
        "low": 100.5,
        "close": 103.5,
        "macd_hist": 0.2,
    }

    signal = TechnicalSignal().generate(pd.DataFrame([previous, current]))

    assert signal.direction == 0
    assert signal.strategy == "trend"


def test_one_minute_scalp_detects_confirmed_pullback_entry():
    previous = {
        "open": 100.0,
        "high": 100.5,
        "low": 99.5,
        "close": 99.8,
        "ema_50": 100.0,
        "ema_50_slope": 0.005,
        "vwap": 100.3,
        "macd_hist": -0.05,
        "rsi_14": 46.0,
        "stoch_k": 35.0,
        "stoch_d": 38.0,
        "vol_ratio": 1.0,
        "bb_percent_b": 0.4,
        "adx": 20.0,
    }
    current = {
        **previous,
        "open": 99.5,
        "high": 100.2,
        "low": 99.4,
        "close": 100.1,
        "ema_50": 100.0,
        "vwap": 100.2,
        "macd_hist": 0.0,
        "stoch_k": 42.0,
        "stoch_d": 39.0,
        "vol_ratio": 1.1,
    }
    index = pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC")

    signal = TechnicalSignal().generate(
        pd.DataFrame([previous, current], index=index), higher_trend_bias=1
    )

    assert signal.direction == 1
    assert signal.strategy == "scalp"
    assert "scalp_pullback_bull" in signal.source


def test_one_minute_without_scalp_setup_does_not_use_swing_fallback():
    previous = {
        "open": 99.0,
        "high": 100.0,
        "low": 98.0,
        "close": 99.0,
        "ema_50": 98.0,
        "ema_200": 100.0,
        "ema_50_slope": 0.01,
        "vwap": 99.0,
        "macd_hist": -0.1,
        "rsi_14": 55.0,
        "stoch_k": 60.0,
        "stoch_d": 50.0,
        "vol_ratio": 0.5,
        "bb_percent_b": 0.5,
        "adx": 30.0,
    }
    current = {
        **previous,
        "close": 101.0,
        "ema_50": 101.0,
        "ema_200": 100.0,
        "macd_hist": 0.1,
    }
    index = pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC")

    signal = TechnicalSignal().generate(
        pd.DataFrame([previous, current], index=index), higher_trend_bias=1
    )

    assert signal.direction == 0
    assert signal.strategy == "scalp"
    assert signal.source == "scalp_no_setup"


def test_three_minute_scalp_detects_pullback_without_stoch_cross():
    previous = {
        "open": 100.2,
        "high": 100.8,
        "low": 99.8,
        "close": 100.0,
        "ema_50": 100.0,
        "ema_50_slope": 0.01,
        "vwap": 100.3,
        "macd_hist": -0.02,
        "rsi_14": 47.0,
        "stoch_k": 35.0,
        "stoch_d": 38.0,
        "vol_ratio": 1.0,
        "bb_percent_b": 0.4,
        "adx": 20.0,
    }
    current = {
        **previous,
        "open": 99.9,
        "high": 100.4,
        "low": 99.7,
        "close": 100.2,
        "ema_50": 100.0,
        "vwap": 100.3,
        "macd_hist": 0.03,
        "stoch_k": 41.0,
        "stoch_d": 38.0,
        "vol_ratio": 1.1,
    }
    index = pd.date_range("2026-01-01", periods=2, freq="3min", tz="UTC")

    signal = TechnicalSignal().generate(
        pd.DataFrame([previous, current], index=index), higher_trend_bias=1
    )

    assert signal.direction == 1
    assert signal.strategy == "scalp"
    assert signal.strength >= 0.60
    assert signal.source.startswith("scalp_pullback_bull")


def test_scalp_still_rejects_low_volume_directional_noise():
    previous = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.8,
        "close": 100.6,
        "ema_50": 99.5,
        "ema_50_slope": 0.03,
        "vwap": 99.8,
        "macd_hist": 0.1,
        "rsi_14": 55.0,
        "stoch_k": 55.0,
        "stoch_d": 48.0,
        "vol_ratio": 0.60,
        "bb_percent_b": 0.7,
        "adx": 24.0,
    }
    current = {**previous, "close": 101.0, "macd_hist": 0.2}
    index = pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC")

    signal = TechnicalSignal().generate(pd.DataFrame([previous, current], index=index))

    assert signal.direction == 0
    assert signal.source == "scalp_no_setup"


def test_non_scalp_trend_structure_can_activate_without_extra_family():
    previous = {
        "open": 104.0,
        "high": 105.0,
        "low": 103.0,
        "close": 104.5,
        "ema_50": 101.0,
        "ema_200": 98.0,
        "ema_50_slope": 0.04,
        "macd_hist": 0.15,
        "rsi_14": 58.0,
        "vol_ratio": 1.15,
        "bb_percent_b": 0.65,
        "plus_di": 31.0,
        "minus_di": 14.0,
        "adx": 26.0,
        "trend_regime": 1,
        "atr": 2.0,
    }
    current = {
        **previous,
        "open": 104.4,
        "high": 106.0,
        "low": 104.0,
        "close": 105.6,
        "macd_hist": 0.22,
    }
    index = pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC")

    signal = TechnicalSignal().generate(pd.DataFrame([previous, current], index=index))

    assert signal.direction == 1
    assert signal.strategy == "trend"
    assert "trend_structure_bull" in signal.source


def test_non_scalp_breakout_can_activate_as_standalone_structure():
    previous = {
        "open": 101.0,
        "high": 103.0,
        "low": 100.0,
        "close": 102.0,
        "ema_50": 100.0,
        "ema_200": 98.0,
        "ema_50_slope": 0.03,
        "macd_hist": 0.1,
        "rsi_14": 57.0,
        "vol_ratio": 1.40,
        "bb_percent_b": 0.70,
        "plus_di": 28.0,
        "minus_di": 12.0,
        "adx": 24.0,
        "trend_regime": 1,
        "atr": 2.0,
        "dc_upper": 103.0,
        "dc_lower": 96.0,
    }
    current = {
        **previous,
        "open": 103.2,
        "high": 105.2,
        "low": 103.0,
        "close": 105.0,
        "dc_upper": 104.0,
    }
    index = pd.date_range("2026-01-01", periods=2, freq="30min", tz="UTC")

    signal = TechnicalSignal().generate(pd.DataFrame([previous, current], index=index))

    assert signal.direction == 1
    assert signal.strategy == "breakout"
    assert "donchian_breakout_bull" in signal.source


class FakeXgbModel:
    def __init__(self, direction: int, confidence: float):
        self.direction = direction
        self.confidence = confidence

    def predict_with_confidence(self, X):
        return np.array([self.direction]), np.array([self.confidence])


def test_signal_aggregator_filters_low_confidence_ml_noise():
    df = pd.DataFrame(
        {
            "open": np.linspace(50000, 50100, 220),
            "high": np.linspace(50100, 50200, 220),
            "low": np.linspace(49900, 50000, 220),
            "close": np.linspace(50050, 50150, 220),
            "volume": np.linspace(1000, 1100, 220),
        }
    )
    ensemble = ModelEnsemble(
        xgb_model=FakeXgbModel(direction=1, confidence=0.4),
        confidence_threshold=0.6,
    )

    signal = SignalAggregator(
        ensemble,
        ta_weight=0.0,
        ml_weight=1.0,
        min_ta_strength=1.0,
        min_ml_strength=0.1,
        decision_threshold=0.1,
    ).generate(df)

    assert signal.direction == 0


def test_signal_aggregator_accepts_high_confidence_ml_signal():
    df = pd.DataFrame(
        {
            "open": np.linspace(50000, 50100, 220),
            "high": np.linspace(50100, 50200, 220),
            "low": np.linspace(49900, 50000, 220),
            "close": np.linspace(50050, 50150, 220),
            "volume": np.linspace(1000, 1100, 220),
        }
    )
    ensemble = ModelEnsemble(
        xgb_model=FakeXgbModel(direction=-1, confidence=0.9),
        confidence_threshold=0.6,
    )

    signal = SignalAggregator(
        ensemble,
        ta_weight=0.0,
        ml_weight=1.0,
        min_ta_strength=1.0,
        min_ml_strength=0.1,
        decision_threshold=0.1,
    ).generate(df)

    assert signal.direction == -1
    assert signal.confidence >= 0.9


def test_signal_confluence_blocks_disagreement(monkeypatch):
    ensemble = ModelEnsemble(
        xgb_model=FakeXgbModel(direction=-1, confidence=0.9),
        confidence_threshold=0.6,
    )
    aggregator = SignalAggregator(
        ensemble,
        min_ta_strength=0.1,
        min_ml_strength=0.1,
        require_confluence=True,
    )
    monkeypatch.setattr(
        aggregator.ta,
        "generate",
        lambda _df: TASignal(direction=1, strength=0.9, source="bullish"),
    )
    monkeypatch.setattr(
        ensemble,
        "predict",
        lambda _x: EnsembleSignal(
            direction=-1,
            confidence=0.9,
            strength=0.9,
            sources={"xgb": -1},
        ),
    )
    aggregator.feature_engineer.create_features = lambda df: df
    aggregator.feature_engineer.get_feature_matrix = lambda _df: np.array([[1.0]])

    signal = aggregator.generate(
        pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
    )

    assert signal.direction == 0
    assert signal.decision_reason == "active ML opposes regime signal"


def test_trend_signal_can_proceed_when_ml_is_inactive(monkeypatch):
    ensemble = ModelEnsemble(
        xgb_model=FakeXgbModel(direction=0, confidence=0.4),
        confidence_threshold=0.55,
    )
    aggregator = SignalAggregator(
        ensemble,
        min_ta_strength=0.1,
        min_ml_strength=0.1,
        decision_threshold=0.3,
        require_confluence=True,
    )
    monkeypatch.setattr(
        aggregator.ta,
        "generate",
        lambda _df: TASignal(
            direction=1,
            strength=0.75,
            source="trend_structure_bull+structure_score_0.88",
            strategy="trend",
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "predict",
        lambda _x: EnsembleSignal(
            direction=0,
            confidence=0.4,
            strength=0.0,
            sources={"xgb": 0},
        ),
    )
    aggregator.feature_engineer.create_features = lambda df: df
    aggregator.feature_engineer.get_feature_matrix = lambda _df: np.array([[1.0]])

    signal = aggregator.generate(
        pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
    )

    assert signal.direction == 1
    assert signal.strategy == "trend"
    assert signal.decision_reason == "aligned signal ready"


def test_range_signal_reports_weak_ta_instead_of_confluence_failure(monkeypatch):
    ensemble = ModelEnsemble(
        xgb_model=FakeXgbModel(direction=0, confidence=0.4),
        confidence_threshold=0.55,
    )
    aggregator = SignalAggregator(
        ensemble,
        min_ta_strength=0.2,
        require_confluence=True,
    )
    monkeypatch.setattr(
        aggregator.ta,
        "generate",
        lambda _df: TASignal(
            direction=-1,
            strength=0.15,
            source="rsi_bearish+bb_upper_reject",
            strategy="range",
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "predict",
        lambda _x: EnsembleSignal(
            direction=0,
            confidence=0.4,
            strength=0.0,
            sources={"xgb": 0},
        ),
    )
    aggregator.feature_engineer.create_features = lambda df: df
    aggregator.feature_engineer.get_feature_matrix = lambda _df: np.array([[1.0]])

    signal = aggregator.generate(
        pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
    )

    assert signal.direction == 0
    assert signal.decision_reason == "TA signal below activation threshold"
    assert signal.confidence == 0.0


def test_range_signal_can_proceed_when_ml_is_inactive(monkeypatch):
    ensemble = ModelEnsemble(
        xgb_model=FakeXgbModel(direction=0, confidence=0.4),
        confidence_threshold=0.55,
    )
    aggregator = SignalAggregator(
        ensemble,
        min_ta_strength=0.1,
        min_ml_strength=0.1,
        decision_threshold=0.3,
        require_confluence=True,
    )
    monkeypatch.setattr(
        aggregator.ta,
        "generate",
        lambda _df: TASignal(
            direction=1,
            strength=0.8,
            source="bb_lower_bounce+rsi_oversold",
            strategy="range",
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "predict",
        lambda _x: EnsembleSignal(
            direction=0,
            confidence=0.4,
            strength=0.0,
            sources={"xgb": 0},
        ),
    )
    aggregator.feature_engineer.create_features = lambda df: df
    aggregator.feature_engineer.get_feature_matrix = lambda _df: np.array([[1.0]])

    signal = aggregator.generate(
        pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
    )

    assert signal.direction == 1
    assert signal.strategy == "range"


def test_range_signal_is_blocked_by_active_opposing_ml(monkeypatch):
    ensemble = ModelEnsemble(
        xgb_model=FakeXgbModel(direction=-1, confidence=0.9),
        confidence_threshold=0.55,
    )
    aggregator = SignalAggregator(
        ensemble,
        min_ta_strength=0.1,
        min_ml_strength=0.1,
        require_confluence=True,
    )
    monkeypatch.setattr(
        aggregator.ta,
        "generate",
        lambda _df: TASignal(
            direction=1,
            strength=0.8,
            source="bb_lower_bounce+rsi_oversold",
            strategy="range",
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "predict",
        lambda _x: EnsembleSignal(
            direction=-1,
            confidence=0.9,
            strength=0.9,
            sources={"xgb": -1},
        ),
    )
    aggregator.feature_engineer.create_features = lambda df: df
    aggregator.feature_engineer.get_feature_matrix = lambda _df: np.array([[1.0]])

    signal = aggregator.generate(
        pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
    )

    assert signal.direction == 0
    assert signal.decision_reason == "active ML opposes regime signal"


def test_signal_fusion_uses_regime_specific_weights():
    aggregator = SignalAggregator(ModelEnsemble())

    assert aggregator._strategy_weights("trend") == (0.4, 0.6)
    assert aggregator._strategy_weights("range") == pytest.approx((0.7, 0.3))
    assert aggregator._strategy_weights("breakout") == pytest.approx((0.6, 0.402))


def test_trend_scalp_rejects_pullback_on_wrong_side_of_ema():
    direction, score = TechnicalSignal._score_trend_scalp(
        direction=1,
        close=99.9,
        vwap=100.0,
        ema_50=100.0,
        ema_slope=0.01,
        macd=0.1,
        previous_macd=0.0,
        rsi=50.0,
        stoch_k=45.0,
        stoch_d=40.0,
        exact_cross=True,
        volume_ratio=1.2,
        candle_confirmed=True,
        higher_trend_bias=1,
    )

    assert direction == 1
    assert score == 0.0


def test_trend_scalp_requires_directional_candle_and_participation():
    common = {
        "direction": 1,
        "close": 100.1,
        "vwap": 100.0,
        "ema_50": 100.0,
        "ema_slope": 0.01,
        "macd": 0.1,
        "previous_macd": 0.0,
        "rsi": 50.0,
        "stoch_k": 45.0,
        "stoch_d": 40.0,
        "exact_cross": True,
        "higher_trend_bias": 1,
    }

    assert (
        TechnicalSignal._score_trend_scalp(
            **common,
            volume_ratio=1.2,
            candle_confirmed=False,
        )[1]
        == 0.0
    )
    assert (
        TechnicalSignal._score_trend_scalp(
            **common,
            volume_ratio=0.99,
            candle_confirmed=True,
        )[1]
        == 0.0
    )


def test_common_trend_scalp_score_clears_configured_confidence_gate():
    confidence = TechnicalSignal._trend_scalp_confidence(0.78)

    assert confidence == pytest.approx(0.656)
    assert confidence >= 0.65
