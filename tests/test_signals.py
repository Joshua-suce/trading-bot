import numpy as np
import pandas as pd

from src.indicators.compute import compute_all_indicators
from src.models.ensemble import ModelEnsemble
from src.signals.aggregator import FinalSignal, SignalAggregator
from src.signals.ta_signal import TechnicalSignal


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
