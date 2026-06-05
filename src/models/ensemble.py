# Model ensemble — weighted voting between XGBoost and optional LSTM
from dataclasses import dataclass

import numpy as np
from loguru import logger


# Output from the ensemble prediction
@dataclass
class EnsembleSignal:
    direction: int
    confidence: float
    strength: float
    sources: dict


class ModelEnsemble:
    # XGBoost gets 60% weight, LSTM gets 40% by default
    def __init__(
        self,
        xgb_model=None,
        lstm_model=None,
        xgb_weight: float = 0.6,
        lstm_weight: float = 0.4,
        confidence_threshold: float = 0.6,
    ):
        self.xgb = xgb_model
        self.lstm = lstm_model
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.confidence_threshold = confidence_threshold

    # Weighted vote across all available sub-models
    def predict(self, X: np.ndarray) -> EnsembleSignal:
        signals = {}
        weights = []
        directions = []
        confidences = []

        if self.xgb is not None:
            try:
                xgb_dir, xgb_conf = self.xgb.predict_with_confidence(X)
                signals["xgb"] = {
                    "direction": int(xgb_dir[-1]),
                    "confidence": float(xgb_conf[-1]),
                }
                weights.append(self.xgb_weight)
                directions.append(xgb_dir[-1])
                confidences.append(xgb_conf[-1])
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")

        if self.lstm is not None:
            try:
                lstm_pred = self.lstm.predict(X)
                lstm_dir = np.sign(lstm_pred[-1])
                lstm_conf = min(abs(lstm_pred[-1]) / 100, 1.0)
                signals["lstm"] = {
                    "direction": int(lstm_dir),
                    "confidence": float(lstm_conf),
                }
                weights.append(self.lstm_weight)
                directions.append(lstm_dir)
                confidences.append(lstm_conf)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        if not signals:
            return EnsembleSignal(direction=0, confidence=0.0, strength=0.0, sources={})

        total_weight = sum(weights)
        weighted_sum = sum(
            w * d * c for w, d, c in zip(weights, directions, confidences, strict=True)
        )
        avg_confidence = (
            sum(w * c for w, c in zip(weights, confidences, strict=True)) / total_weight
        )

        if abs(weighted_sum / total_weight) < 0.1:
            direction = 0
        else:
            direction = 1 if weighted_sum > 0 else -1

        strength = abs(weighted_sum / total_weight)

        return EnsembleSignal(
            direction=direction,
            confidence=avg_confidence,
            strength=strength,
            sources=signals,
        )

    # Check whether any sub-model is available for prediction
    def is_ready(self) -> bool:
        return self.xgb is not None or self.lstm is not None
