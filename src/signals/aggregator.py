# Signal fusion — combines TA (40%) and ML (60%) signals into a single decision
from dataclasses import dataclass

from loguru import logger

from src.indicators.compute import compute_all_indicators
from src.models.ensemble import EnsembleSignal, ModelEnsemble
from src.models.feature_engineer import FeatureEngineer
from src.signals.ta_signal import TechnicalSignal


# The final fused signal returned to the trading loop
@dataclass
class FinalSignal:
    direction: int  # -1 sell, 0 hold, 1 buy
    confidence: float
    ta_source: str
    ml_strength: float
    ml_confidence: float


class SignalAggregator:
    # Initialise with an ML ensemble and configurable TA/ML weights
    def __init__(
        self,
        ml_ensemble: ModelEnsemble,
        ta_weight: float = 0.4,
        ml_weight: float = 0.6,
        min_ta_strength: float = 0.20,
        min_ml_strength: float = 0.35,
        decision_threshold: float = 0.30,
    ):
        self.ta = TechnicalSignal()
        self.ensemble = ml_ensemble
        self.ta_weight = ta_weight
        self.ml_weight = ml_weight
        self.min_ta_strength = min_ta_strength
        self.min_ml_strength = min_ml_strength
        self.decision_threshold = decision_threshold
        self.feature_engineer = FeatureEngineer()

    # Compute indicators, get TA signal, optionally get ML signal, then fuse
    def generate(self, df) -> FinalSignal:
        df_ind = compute_all_indicators(df)

        # TA signal
        ta_signal = self.ta.generate(df_ind)

        # ML signal
        ml_signal = EnsembleSignal(
            direction=0, confidence=0.0, strength=0.0, sources={}
        )
        if self.ensemble.is_ready():
            try:
                df_feat = self.feature_engineer.create_features(df_ind)
                if len(df_feat) > 0:
                    X = self.feature_engineer.get_feature_matrix(df_feat)
                    if len(X) > 0:
                        X_3d = X[-1:].reshape(1, -1) if X.ndim == 1 else X[-1:]
                        ml_signal = self.ensemble.predict(X_3d)
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

        active_signals = []
        if ta_signal.direction != 0 and ta_signal.strength >= self.min_ta_strength:
            active_signals.append(
                (self.ta_weight, ta_signal.direction, ta_signal.strength)
            )
        if (
            ml_signal.direction != 0
            and ml_signal.strength >= self.min_ml_strength
            and ml_signal.confidence >= self.ensemble.confidence_threshold
        ):
            active_signals.append(
                (self.ml_weight, ml_signal.direction, ml_signal.strength)
            )

        if not active_signals:
            combined = 0.0
            confidence = 0.0
            direction = 0
        else:
            total_weight = sum(weight for weight, _, _ in active_signals)
            combined = (
                sum(
                    weight * direction * strength
                    for weight, direction, strength in active_signals
                )
                / total_weight
            )
            confidence = abs(combined)
            if confidence < self.decision_threshold:
                direction = 0
            else:
                direction = 1 if combined > 0 else -1

        return FinalSignal(
            direction=direction,
            confidence=min(confidence, 1.0),
            ta_source=ta_signal.source,
            ml_strength=ml_signal.strength,
            ml_confidence=ml_signal.confidence,
        )
