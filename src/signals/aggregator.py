# Signal fusion — combines TA (40%) and ML (60%) signals into a single decision
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from src.config import settings
from src.indicators.compute import compute_all_indicators
from src.models.ensemble import EnsembleSignal, ModelEnsemble
from src.models.feature_engineer import FeatureEngineer
from src.signals.invocation import generate_with_context
from src.signals.ta_signal import TechnicalSignal


# The final fused signal returned to the trading loop
@dataclass
class FinalSignal:
    direction: int  # -1 sell, 0 hold, 1 buy
    confidence: float
    ta_source: str
    ml_strength: float
    ml_confidence: float
    decision_reason: str = ""
    strategy: str = "trend"


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
        require_confluence: bool = False,
        on_ml_degraded: Callable[[], None] | None = None,
        on_ml_recovered: Callable[[], None] | None = None,
    ):
        self.ta = TechnicalSignal()
        self.ensemble = ml_ensemble
        self.ta_weight = ta_weight
        self.ml_weight = ml_weight
        self.min_ta_strength = min_ta_strength
        self.min_ml_strength = min_ml_strength
        self.decision_threshold = decision_threshold
        self.require_confluence = require_confluence
        self.feature_engineer = FeatureEngineer(
            lookback=settings.feature_lookback,
            prediction_horizon=settings.prediction_horizon,
            label_atr_multiplier=settings.ml_label_atr_multiplier,
            label_min_return=settings.ml_effective_label_min_return,
        )
        self._ml_success_count = 0
        self._ml_failure_count = 0
        self._ml_healthy = True
        self._on_ml_degraded = on_ml_degraded
        self._on_ml_recovered = on_ml_recovered

    # Compute indicators, get TA signal, optionally get ML signal, then fuse
    def generate(self, df, higher_trend_bias: int = 0) -> FinalSignal:  # noqa: C901
        df_ind = compute_all_indicators(df)

        # TA signal
        ta_signal = generate_with_context(self.ta, df_ind, higher_trend_bias)
        if ta_signal.strategy == "countertrend" and not self.ensemble.is_ready():
            ta_signal.strategy = "transition"

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
                if not ml_signal.sources:
                    self._record_ml_failure()
                else:
                    self._record_ml_success()
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                self._record_ml_failure()

        ta_active = (
            ta_signal.direction != 0 and ta_signal.strength >= self.min_ta_strength
        )
        ml_active = (
            ml_signal.direction != 0
            and ml_signal.strength >= self.min_ml_strength
            and ml_signal.confidence >= self.ensemble.confidence_threshold
        )
        needs_ml_confirmation = ta_signal.strategy in {
            "transition",
            "countertrend",
        }
        confluence_failure_reason = ""
        if not ta_active:
            confluence_failure_reason = "TA signal below activation threshold"
        elif needs_ml_confirmation and not ml_active:
            confluence_failure_reason = "ML confirmation below activation threshold"
        elif ml_active and ta_signal.direction != ml_signal.direction:
            confluence_failure_reason = (
                "TA/ML directions disagree"
                if needs_ml_confirmation
                else "active ML opposes regime signal"
            )

        if (
            self.require_confluence
            and self.ensemble.is_ready()
            and confluence_failure_reason
        ):
            return FinalSignal(
                direction=0,
                confidence=0.0,
                ta_source=ta_signal.source,
                ml_strength=ml_signal.strength,
                ml_confidence=ml_signal.confidence,
                decision_reason=confluence_failure_reason,
                strategy=ta_signal.strategy,
            )

        active_signals = []
        ta_weight, ml_weight = self._strategy_weights(ta_signal.strategy)
        if ta_active:
            active_signals.append((ta_weight, ta_signal.direction, ta_signal.strength))
        if ml_active:
            active_signals.append((ml_weight, ml_signal.direction, ml_signal.strength))

        if not active_signals:
            combined = 0.0
            confidence = 0.0
            direction = 0
            decision_reason = "no active signal"
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
                decision_reason = "combined signal below decision threshold"
            else:
                direction = 1 if combined > 0 else -1
                decision_reason = "aligned signal ready"

        return FinalSignal(
            direction=direction,
            confidence=min(confidence, 1.0),
            ta_source=ta_signal.source,
            ml_strength=ml_signal.strength,
            ml_confidence=ml_signal.confidence,
            decision_reason=decision_reason,
            strategy=ta_signal.strategy,
        )

    def _strategy_weights(self, strategy: str) -> tuple[float, float]:
        multipliers = {
            "trend": (1.0, 1.0),
            "transition": (1.0, 1.0),
            "range": (1.75, 0.50),
            "breakout": (1.50, 0.67),
            "reversal": (1.25, 0.75),
            "scalp": (2.0, 0.33),
            "countertrend": (1.0, 1.0),
        }
        ta_multiplier, ml_multiplier = multipliers.get(strategy, (1.0, 1.0))
        return self.ta_weight * ta_multiplier, self.ml_weight * ml_multiplier

    @property
    def ml_healthy(self) -> bool:
        return self._ml_healthy

    def _record_ml_success(self) -> None:
        self._ml_success_count += 1
        self._ml_failure_count = 0
        if not self._ml_healthy:
            self._ml_healthy = True
            if self._on_ml_recovered:
                self._on_ml_recovered()

    def _record_ml_failure(self) -> None:
        self._ml_failure_count += 1
        self._ml_success_count = 0
        if self._ml_failure_count >= 3 and self._ml_healthy:
            self._ml_healthy = False
            if self._on_ml_degraded:
                self._on_ml_degraded()
