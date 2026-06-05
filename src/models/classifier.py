# XGBoost classifier — 3-class model (-1 sell, 0 hold, +1 buy)
from typing import Optional, Tuple

import joblib
import numpy as np
from loguru import logger
from xgboost import XGBClassifier


class XGBoostClassifier:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        model_path: Optional[str] = None,
    ):
        # Store hyperparameters for reproducibility
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
        }
        self.model = XGBClassifier(**self.params)
        self.model_path = model_path
        self.is_trained = False

    # Train on feature matrix X and labels y, optionally with a validation set
    def train(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        logger.info(f"Training XGBoost classifier: {X.shape}")
        eval_sets = [(X, y)]
        if eval_set:
            eval_sets.append(eval_set)
        self.model.fit(
            X,
            y,
            eval_set=eval_sets,
            verbose=False,
        )
        self.is_trained = True
        logger.info("XGBoost training complete")

    # Predict class labels (-1, 0, 1)
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.model.predict(X)

    # Predict class probabilities
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.model.predict_proba(X)

    # Predict with confidence scores: returns (predictions, confidences)
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1) - 1  # Map 0->-1, 1->0, 2->1
        confidences = np.max(probs, axis=1)
        return predictions, confidences

    # Persist the trained model to disk via joblib
    def save(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    # Load a previously trained model from disk
    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
