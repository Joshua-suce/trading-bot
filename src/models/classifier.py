# XGBoost classifier — 3-class model (-1 sell, 0 hold, +1 buy)
import json
from pathlib import Path
from typing import Any, Optional, Tuple

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
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": 42,
        }
        self.model = XGBClassifier(**self.params)
        self.model_path = model_path
        self.is_trained = False
        self.classes_: np.ndarray = np.array([], dtype=int)

    # Train on feature matrix X and labels y, optionally with a validation set
    def train(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple] = None):
        logger.info(f"Training XGBoost classifier: {X.shape}")
        self.classes_ = np.unique(y).astype(int)
        if len(self.classes_) < 2:
            raise ValueError("XGBoost training requires at least two target classes")
        invalid = set(self.classes_) - {-1, 0, 1}
        if invalid:
            raise ValueError(f"Unsupported direction labels: {sorted(invalid)}")

        encoded_y = self._encode_labels(y)
        model_params = dict(self.params)
        if len(self.classes_) == 2:
            model_params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            model_params.update(
                objective="multi:softprob",
                num_class=len(self.classes_),
                eval_metric="mlogloss",
            )
        self.model = XGBClassifier(**model_params)

        eval_sets = [(X, encoded_y)]
        if eval_set:
            eval_X, eval_y = eval_set
            unseen = set(np.unique(eval_y).astype(int)) - set(self.classes_)
            if unseen:
                raise ValueError(
                    f"Evaluation data contains unseen classes: {sorted(unseen)}"
                )
            eval_sets.append((eval_X, self._encode_labels(eval_y)))
        self.model.fit(
            X,
            encoded_y,
            eval_set=eval_sets,
            verbose=False,
        )
        self.is_trained = True
        logger.info("XGBoost training complete")

    # Predict class labels (-1, 0, 1)
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        encoded = self.model.predict(X).astype(int)
        return self.classes_[encoded]

    # Predict class probabilities
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        return self.model.predict_proba(X)

    # Predict with confidence scores: returns (predictions, confidences)
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probs, axis=1)]
        confidences = np.max(probs, axis=1)
        return predictions, confidences

    # Persist the trained model without pickle-based serialization.
    def save(self, path: str):
        if not self.is_trained or len(self.classes_) < 2:
            raise RuntimeError("Cannot save an untrained XGBoost model")
        model_path = Path(path)
        if model_path.suffix.lower() != ".json":
            raise ValueError("XGBoost model path must use the .json extension")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = self._metadata_path(model_path)
        temporary_model = model_path.with_suffix(".tmp.json")
        temporary_metadata = metadata_path.with_suffix(".tmp.json")
        self.model.save_model(temporary_model)
        temporary_metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "classes": self.classes_.tolist(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_model.replace(model_path)
        temporary_metadata.replace(metadata_path)
        logger.info(f"Model saved to {model_path}")

    # Load a previously trained model from disk
    def load(self, path: str):
        model_path = Path(path)
        if model_path.suffix.lower() != ".json":
            raise ValueError(
                "Unsupported model artifact; retrain it with the current bot version"
            )
        metadata_path = self._metadata_path(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported XGBoost model metadata version")
        self.classes_ = np.asarray(metadata.get("classes", []), dtype=int)
        if len(self.classes_) < 2:
            raise ValueError("Model artifact does not contain valid class metadata")
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        lookup = {label: index for index, label in enumerate(self.classes_)}
        return np.asarray([lookup[int(label)] for label in labels], dtype=int)

    @staticmethod
    def _metadata_path(model_path: Path) -> Path:
        return model_path.with_name(f"{model_path.stem}.meta.json")
