import json

import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.models.drift import FeatureDriftMonitor
from src.models.registry import ModelArtifactRegistry
from src.models.trainer import ModelTrainer


def test_feature_drift_detects_shifted_distribution():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 1.0, size=(500, 2))
    current = reference.copy()
    current[:, 1] += 3.0
    monitor = FeatureDriftMonitor(min_samples=100)

    report = monitor.compare(
        reference,
        current,
        feature_names=["stable", "shifted"],
    )

    assert "stable" not in report.drifted_features
    assert "shifted" in report.drifted_features
    assert report.severe is True


def test_feature_drift_requires_sufficient_samples():
    monitor = FeatureDriftMonitor(min_samples=100)

    with pytest.raises(ValueError, match="insufficient samples"):
        monitor.compare(np.zeros((10, 2)), np.zeros((10, 2)))


def _write_artifact(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    meta = path.with_name(f"{path.stem}.meta.json")
    meta.write_text(json.dumps({"artifact": content}), encoding="utf-8")


def test_model_registry_promotes_and_rolls_back(tmp_path):
    registry = ModelArtifactRegistry(str(tmp_path))
    active = registry.active_path("BTCUSDT", "1h")
    candidate = registry.candidate_path("BTCUSDT", "1h")
    _write_artifact(active, "old")
    _write_artifact(candidate, "new")

    promotion = registry.promote(
        "BTCUSDT",
        "1h",
        validation={"approved": True, "accuracy": 0.60},
    )

    assert active.read_text(encoding="utf-8") == "new"
    assert promotion.previous_available is True
    registry.rollback("BTCUSDT", "1h")
    assert active.read_text(encoding="utf-8") == "old"


def test_model_registry_rejects_unapproved_candidate(tmp_path):
    registry = ModelArtifactRegistry(str(tmp_path))
    _write_artifact(registry.candidate_path("BTCUSDT", "1h"), "new")

    with pytest.raises(ValueError, match="not been approved"):
        registry.promote(
            "BTCUSDT",
            "1h",
            validation={"approved": False},
        )


def test_trainer_writes_validated_candidate_without_replacing_active(
    tmp_path,
    monkeypatch,
):
    class FakeModel:
        def __init__(self):
            self.metadata = {}
            self.expected = np.array([])

        def train(self, X, y, eval_set=None):
            self.expected = np.asarray(eval_set[1])

        def predict(self, X):
            return self.expected

        def save(self, path):
            artifact = tmp_path / str(path).split("\\")[-1]
            artifact.write_text("candidate", encoding="utf-8")
            artifact.with_name(f"{artifact.stem}.meta.json").write_text(
                "{}",
                encoding="utf-8",
            )

    monkeypatch.setattr("src.models.trainer.XGBModel", FakeModel)
    monkeypatch.setattr("src.models.trainer.XGB_AVAILABLE", True)
    monkeypatch.setattr(settings, "ml_candidate_min_accuracy", 0.9)
    monkeypatch.setattr(settings, "ml_candidate_min_macro_f1", 0.9)
    trainer = ModelTrainer(model_dir=str(tmp_path))
    X = np.arange(300, dtype=float).reshape(100, 3)
    y = np.array([-1, 0, 1, 0] * 25)
    monkeypatch.setattr(trainer, "prepare_data", lambda df: (X, y))

    _, validation = trainer.train_candidate(
        pd.DataFrame(),
        symbol="BTCUSDT",
        timeframe="1h",
    )

    candidate = ModelArtifactRegistry(str(tmp_path)).candidate_path("BTCUSDT", "1h")
    assert validation["approved"] is True
    assert candidate.exists()
    assert (
        not ModelArtifactRegistry(str(tmp_path)).active_path("BTCUSDT", "1h").exists()
    )
