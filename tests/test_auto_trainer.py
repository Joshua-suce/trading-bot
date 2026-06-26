import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.audit import AuditStore
from src.models import auto_trainer as auto_trainer_module
from src.models.auto_trainer import AutomaticModelTrainer


@pytest.fixture
def configured_auto_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        auto_trainer_module.settings,
        "model_dir",
        str(tmp_path / "models"),
    )
    monkeypatch.setattr(
        auto_trainer_module.settings,
        "symbols",
        "BTCUSDT",
    )
    monkeypatch.setattr(
        auto_trainer_module.settings,
        "timeframes",
        "1h",
    )
    monkeypatch.setattr(
        auto_trainer_module.settings,
        "disabled_strategy_scopes",
        "",
    )
    monkeypatch.setattr(
        auto_trainer_module.settings,
        "auto_retrain_scope_delay_seconds",
        0.0,
    )
    monkeypatch.setattr(
        auto_trainer_module.settings,
        "auto_retrain_candle_limit",
        500,
    )
    return tmp_path


def test_model_is_due_when_missing_legacy_or_stale(
    configured_auto_trainer,
    monkeypatch,
):
    client = SimpleNamespace()
    trainer = AutomaticModelTrainer(
        client,
        AuditStore(configured_auto_trainer / "audit.db"),
        lambda *_: None,
    )
    model_path = trainer._model_path("BTCUSDT", "1h")
    metadata_path = model_path.with_name(f"{model_path.stem}.meta.json")

    assert trainer._model_is_due("BTCUSDT", "1h") is True

    model_path.parent.mkdir(parents=True)
    model_path.write_text("model", encoding="utf-8")
    metadata_path.write_text(
        json.dumps({"metadata": {"label_schema": "legacy"}}),
        encoding="utf-8",
    )
    assert trainer._model_is_due("BTCUSDT", "1h") is True

    metadata_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "label_schema": "cost_adjusted_horizon_v3",
                    "prediction_horizon": str(
                        auto_trainer_module.settings.prediction_horizon
                    ),
                    "label_atr_multiplier": str(
                        auto_trainer_module.settings.ml_label_atr_multiplier
                    ),
                    "label_min_return": str(
                        auto_trainer_module.settings.ml_effective_label_min_return
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    assert trainer._model_is_due("BTCUSDT", "1h") is False

    stale = time.time() - 25 * 3600
    model_path.touch()
    monkeypatch.setattr(
        auto_trainer_module.settings,
        "model_update_interval_hours",
        24,
    )
    import os

    os.utime(model_path, (stale, stale))
    assert trainer._model_is_due("BTCUSDT", "1h") is True


@pytest.mark.asyncio
async def test_training_uses_closed_candles_and_activates_model(
    configured_auto_trainer,
    monkeypatch,
):
    rows = AutomaticModelTrainer._minimum_training_rows() + 10
    candles = pd.DataFrame(
        {
            "open": np.arange(rows, dtype=float),
            "high": np.arange(rows, dtype=float) + 1,
            "low": np.arange(rows, dtype=float) - 1,
            "close": np.arange(rows, dtype=float) + 0.5,
            "volume": np.ones(rows),
        }
    )
    client = SimpleNamespace(fetch_ohlcv=AsyncMock(return_value=candles))
    captured = {}
    model = SimpleNamespace(
        is_trained=True,
        classes_=np.array([-1, 0, 1]),
        metadata={"label_schema": "cost_adjusted_horizon_v3"},
    )

    class FakeTrainer:
        def train_ensemble(self, df, *, symbol, timeframe):
            captured["rows"] = len(df)
            captured["scope"] = (symbol, timeframe)
            return SimpleNamespace(xgb=model)

    monkeypatch.setattr(auto_trainer_module, "ModelTrainer", FakeTrainer)
    activated = []
    trainer = AutomaticModelTrainer(
        client,
        AuditStore(configured_auto_trainer / "audit.db"),
        lambda symbol, timeframe, ready: activated.append((symbol, timeframe, ready)),
    )

    assert await trainer._train_scope("BTCUSDT", "1h") is True
    assert captured == {"rows": rows - 1, "scope": ("BTCUSDT", "1h")}
    assert activated == [("BTCUSDT", "1h", model)]


@pytest.mark.asyncio
async def test_failed_training_keeps_previous_model(
    configured_auto_trainer,
    monkeypatch,
):
    rows = AutomaticModelTrainer._minimum_training_rows() + 1
    client = SimpleNamespace(
        fetch_ohlcv=AsyncMock(return_value=pd.DataFrame(index=range(rows)))
    )

    class FailingTrainer:
        def train_ensemble(self, *_args, **_kwargs):
            raise RuntimeError("training failed")

    monkeypatch.setattr(auto_trainer_module, "ModelTrainer", FailingTrainer)
    activated = []
    audit = AuditStore(configured_auto_trainer / "audit.db")
    trainer = AutomaticModelTrainer(
        client,
        audit,
        lambda *args: activated.append(args),
    )

    assert await trainer._train_scope("BTCUSDT", "1h") is False
    assert activated == []
    events = audit.load_recent_events(5)
    assert any(event["event_type"] == "ml_retraining_failed" for event in events)
