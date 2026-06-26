import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.monitoring.governance import build_scope_governance


def _signals(
    *,
    count: int,
    correct: int,
    return_bps: float,
    strategy: str = "trend",
) -> list[dict]:
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)
    return [
        {
            "candle_timestamp": (now - timedelta(minutes=index)).isoformat(),
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "strategy": strategy,
            "decision": "accepted",
            "direction": 1,
            "outcome_status": "resolved",
            "direction_correct": int(index < correct),
            "directional_return_bps": (
                return_bps if index < correct else -abs(return_bps)
            ),
        }
        for index in range(count)
    ]


def _trades(pnls: list[float], strategy: str = "trend") -> list[dict]:
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)
    return [
        {
            "closed_at": (now - timedelta(hours=index)).isoformat(),
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "strategy": strategy,
            "status": "closed",
            "pnl": pnl,
        }
        for index, pnl in enumerate(pnls)
    ]


def _governance(signals, trades=None):
    return build_scope_governance(
        signals,
        trades or [],
        window_days=30,
        min_signals=30,
        min_trades=10,
        promote_min_accuracy=0.55,
        disable_max_accuracy=0.42,
        promote_min_return_bps=2.0,
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )


def test_governance_requires_minimum_signal_sample():
    row = _governance(_signals(count=10, correct=10, return_bps=20.0))[0]

    assert row["recommendation"] == "insufficient"


def test_governance_promotes_statistically_supported_scope():
    row = _governance(
        _signals(count=40, correct=32, return_bps=10.0),
        _trades([5.0, -2.0, 4.0, -1.0, 3.0, 2.0, -1.0, 4.0, -1.0, 2.0]),
    )[0]

    assert row["recommendation"] == "promote_candidate"
    assert row["direction_accuracy_pct"] == 80.0
    assert row["profit_factor"] > 1.0


def test_governance_flags_persistently_weak_scope():
    row = _governance(
        _signals(count=40, correct=8, return_bps=-10.0),
    )[0]

    assert row["recommendation"] == "disable_candidate"
    assert "signal outcomes" in row["reason"]


def test_bad_realized_trades_prevent_signal_promotion():
    row = _governance(
        _signals(count=40, correct=32, return_bps=10.0),
        _trades([-5.0, 1.0, -4.0, -3.0, 1.0, -2.0, -1.0, 1.0, -3.0, -2.0]),
    )[0]

    assert row["recommendation"] == "disable_candidate"
    assert "realized trade results" in row["reason"]


def test_governance_marks_configured_scope_state():
    rows = build_scope_governance(
        _signals(count=40, correct=24, return_bps=4.0),
        [],
        window_days=30,
        min_signals=30,
        min_trades=10,
        promote_min_accuracy=0.55,
        disable_max_accuracy=0.42,
        promote_min_return_bps=2.0,
        disabled_scopes={"BTCUSDT:5m"},
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )

    assert rows[0]["current_state"] == "disabled"


def test_governance_ignores_nan_signal_outcomes():
    signals = _signals(count=35, correct=25, return_bps=5.0)
    signals[0]["direction_correct"] = math.nan
    signals[0]["directional_return_bps"] = math.nan
    signals[1]["direction_correct"] = pd.NA
    signals[1]["directional_return_bps"] = pd.NA

    row = _governance(signals)[0]

    assert row["scored_signals"] == 33
    assert row["resolved_signals"] == 35
