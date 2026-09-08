import json
from typing import Any

from src.governance.rollout import (
    RolloutArtifactStore,
    RolloutEvidence,
    RolloutGate,
    RolloutPolicy,
    build_demo_canary_report,
)


def evidence(**overrides) -> RolloutEvidence:
    values: dict[str, Any] = {
        "tests_passed": True,
        "replay_scopes": [{"evidence_status": "promotion_candidate"}],
        "walk_forward": {"approved": True},
        "soak": {
            "status": "ok",
            "failed_trades": 0,
            "open_trades_after_shutdown": 0,
        },
        "demo": {
            "hours": 336,
            "trades": 30,
            "profit_factor": 1.2,
            "max_drawdown_pct": 5.0,
            "protection_failures": 0,
            "reconciliation_failures": 0,
        },
    }
    values.update(overrides)
    return RolloutEvidence(**values)


def test_rollout_gate_stops_at_first_missing_stage():
    decision = RolloutGate().evaluate(evidence(walk_forward={"approved": False}))

    assert decision.stage == "walk_forward"
    assert decision.passed is False


def test_rollout_gate_requires_long_running_demo_evidence():
    demo = evidence().demo | {"hours": 24}

    decision = RolloutGate().evaluate(evidence(demo=demo))

    assert decision.stage == "demo_canary"
    assert decision.passed is False


def test_rollout_gate_approves_only_limited_production():
    decision = RolloutGate().evaluate(evidence())

    assert decision.stage == "limited_production"
    assert decision.passed is True
    assert "canary risk limits" in decision.reason


def test_rollout_artifact_is_atomic_and_readable(tmp_path):
    rollout_evidence = evidence()
    decision = RolloutGate().evaluate(rollout_evidence)
    path = tmp_path / "rollout.json"

    RolloutArtifactStore(str(path)).write(rollout_evidence, decision)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["decision"]["stage"] == "limited_production"
    assert not (tmp_path / "rollout.json.tmp").exists()


class FakeAudit:
    def __init__(self, trades, events=None, open_trades=None):
        self.trades = trades
        self.events = events or []
        self.open_trades = open_trades or []

    def load_trade_history(self, *, limit, status):
        assert status == "closed"
        return self.trades[:limit]

    def load_recent_events(self, limit):
        return self.events[:limit]

    def load_open_trades(self, mode):
        assert mode == "trade"
        return self.open_trades


def test_demo_canary_report_uses_real_trade_history():
    trades = [
        {
            "mode": "trade",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "strategy": "scalp",
            "opened_at": "2026-01-01T00:00:00+00:00",
            "closed_at": "2026-01-01T01:00:00+00:00",
            "pnl": 2.0,
            "pnl_pct": 0.02,
        },
        {
            "mode": "trade",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "strategy": "scalp",
            "opened_at": "2026-01-01T01:00:00+00:00",
            "closed_at": "2026-01-01T02:00:00+00:00",
            "pnl": -1.0,
            "pnl_pct": -0.01,
        },
    ]

    report = build_demo_canary_report(
        FakeAudit(trades),
        policy=RolloutPolicy(
            min_demo_hours=2,
            min_demo_trades=2,
            min_demo_profit_factor=1.1,
            max_demo_drawdown_pct=5.0,
        ),
    )

    assert report["status"] == "promotion_candidate"
    assert report["trades"] == 2
    assert report["profit_factor"] == 2.0
    assert report["scopes"][0]["strategy"] == "scalp"


def test_demo_canary_report_blocks_on_protection_failure():
    report = build_demo_canary_report(
        FakeAudit(
            [],
            events=[{"event_type": "open_order_protection_failed"}],
        )
    )

    assert report["status"] == "blocked"
    assert report["protection_failures"] == 1


def test_demo_canary_report_can_start_fresh_window():
    trades = [
        {
            "mode": "trade",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "strategy": "scalp",
            "opened_at": "2026-01-01T00:00:00+00:00",
            "closed_at": "2026-01-01T01:00:00+00:00",
            "pnl": -10.0,
            "pnl_pct": -0.10,
        },
        {
            "mode": "trade",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "strategy": "scalp",
            "opened_at": "2026-02-01T00:00:00+00:00",
            "closed_at": "2026-02-01T01:00:00+00:00",
            "pnl": 1.0,
            "pnl_pct": 0.01,
        },
    ]

    report = build_demo_canary_report(
        FakeAudit(trades),
        since="2026-02-01T00:00:00+00:00",
    )

    assert report["trades"] == 1
    assert report["net_pnl"] == 1.0
