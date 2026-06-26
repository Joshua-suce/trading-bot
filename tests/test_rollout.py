import json

from src.governance.rollout import (
    RolloutArtifactStore,
    RolloutEvidence,
    RolloutGate,
)


def evidence(**overrides) -> RolloutEvidence:
    values = {
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
