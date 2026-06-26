import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RolloutEvidence:
    tests_passed: bool
    replay_scopes: list[dict[str, Any]]
    walk_forward: dict[str, Any]
    soak: dict[str, Any]
    demo: dict[str, Any]


@dataclass(frozen=True)
class RolloutDecision:
    stage: str
    passed: bool
    reason: str
    evaluated_at: str


@dataclass(frozen=True)
class RolloutPolicy:
    min_demo_hours: float = 336.0
    min_demo_trades: int = 30
    min_demo_profit_factor: float = 1.10
    max_demo_drawdown_pct: float = 10.0


class RolloutGate:
    def __init__(self, *, policy: RolloutPolicy | None = None) -> None:
        self.policy = policy or RolloutPolicy()

    def evaluate(self, evidence: RolloutEvidence) -> RolloutDecision:
        stage, passed, reason = self._evaluate(evidence)
        return RolloutDecision(
            stage=stage,
            passed=passed,
            reason=reason,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _evaluate(self, evidence: RolloutEvidence) -> tuple[str, bool, str]:
        if not evidence.tests_passed:
            return "tests", False, "automated tests have not passed"
        if not evidence.replay_scopes or any(
            scope.get("evidence_status") != "promotion_candidate"
            for scope in evidence.replay_scopes
        ):
            return "replay", False, "replay scopes lack sufficient after-cost evidence"
        if not evidence.walk_forward.get("approved"):
            return "walk_forward", False, "walk-forward validation has not passed"
        soak = evidence.soak
        if (
            soak.get("status") != "ok"
            or int(soak.get("failed_trades", 0)) > 0
            or int(soak.get("open_trades_after_shutdown", 0)) > 0
        ):
            return "soak", False, "offline soak has unresolved failures"
        demo = evidence.demo
        demo_checks = [
            float(demo.get("hours", 0.0)) >= self.policy.min_demo_hours,
            int(demo.get("trades", 0)) >= self.policy.min_demo_trades,
            float(demo.get("profit_factor", 0.0)) >= self.policy.min_demo_profit_factor,
            float(demo.get("max_drawdown_pct", 100.0))
            <= self.policy.max_demo_drawdown_pct,
            int(demo.get("protection_failures", 0)) == 0,
            int(demo.get("reconciliation_failures", 0)) == 0,
        ]
        if not all(demo_checks):
            return "demo_canary", False, "demo canary evidence is incomplete or weak"
        return (
            "limited_production",
            True,
            "all rollout gates passed; retain mainnet canary risk limits",
        )


class RolloutArtifactStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path).expanduser().resolve()

    def write(
        self,
        evidence: RolloutEvidence,
        decision: RolloutDecision,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "evidence": asdict(evidence),
            "decision": asdict(decision),
        }
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)
