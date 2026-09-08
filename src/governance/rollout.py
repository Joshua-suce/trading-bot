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


def build_demo_canary_report(
    audit_store,
    *,
    policy: RolloutPolicy | None = None,
    trade_limit: int = 10_000,
    event_limit: int = 10_000,
    since: str | None = None,
    window_hours: float | None = None,
) -> dict[str, Any]:
    policy = policy or RolloutPolicy()
    cutoff = _cutoff_time(since=since, window_hours=window_hours)
    trades = [
        trade
        for trade in audit_store.load_trade_history(
            limit=trade_limit,
            status="closed",
        )
        if str(trade.get("mode") or "") == "trade"
        and _after_cutoff(trade.get("closed_at"), cutoff)
    ]
    trades.sort(key=lambda trade: _timestamp_text(trade.get("closed_at")))
    events = [
        event
        for event in audit_store.load_recent_events(event_limit)
        if _after_cutoff(event.get("ts_utc"), cutoff)
    ]
    protection_failures = _failure_count(events, "protection")
    reconciliation_failures = _failure_count(events, "reconciliation")
    open_trades = audit_store.load_open_trades("trade")

    summary = _trade_summary(trades)
    started_at = _first_trade_time(trades)
    ended_at = _last_trade_time(trades)
    hours = _elapsed_hours(started_at, ended_at)
    scopes = [
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            **_trade_summary(scope_trades),
        }
        for (symbol, timeframe, strategy), scope_trades in sorted(
            _group_trades(trades).items()
        )
    ]
    report = {
        "status": _demo_status(
            policy,
            hours=hours,
            trades=summary["trades"],
            profit_factor=summary["profit_factor"],
            max_drawdown_pct=summary["max_drawdown_pct"],
            protection_failures=protection_failures,
            reconciliation_failures=reconciliation_failures,
        ),
        "reason": _demo_reason(
            policy,
            hours=hours,
            trades=summary["trades"],
            profit_factor=summary["profit_factor"],
            max_drawdown_pct=summary["max_drawdown_pct"],
            protection_failures=protection_failures,
            reconciliation_failures=reconciliation_failures,
        ),
        "started_at": started_at,
        "ended_at": ended_at,
        "cutoff_at": cutoff.isoformat() if cutoff else "",
        "hours": round(hours, 2),
        "open_trades": len(open_trades),
        "protection_failures": protection_failures,
        "reconciliation_failures": reconciliation_failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scopes": scopes,
        **summary,
    }
    return report


def _trade_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_number(trade.get("pnl")) for trade in trades]
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    return {
        "trades": len(trades),
        "wins": sum(1 for pnl in pnls if pnl > 0),
        "losses": sum(1 for pnl in pnls if pnl < 0),
        "net_pnl": round(sum(pnls), 8),
        "gross_profit": round(gross_profit, 8),
        "gross_loss": round(gross_loss, 8),
        "profit_factor": (
            round(profit_factor, 4) if profit_factor != float("inf") else "inf"
        ),
        "win_rate": (
            round(sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100, 2)
            if pnls
            else 0.0
        ),
        "max_drawdown_pct": round(_max_drawdown_pct(trades), 4),
    }


def _group_trades(
    trades: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for trade in trades:
        key = (
            str(trade.get("symbol") or "UNKNOWN"),
            str(trade.get("timeframe") or "-"),
            str(trade.get("strategy") or "unknown"),
        )
        grouped.setdefault(key, []).append(trade)
    return grouped


def _max_drawdown_pct(trades: list[dict[str, Any]]) -> float:
    equity = 100.0
    peak = equity
    max_drawdown = 0.0
    for trade in trades:
        pnl_pct = _number(trade.get("pnl_pct"))
        equity *= max(1.0 + pnl_pct, 0.0)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    return max_drawdown


def _failure_count(events: list[dict[str, Any]], family: str) -> int:
    return sum(
        1
        for event in events
        if family in str(event.get("event_type") or "")
        and "fail" in str(event.get("event_type") or "")
    )


def _demo_status(
    policy: RolloutPolicy,
    *,
    hours: float,
    trades: int,
    profit_factor: float | str,
    max_drawdown_pct: float,
    protection_failures: int,
    reconciliation_failures: int,
) -> str:
    if protection_failures or reconciliation_failures:
        return "blocked"
    if (
        hours >= policy.min_demo_hours
        and trades >= policy.min_demo_trades
        and _number(profit_factor) >= policy.min_demo_profit_factor
        and max_drawdown_pct <= policy.max_demo_drawdown_pct
    ):
        return "promotion_candidate"
    return "collecting"


def _demo_reason(
    policy: RolloutPolicy,
    *,
    hours: float,
    trades: int,
    profit_factor: float | str,
    max_drawdown_pct: float,
    protection_failures: int,
    reconciliation_failures: int,
) -> str:
    if protection_failures:
        return f"demo has protection failures: {protection_failures}"
    if reconciliation_failures:
        return f"demo has reconciliation failures: {reconciliation_failures}"
    missing = []
    if hours < policy.min_demo_hours:
        missing.append(f"hours {hours:.2f}/{policy.min_demo_hours:.2f}")
    if trades < policy.min_demo_trades:
        missing.append(f"trades {trades}/{policy.min_demo_trades}")
    if _number(profit_factor) < policy.min_demo_profit_factor:
        missing.append(
            f"profit_factor {_number(profit_factor):.2f}/"
            f"{policy.min_demo_profit_factor:.2f}"
        )
    if max_drawdown_pct > policy.max_demo_drawdown_pct:
        missing.append(
            f"drawdown {max_drawdown_pct:.2f}%/" f"{policy.max_demo_drawdown_pct:.2f}%"
        )
    if missing:
        return "collecting demo evidence: " + ", ".join(missing)
    return "demo canary passed configured evidence thresholds"


def _first_trade_time(trades: list[dict[str, Any]]) -> str:
    values = [
        _timestamp_text(trade.get("opened_at") or trade.get("entry_time"))
        for trade in trades
    ]
    values = [value for value in values if value]
    return min(values) if values else ""


def _last_trade_time(trades: list[dict[str, Any]]) -> str:
    values = [_timestamp_text(trade.get("closed_at")) for trade in trades]
    values = [value for value in values if value]
    return max(values) if values else ""


def _elapsed_hours(started_at: str, ended_at: str) -> float:
    start = _parse_time(started_at)
    end = _parse_time(ended_at)
    if start is None or end is None or end < start:
        return 0.0
    return (end - start).total_seconds() / 3600


def _timestamp_text(value: Any) -> str:
    timestamp = _parse_time(value)
    return timestamp.isoformat() if timestamp else ""


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cutoff_time(
    *,
    since: str | None,
    window_hours: float | None,
) -> datetime | None:
    if since:
        return _parse_time(since)
    if window_hours is not None and window_hours > 0:
        return datetime.now(timezone.utc) - _seconds_to_delta(window_hours * 3600)
    return None


def _seconds_to_delta(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=seconds)


def _after_cutoff(value: Any, cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    timestamp = _parse_time(value)
    return timestamp is not None and timestamp >= cutoff


def _number(value: Any) -> float:
    try:
        if value == "inf":
            return float("inf")
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
