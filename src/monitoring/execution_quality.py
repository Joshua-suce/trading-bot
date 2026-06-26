from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any


def execution_quality_summary(
    attempts: list[dict[str, Any]],
    *,
    window_hours: int,
    min_attempts: int,
    warning_slippage_bps: float,
    critical_failure_rate: float,
    warning_protection_latency_ms: float,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=window_hours)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        if _timestamp(attempt.get("started_at")) < cutoff:
            continue
        groups[
            (
                str(attempt.get("symbol") or "UNKNOWN").upper(),
                str(attempt.get("phase") or "unknown"),
            )
        ].append(attempt)

    rows = []
    for (symbol, phase), records in sorted(groups.items()):
        completed = [
            record for record in records if record.get("status") == "completed"
        ]
        failures = len(records) - len(completed)
        failure_rate = failures / len(records)
        slippage = _numbers(completed, "slippage_bps")
        order_latency = _numbers(records, "order_latency_ms")
        resolution_latency = _numbers(records, "fill_resolution_latency_ms")
        protection_latency = _numbers(completed, "protection_latency_ms")
        recovered = sum(bool(record.get("recovered_order")) for record in records)
        avg_slippage = _average(slippage)
        avg_protection_latency = _average(protection_latency)
        status, reason = _quality_status(
            attempts=len(records),
            min_attempts=min_attempts,
            failure_rate=failure_rate,
            avg_slippage_bps=avg_slippage,
            warning_slippage_bps=warning_slippage_bps,
            avg_protection_latency_ms=avg_protection_latency,
            warning_protection_latency_ms=warning_protection_latency_ms,
            critical_failure_rate=critical_failure_rate,
        )
        rows.append(
            {
                "symbol": symbol,
                "phase": phase,
                "status": status,
                "reason": reason,
                "attempts": len(records),
                "completed": len(completed),
                "failures": failures,
                "failure_rate_pct": round(failure_rate * 100, 2),
                "avg_slippage_bps": _rounded(avg_slippage),
                "p95_slippage_bps": _rounded(_percentile(slippage, 0.95)),
                "avg_order_latency_ms": _rounded(_average(order_latency)),
                "p95_order_latency_ms": _rounded(_percentile(order_latency, 0.95)),
                "avg_fill_resolution_latency_ms": _rounded(
                    _average(resolution_latency)
                ),
                "avg_protection_latency_ms": _rounded(avg_protection_latency),
                "recovered_orders": recovered,
                "window_hours": window_hours,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            {"critical": 0, "degraded": 1, "healthy": 2, "insufficient": 3}.get(
                str(row["status"]), 4
            ),
            str(row["symbol"]),
            str(row["phase"]),
        ),
    )


def _quality_status(
    *,
    attempts: int,
    min_attempts: int,
    failure_rate: float,
    avg_slippage_bps: float | None,
    warning_slippage_bps: float,
    avg_protection_latency_ms: float | None,
    warning_protection_latency_ms: float,
    critical_failure_rate: float,
) -> tuple[str, str]:
    if attempts < min_attempts:
        return "insufficient", f"needs {min_attempts} attempts"
    if failure_rate >= critical_failure_rate:
        return "critical", "execution failure rate exceeds policy"
    reasons = []
    if avg_slippage_bps is not None and avg_slippage_bps >= warning_slippage_bps:
        reasons.append("average slippage is elevated")
    if (
        avg_protection_latency_ms is not None
        and avg_protection_latency_ms >= warning_protection_latency_ms
    ):
        reasons.append("protection placement is slow")
    if reasons:
        return "degraded", "; ".join(reasons)
    return "healthy", "execution quality within policy"


def _numbers(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return ordered[index]


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _timestamp(value: Any) -> datetime:
    try:
        text = str(value or "")
        if not text:
            return datetime.min.replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
