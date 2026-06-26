from datetime import datetime, timezone

import pytest

from src.audit import AuditStore
from src.monitoring.execution_quality import execution_quality_summary


def _attempts(
    *,
    count: int,
    failures: int = 0,
    slippage_bps: float = 5.0,
    protection_latency_ms: float = 500.0,
) -> list[dict]:
    now = datetime(2026, 6, 15, tzinfo=timezone.utc).isoformat()
    return [
        {
            "started_at": now,
            "symbol": "BTCUSDT",
            "phase": "entry",
            "status": "order_failed" if index < failures else "completed",
            "slippage_bps": None if index < failures else slippage_bps,
            "order_latency_ms": 100.0 + index,
            "fill_resolution_latency_ms": 50.0,
            "protection_latency_ms": (
                None if index < failures else protection_latency_ms
            ),
            "recovered_order": index == count - 1,
        }
        for index in range(count)
    ]


def _summary(attempts):
    return execution_quality_summary(
        attempts,
        window_hours=24,
        min_attempts=10,
        warning_slippage_bps=20.0,
        critical_failure_rate=0.25,
        warning_protection_latency_ms=5000.0,
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )[0]


def test_execution_quality_requires_minimum_sample():
    assert _summary(_attempts(count=5))["status"] == "insufficient"


def test_execution_quality_reports_healthy_metrics():
    row = _summary(_attempts(count=10))

    assert row["status"] == "healthy"
    assert row["failure_rate_pct"] == 0.0
    assert row["avg_slippage_bps"] == 5.0
    assert row["recovered_orders"] == 1


def test_execution_quality_detects_failure_rate():
    row = _summary(_attempts(count=12, failures=4))

    assert row["status"] == "critical"
    assert row["failure_rate_pct"] == pytest.approx(33.33, abs=0.01)


def test_execution_quality_detects_slippage_and_protection_latency():
    row = _summary(
        _attempts(
            count=10,
            slippage_bps=25.0,
            protection_latency_ms=6000.0,
        )
    )

    assert row["status"] == "degraded"
    assert "slippage" in row["reason"]
    assert "protection" in row["reason"]


def test_audit_store_persists_execution_attempt_updates(tmp_path):
    audit = AuditStore(tmp_path / "execution.db")
    audit.record_execution_attempt(
        execution_id="exec-1",
        correlation_id="corr-1",
        phase="entry",
        symbol="BTCUSDT",
        timeframe="5m",
        strategy="trend",
        side="long",
        status="submitted",
        expected_price=100.0,
        quantity=1.0,
        started_at="2026-06-15T10:00:00+00:00",
    )
    audit.record_execution_attempt(
        execution_id="exec-1",
        correlation_id="corr-1",
        phase="entry",
        symbol="BTCUSDT",
        timeframe="5m",
        strategy="trend",
        side="long",
        status="completed",
        expected_price=100.0,
        actual_price=100.1,
        quantity=1.0,
        slippage_bps=10.0,
        order_latency_ms=120.0,
        protection_latency_ms=300.0,
        fill_source="order_payload",
        order_id="order-1",
        recovered_order=True,
        completed=True,
        started_at="2026-06-15T10:00:00+00:00",
    )

    row = audit.load_execution_attempts(1)[0]
    assert row["status"] == "completed"
    assert row["slippage_bps"] == 10.0
    assert row["recovered_order"] == 1
