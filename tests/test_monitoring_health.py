import json
import threading
from datetime import datetime, timezone

import pytest

from src.audit import AuditStore
from src.main import parse_args
from src.monitoring.dashboard import (
    _apply_control_action,
    _equity_curve,
    _events_frame,
    _execution_quality_frame,
    _executions_frame,
    _signal_metrics,
    _signals_frame,
    _trade_metrics,
    _trades_frame,
)
from src.monitoring.health import HealthChecker
from src.risk.portfolio import TradeRecord


def test_health_check_reports_ok_with_writable_audit_db(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))

    report = HealthChecker(audit).check()

    assert report.audit_db_ok is True
    assert report.status == "ok"
    payload = json.loads(report.to_json())
    assert payload["audit_db_ok"] is True


def test_health_check_reports_degraded_when_trading_paused(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    audit.pause_trading("test pause")

    report = HealthChecker(audit).check()

    assert report.status == "degraded"
    assert report.trading_allowed is False


def test_trading_level_reasons_are_reported_accurately(tmp_path):
    audit = AuditStore(str(tmp_path / "levels.db"))

    audit.set_trading_level(audit.TRADING_LEVEL_YELLOW, "temporary mismatch")
    assert audit.trading_allowed() == (
        False,
        "trading degraded (level 1): temporary mismatch",
    )

    audit.set_trading_level(audit.TRADING_LEVEL_ORANGE, "persistent mismatch")
    assert audit.trading_allowed() == (
        False,
        "trading partially halted (level 2): persistent mismatch",
    )

    audit.set_trading_level(audit.TRADING_LEVEL_RED, "operator stop")
    assert audit.trading_allowed() == (
        False,
        "emergency stop active (level 3): operator stop",
    )


def test_health_counts_critical_events_beyond_recent_event_window(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    audit.record_event("incident", "critical incident", severity="critical")
    for index in range(300):
        audit.record_event("noise", f"debug event {index}", severity="debug")

    report = HealthChecker(audit).check()

    assert report.status == "degraded"
    assert report.critical_events_24h == 1


def test_concurrent_health_checks_share_audit_database(tmp_path):
    db_path = str(tmp_path / "audit.db")
    statuses = []
    errors = []

    def check_health():
        try:
            statuses.append(HealthChecker(AuditStore(db_path)).check().status)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=check_health) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert statuses == ["ok", "ok"]


def test_audit_store_retires_open_trades_from_removed_modes(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    audit = AuditStore(db_path)
    with audit._connection() as conn:
        conn.execute(
            """
            INSERT INTO trades (
                correlation_id, symbol, side, mode, status, entry_price,
                quantity, opened_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-paper",
                "BTCUSDT",
                "long",
                "paper",
                "open",
                100.0,
                1.0,
                audit._now(),
                audit._now(),
            ),
        )

    reloaded = AuditStore(db_path)

    assert reloaded.load_open_trades() == []
    events = reloaded.load_recent_events(1)
    assert events[0]["event_type"] == "legacy_open_trades_retired"


def test_audit_store_updates_open_trade_without_duplicate_open_event(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(timezone.utc),
        timeframe="1m",
        strategy="scalp",
    )
    audit.record_open_trade(
        trade,
        mode="trade",
        correlation_id="corr-scalp",
        stop_loss=99.0,
        take_profit=102.0,
        stop_order_id="sl-1",
        take_profit_order_id="tp-1",
    )

    audit.update_open_trade_state(
        "corr-scalp",
        quantity=0.5,
        stop_loss=100.1,
        take_profit=102.0,
        stop_order_id="sl-2",
        take_profit_order_id="tp-2",
    )

    row = audit.load_open_trades("trade")[0]
    assert row["quantity"] == 0.5
    assert row["stop_loss"] == 100.1
    opened_events = [
        event
        for event in audit.load_recent_events(20)
        if event["event_type"] == "trade_opened"
    ]
    assert len(opened_events) == 1


def test_dashboard_frames_read_audit_store(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(timezone.utc),
    )
    audit.record_open_trade(
        trade,
        mode="trade",
        correlation_id="dash-1",
        stop_loss=95.0,
        take_profit=110.0,
    )
    audit.record_event("dashboard_test", "dashboard event", symbol="BTCUSDT")

    trades = _trades_frame(audit)
    events = _events_frame(audit)
    curve = _equity_curve(trades)
    metrics = _trade_metrics(trades)

    assert trades.iloc[0]["symbol"] == "BTCUSDT"
    assert "dashboard_test" in set(events["event_type"])
    assert curve.empty
    assert metrics["open"] == 1


def test_signal_observations_are_idempotent_and_resolve_forward_outcomes(tmp_path):
    audit = AuditStore(str(tmp_path / "signals.db"))
    timestamp = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)

    observation_id = audit.record_signal_observation(
        candle_timestamp=timestamp,
        symbol="BTCUSDT",
        timeframe="5m",
        strategy="breakout",
        direction=1,
        confidence=0.72,
        minimum_confidence=0.55,
        decision="accepted",
        reason="confirmed",
        signal_price=100.0,
        ta_source="ema_fibonacci",
        ml_strength=0.65,
        ml_confidence=0.80,
        quality_score=0.84,
        quality_reason="confirmed",
        metrics={"adx": 31.0},
    )
    duplicate_id = audit.record_signal_observation(
        candle_timestamp=timestamp,
        symbol="BTCUSDT",
        timeframe="5m",
        strategy="breakout",
        direction=1,
        confidence=0.75,
        minimum_confidence=0.55,
        decision="accepted",
        reason="updated",
        signal_price=100.0,
        ta_source="ema_fibonacci",
        ml_strength=0.68,
        ml_confidence=0.82,
    )
    audit.update_signal_execution(observation_id, "opened")
    resolved = audit.resolve_signal_observations(
        symbol="BTCUSDT",
        timeframe="5m",
        candle_timestamp=datetime(2026, 6, 15, 10, 5, tzinfo=timezone.utc),
        outcome_price=101.0,
    )

    observations = audit.load_signal_observations()
    assert duplicate_id == observation_id
    assert len(observations) == 1
    assert resolved == 1
    assert observations[0]["confidence"] == 0.75
    assert observations[0]["execution_status"] == "opened"
    assert observations[0]["outcome_status"] == "resolved"
    assert observations[0]["directional_return_bps"] == pytest.approx(100.0)
    assert observations[0]["direction_correct"] == 1
    assert observations[0]["outcome_horizon_seconds"] == 300.0


def test_signal_dashboard_reports_strategy_decision_quality(tmp_path):
    audit = AuditStore(str(tmp_path / "signal-dashboard.db"))
    timestamp = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
    audit.record_signal_observation(
        candle_timestamp=timestamp,
        symbol="ETHUSDT",
        timeframe="15m",
        strategy="range",
        direction=-1,
        confidence=0.64,
        minimum_confidence=0.50,
        decision="quality_rejected",
        reason="volume too low",
        signal_price=100.0,
        ta_source="range",
        ml_strength=-0.20,
        ml_confidence=0.60,
    )
    audit.resolve_signal_observations(
        symbol="ETHUSDT",
        timeframe="15m",
        candle_timestamp=datetime(2026, 6, 15, 10, 15, tzinfo=timezone.utc),
        outcome_price=99.0,
    )

    signals = _signals_frame(audit)
    metrics = _signal_metrics(signals)

    assert signals.iloc[0]["decision"] == "quality_rejected"
    assert metrics.iloc[0]["strategy"] == "range"
    assert metrics.iloc[0]["resolved"] == 1
    assert metrics.iloc[0]["direction_accuracy_pct"] == 100.0
    assert metrics.iloc[0]["avg_directional_return_bps"] == pytest.approx(100.0)


def test_execution_dashboard_reads_attempts_and_quality(tmp_path):
    audit = AuditStore(str(tmp_path / "execution-dashboard.db"))
    for index in range(10):
        audit.record_execution_attempt(
            execution_id=f"exec-{index}",
            correlation_id=f"corr-{index}",
            phase="entry",
            symbol="BTCUSDT",
            timeframe="5m",
            strategy="trend",
            side="long",
            status="completed",
            expected_price=100.0,
            actual_price=100.05,
            quantity=1.0,
            slippage_bps=5.0,
            order_latency_ms=100.0,
            fill_resolution_latency_ms=50.0,
            protection_latency_ms=250.0,
            fill_source="order_payload",
            completed=True,
        )

    executions = _executions_frame(audit)
    quality = _execution_quality_frame(executions)

    assert len(executions) == 10
    assert quality.iloc[0]["status"] == "healthy"
    assert quality.iloc[0]["avg_slippage_bps"] == 5.0


@pytest.mark.parametrize(
    ("action", "allowed"),
    [
        ("pause", False),
        ("resume", True),
        ("emergency-stop", False),
        ("clear-emergency", True),
    ],
)
def test_dashboard_control_actions(tmp_path, action, allowed):
    audit = AuditStore(str(tmp_path / f"{action}.db"))

    message = _apply_control_action(audit, action, "operator test")

    assert message
    assert audit.trading_allowed()[0] is allowed
    event = audit.load_recent_events(1)[0]
    assert event["event_type"] == "dashboard_admin_action"


def test_dashboard_rejects_unknown_control_action(tmp_path):
    audit = AuditStore(str(tmp_path / "unknown.db"))

    with pytest.raises(ValueError, match="Unsupported dashboard action"):
        _apply_control_action(audit, "delete-everything", "test")


def test_manual_trade_request_lifecycle(tmp_path):
    audit = AuditStore(str(tmp_path / "manual-requests.db"))
    request_id = audit.create_manual_trade_request(
        action="open",
        symbol="BTCUSDT",
        side="long",
        timeframe="1h",
        reason="operator entry",
        options={"leverage": 2, "require_quality": True},
    )

    claimed = audit.claim_next_manual_trade_request()

    assert claimed is not None
    assert claimed["request_id"] == request_id
    assert claimed["status"] == "processing"
    assert "leverage" not in claimed["options"]
    assert audit.claim_next_manual_trade_request() is None

    audit.finish_manual_trade_request(
        request_id,
        status="completed",
        result={"position_key": "BTCUSDT:1h"},
    )
    saved = audit.load_manual_trade_requests(1)[0]
    assert saved["status"] == "completed"
    assert saved["result"]["position_key"] == "BTCUSDT:1h"


def test_manual_close_request_requires_correlation_id(tmp_path):
    audit = AuditStore(str(tmp_path / "manual-close.db"))

    with pytest.raises(ValueError, match="correlation_id"):
        audit.create_manual_trade_request(
            action="close",
            symbol="BTCUSDT",
            reason="operator exit",
        )


def test_pending_manual_request_can_be_cancelled(tmp_path):
    audit = AuditStore(str(tmp_path / "manual-cancel.db"))
    request_id = audit.create_manual_trade_request(
        action="open",
        symbol="ETHUSDT",
        side="short",
        timeframe="1h",
        reason="operator entry",
        options={"require_quality": True},
    )

    assert audit.cancel_manual_trade_request(request_id, "changed mind") is True
    assert audit.claim_next_manual_trade_request() is None
    assert audit.load_manual_trade_requests(1)[0]["status"] == "cancelled"


def test_duplicate_active_manual_request_is_rejected(tmp_path):
    audit = AuditStore(str(tmp_path / "manual-duplicate.db"))
    kwargs = {
        "action": "open",
        "symbol": "BNBUSDT",
        "side": "long",
        "timeframe": "1h",
        "reason": "operator entry",
    }
    audit.create_manual_trade_request(**kwargs)

    with pytest.raises(ValueError, match="already active"):
        audit.create_manual_trade_request(**kwargs)


def test_concurrent_duplicate_manual_requests_are_atomic(tmp_path):
    db_path = str(tmp_path / "manual-concurrent.db")
    AuditStore(db_path)
    request_ids = []
    errors = []

    def create_request():
        try:
            request_ids.append(
                AuditStore(db_path).create_manual_trade_request(
                    action="close-all",
                    symbol="ALL",
                    reason="operator exit",
                )
            )
        except ValueError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=create_request) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(request_ids) == 1
    assert len(errors) == 1
    assert "already active" in errors[0]


def test_interrupted_processing_request_is_not_retried(tmp_path):
    audit = AuditStore(str(tmp_path / "manual-interrupted.db"))
    request_id = audit.create_manual_trade_request(
        action="close-all",
        symbol="ALL",
        reason="operator exit",
    )
    assert audit.claim_next_manual_trade_request()["request_id"] == request_id

    assert audit.fail_interrupted_manual_trade_requests() == 1

    request = audit.load_manual_trade_requests(1)[0]
    assert request["status"] == "failed"
    assert "inspect exchange" in request["result"]["reason"]
    assert audit.claim_next_manual_trade_request() is None


def test_parse_args_health_mode():
    args = parse_args(["--mode", "health"])
    assert args.mode == "health"
