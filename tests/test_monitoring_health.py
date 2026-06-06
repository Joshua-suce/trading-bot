import json
import threading
from datetime import datetime, timezone

from src.audit import AuditStore
from src.main import parse_args
from src.monitoring.dashboard import _equity_curve, _events_frame, _trades_frame
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

    assert trades.iloc[0]["symbol"] == "BTCUSDT"
    assert "dashboard_test" in set(events["event_type"])
    assert curve.empty


def test_parse_args_health_mode():
    args = parse_args(["--mode", "health"])
    assert args.mode == "health"
