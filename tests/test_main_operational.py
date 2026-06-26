import json

import pytest

from src.audit import AuditStore
from src.governance import StrategyApprovalStore
from src.main import run_admin, run_alert_test, run_health


def test_run_health_uses_configured_audit_store(tmp_path, monkeypatch, capsys):
    audit_path = tmp_path / "health.db"
    monkeypatch.setattr("src.config.settings.audit_db_path", str(audit_path))

    exit_code = run_health()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "ok"
    assert output["audit_db_path"] == str(audit_path)
    assert audit_path.exists()


def test_run_admin_updates_controls_and_strategy_state(tmp_path, monkeypatch):
    audit_path = tmp_path / "admin.db"
    approval_path = tmp_path / "approval.json"
    monkeypatch.setattr("src.config.settings.audit_db_path", str(audit_path))
    monkeypatch.setattr(
        "src.config.settings.strategy_approval_path", str(approval_path)
    )

    run_admin("pause", "maintenance")
    audit = AuditStore(str(audit_path))
    assert audit.trading_allowed() == (False, "manual pause is active")
    assert audit.get_controls()["manual_pause"]["reason"] == "maintenance"

    run_admin("resume", "ready")
    assert audit.trading_allowed() == (True, "ok")

    run_admin("emergency-stop", "operator stop")
    assert audit.trading_allowed() == (
        False,
        "emergency stop active (level 3): operator stop",
    )

    run_admin("clear-emergency", "operator clear")
    assert audit.trading_allowed() == (True, "ok")

    run_admin("revoke-strategy", "approval expired")
    approval = StrategyApprovalStore(str(approval_path)).load()
    assert approval is not None
    assert approval.approved is False
    assert approval.reason == "approval expired"
    assert audit.load_recent_events(1)[0]["event_type"] == "strategy_approval_revoked"


@pytest.mark.asyncio
async def test_run_alert_test_reports_success(monkeypatch):
    class FakeAlerter:
        delivery_successes = 0
        delivery_failures = 0
        enabled = True

        def __init__(self, **kwargs):
            pass

        async def start(self):
            pass

        async def send(self, message, level):
            assert "Alert Test" in message
            assert level == "test"

        async def stop(self):
            self.delivery_successes = 1

    monkeypatch.setattr("src.monitoring.alerter.Alerter", FakeAlerter)

    await run_alert_test()


@pytest.mark.asyncio
async def test_run_alert_test_rejects_missing_channels(monkeypatch):
    class DisabledAlerter:
        enabled = False

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("src.monitoring.alerter.Alerter", DisabledAlerter)

    with pytest.raises(RuntimeError, match="No alert channel configured"):
        await run_alert_test()
