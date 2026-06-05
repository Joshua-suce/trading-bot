import json

from src.audit import AuditStore
from src.governance import StrategyApprovalStore
from src.main import run_admin, run_health


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
    assert audit.trading_allowed() == (False, "manual trading pause is active")
    assert audit.get_controls()["manual_pause"]["reason"] == "maintenance"

    run_admin("resume", "ready")
    assert audit.trading_allowed() == (True, "ok")

    run_admin("emergency-stop", "operator stop")
    assert audit.trading_allowed() == (False, "emergency stop is active")

    run_admin("clear-emergency", "operator clear")
    assert audit.trading_allowed() == (True, "ok")

    run_admin("revoke-strategy", "approval expired")
    approval = StrategyApprovalStore(str(approval_path)).load()
    assert approval is not None
    assert approval.approved is False
    assert approval.reason == "approval expired"
    assert audit.load_recent_events(1)[0]["event_type"] == "strategy_approval_revoked"
