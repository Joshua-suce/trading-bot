import json
import sqlite3
from contextlib import closing

from scripts.secret_scan import scan
from src.audit import AuditStore
from src.security import redact_mapping, redact_text


def test_redact_text_hides_tokens_and_urls():
    token = "123456:" + ("ABCdef" * 5)
    api_key = "abc" + "123456789" + "xyz"
    text = f"telegram https://api.telegram.org/bot{token}/send " f"api_key='{api_key}'"

    redacted = redact_text(text)

    assert token not in redacted
    assert api_key not in redacted
    assert "***" in redacted


def test_redact_mapping_hides_sensitive_keys():
    token = "123456:" + ("ABCdef" * 5)
    payload = {
        "api_secret": "super-secret-value",
        "nested": {"telegram_token": token},
        "safe": "BTCUSDT",
    }

    redacted = redact_mapping(payload)

    assert redacted["api_secret"] == "***"
    assert redacted["nested"]["telegram_token"] == "***"
    assert redacted["safe"] == "BTCUSDT"


def test_audit_store_redacts_event_payloads(tmp_path):
    api_key = "abc" + "123456789" + "xyz"
    audit = AuditStore(str(tmp_path / "audit.db"))
    audit.record_event(
        "test",
        f"token='{api_key}'",
        payload={"api_key": api_key},
    )

    with closing(sqlite3.connect(tmp_path / "audit.db")) as conn:
        row = conn.execute(
            "SELECT message, payload_json FROM audit_events LIMIT 1"
        ).fetchone()

    assert api_key not in row[0]
    payload = json.loads(row[1])
    assert payload["api_key"] == "***"


def test_secret_scan_detects_hardcoded_secret(tmp_path):
    sample = tmp_path / "bad.py"
    secret_value = "abc" + "123456789" + "xyz"
    key_name = "API" + "_KEY"
    sample.write_text(f"{key_name} = '{secret_value}'\n", encoding="utf-8")

    findings = scan(tmp_path)

    assert findings
    assert "hardcoded-secret" in findings[0]
