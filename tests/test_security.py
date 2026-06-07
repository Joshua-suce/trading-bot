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


def test_redact_text_hides_binance_request_signature():
    signature = "a" * 64
    text = (
        "https://demo-fapi.binance.com/fapi/v3/positionRisk?"
        f"timestamp=1&signature={signature}"
    )

    redacted = redact_text(text)

    assert signature not in redacted
    assert "signature=<redacted>" in redacted


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


def test_audit_store_redacts_legacy_signed_urls_on_open(tmp_path):
    from src.audit import AuditStore

    db_path = tmp_path / "legacy-signature.db"
    audit = AuditStore(str(db_path))
    signature = "b" * 64
    with audit._connection() as conn:
        conn.execute(
            """
            INSERT INTO audit_events (
                id, ts_utc, event_type, severity, message, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-signed-event",
                audit._now(),
                "legacy",
                "error",
                f"request failed?signature={signature}",
                f'{{"error": "request failed?signature={signature}"}}',
            ),
        )

    reloaded = AuditStore(str(db_path))
    event = reloaded.load_recent_events(1)[0]

    assert signature not in event["message"]
    assert signature not in event["payload"]["error"]
