import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.audit import AuditStore
from src.config import settings


@dataclass(frozen=True)
class HealthReport:
    status: str
    audit_db_ok: bool
    trading_allowed: bool
    trading_state_reason: str
    open_trades: int
    critical_events_24h: int
    audit_db_path: str
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class HealthChecker:
    def __init__(self, audit_store: AuditStore | None = None) -> None:
        self.audit_store = audit_store or AuditStore()

    def check(self) -> HealthReport:
        audit_ok = self._audit_db_writable()
        trading_allowed, reason = self.audit_store.trading_allowed()
        critical_events = (
            self.audit_store.count_events_since(
                severity="critical",
                since_utc=(
                    datetime.now(timezone.utc) - timedelta(hours=24)
                ).isoformat(),
            )
            if audit_ok
            else 0
        )
        status = self._status(audit_ok, trading_allowed, critical_events)

        return HealthReport(
            status=status,
            audit_db_ok=audit_ok,
            trading_allowed=trading_allowed,
            trading_state_reason=reason,
            open_trades=len(self.audit_store.load_open_trades()) if audit_ok else 0,
            critical_events_24h=critical_events,
            audit_db_path=str(Path(settings.audit_db_path)),
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _audit_db_writable(self) -> bool:
        try:
            self.audit_store.record_event(
                "health_check",
                "Health check audit write",
                severity="debug",
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _status(audit_ok: bool, trading_allowed: bool, critical_events_24h: int) -> str:
        if not audit_ok:
            return "critical"
        if not trading_allowed or critical_events_24h > 0:
            return "degraded"
        return "ok"
