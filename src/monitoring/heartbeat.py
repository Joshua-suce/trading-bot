import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HeartbeatSnapshot:
    pid: int
    state: str
    updated_at: str
    details: dict[str, Any]

    @property
    def age_seconds(self) -> float:
        updated = datetime.fromisoformat(self.updated_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - updated).total_seconds(), 0.0)


class RuntimeHeartbeat:
    def __init__(self, path: str) -> None:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            resolved = project_root / resolved
        self.path = resolved

    def write(self, state: str, **details: Any) -> HeartbeatSnapshot:
        instance_id = os.environ.get("TRADING_BOT_INSTANCE_ID")
        if instance_id:
            details.setdefault("instance_id", instance_id)
        snapshot = HeartbeatSnapshot(
            pid=os.getpid(),
            state=state,
            updated_at=datetime.now(timezone.utc).isoformat(),
            details=details,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(snapshot), sort_keys=True)
        self.path.write_text(payload, encoding="utf-8")
        return snapshot

    def read(self) -> HeartbeatSnapshot | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return HeartbeatSnapshot(
                pid=int(payload["pid"]),
                state=str(payload["state"]),
                updated_at=str(payload["updated_at"]),
                details=dict(payload.get("details") or {}),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return None
