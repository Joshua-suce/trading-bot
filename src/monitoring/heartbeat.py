import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


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
        tmp_path = self.path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
        tmp_path.write_text(payload, encoding="utf-8")
        for attempt in range(3):
            try:
                tmp_path.replace(self.path)
                return snapshot
            except OSError:
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                logger.warning(
                    "Heartbeat replace failed after 3 attempts: {} -> {}",
                    tmp_path,
                    self.path,
                )
                raise
        return snapshot  # unreachable

    async def write_async(self, state: str, **details: Any) -> HeartbeatSnapshot:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.write(state, **details))

    def read(self) -> HeartbeatSnapshot | None:
        for attempt in range(3):
            if not self.path.exists():
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                return None
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                return HeartbeatSnapshot(
                    pid=int(payload["pid"]),
                    state=str(payload["state"]),
                    updated_at=str(payload["updated_at"]),
                    details=dict(payload.get("details") or {}),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                return None
            except OSError:
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                return None
        return None
