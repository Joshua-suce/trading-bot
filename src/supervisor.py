import os
import subprocess
import sys
import time
import uuid
from collections import deque

from loguru import logger

from src.config import Settings, settings
from src.monitoring.heartbeat import RuntimeHeartbeat


class TradingSupervisor:
    def __init__(self, *, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self.heartbeat = RuntimeHeartbeat(cfg.runtime_heartbeat_path)

    def run(self) -> int:
        restarts: deque[float] = deque()
        backoff = self.cfg.supervisor_restart_backoff_seconds
        while True:
            started = time.monotonic()
            instance_id = uuid.uuid4().hex
            child_environment = os.environ.copy()
            child_environment["TRADING_BOT_INSTANCE_ID"] = instance_id
            process = subprocess.Popen(
                [sys.executable, "-m", "src.main", "--mode", "trade"],
                env=child_environment,
            )
            logger.info(
                "Trading child started with launcher_pid={} instance_id={}",
                process.pid,
                instance_id[:8],
            )
            self.heartbeat.write(
                "launching",
                instance_id=instance_id,
                launcher_pid=process.pid,
            )
            try:
                hung = self._monitor(process, started, instance_id)
            except KeyboardInterrupt:
                logger.info("Supervisor interrupted; waiting for trading child")
                self._wait_for_interrupt_shutdown(process)
                return 130
            return_code = process.poll()
            snapshot = self.heartbeat.read()
            fatal_state = bool(
                snapshot
                and snapshot.details.get("instance_id") == instance_id
                and snapshot.details.get("reason") == "fatal error"
            )
            if not hung and return_code == 0 and not fatal_state:
                return 0
            now = time.monotonic()
            window_start = now - 3600
            while restarts and restarts[0] < window_start:
                restarts.popleft()
            if len(restarts) >= self.cfg.supervisor_max_restarts_per_hour:
                logger.critical("Supervisor restart budget exhausted")
                return 1
            restarts.append(now)
            logger.warning("Restarting trading child in {:.1f}s", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, self.cfg.supervisor_max_backoff_seconds)

    def _monitor(
        self,
        process: subprocess.Popen,
        started: float,
        instance_id: str,
    ) -> bool:
        while process.poll() is None:
            time.sleep(self.cfg.supervisor_check_interval_seconds)
            if time.monotonic() - started < self.cfg.supervisor_startup_grace_seconds:
                continue
            snapshot = self.heartbeat.read()
            if snapshot is None or snapshot.details.get("instance_id") != instance_id:
                logger.error(
                    "Trading child did not publish a valid heartbeat; restarting"
                )
                self._terminate(process)
                return True
            stale_limit = self.cfg.supervisor_heartbeat_stale_seconds
            if snapshot.state in {"launching", "bootstrapping"}:
                stale_limit = max(
                    stale_limit,
                    self.cfg.supervisor_startup_grace_seconds,
                )
            if snapshot.age_seconds <= stale_limit:
                continue
            logger.error(
                "Trading child heartbeat stale ({:.1f}s); restarting",
                snapshot.age_seconds,
            )
            self._terminate(process)
            return True
        return False

    def _terminate(self, process: subprocess.Popen) -> None:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
            try:
                process.wait(timeout=self.cfg.supervisor_shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return
        process.terminate()
        try:
            process.wait(timeout=self.cfg.supervisor_shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _wait_for_interrupt_shutdown(self, process: subprocess.Popen) -> None:
        try:
            process.wait(timeout=self.cfg.supervisor_shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._terminate(process)
