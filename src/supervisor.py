import asyncio
import os
import subprocess
import sys
import time
import uuid
from collections import deque

from loguru import logger

from src.config import Settings, settings
from src.monitoring.alerter import Alerter
from src.monitoring.heartbeat import RuntimeHeartbeat


async def _send_with_timeout(alerter: Alerter, message: str, timeout: float) -> bool:
    """Send one alert, bounded to `timeout` total regardless of how many
    channels/retries Alerter would otherwise try. Returns False (rather
    than raising) on timeout so the caller's logging path is uniform with
    an ordinary delivery failure."""
    try:
        return await asyncio.wait_for(
            alerter.send(message, level="critical"), timeout=timeout
        )
    except TimeoutError:
        logger.error(
            "Supervisor-exhausted alert timed out after {:.0f}s", timeout
        )
        return False


class TradingSupervisor:
    def __init__(self, *, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self.heartbeat = RuntimeHeartbeat(cfg.runtime_heartbeat_path)

    def run(self) -> int:
        self._warn_if_exhaustion_unreachable()
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
                # This is the single most important event to alert on: the
                # bot is now fully offline (no more reconciliation, exposure
                # checks, or trade management - only whatever protective
                # stop-loss/take-profit orders are already resting on the
                # exchange). Before this, the only trace was the log file
                # and a CRITICAL line nobody was watching; a live run on
                # 2026-09-07 died silently for hours until the operator
                # happened to be looking at the terminal.
                try:
                    self._alert_exhausted(len(restarts))
                except KeyboardInterrupt:
                    # The alert-send call below is bounded but can still
                    # take several seconds; an operator hitting Ctrl+C
                    # during that window should get the same clean exit as
                    # everywhere else in this method, not a raw traceback.
                    logger.info(
                        "Supervisor interrupted while sending the shutdown "
                        "alert; exiting anyway"
                    )
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

    def _warn_if_exhaustion_unreachable(self) -> None:
        # The exhaustion check (below) only fires once `len(restarts) >=
        # supervisor_max_restarts_per_hour` restarts land inside a *sliding*
        # 3600s window - entries older than that are pruned every cycle. If
        # a full restart cycle (startup grace + fully-backed-off retry)
        # times the restart budget reaches or exceeds 3600s, old restarts
        # get evicted just as fast as new ones accumulate and the count can
        # never reach the threshold: a persistently-hanging child then
        # restart-loops forever with no critical alert ever firing - the
        # exact silent-failure mode this alert exists to close. This is
        # reachable by raising supervisor_startup_grace_seconds (e.g. for a
        # slower exchange/VPS) without also lowering the restart budget, so
        # warn loudly rather than fail silently.
        # Model EVERY path that can end a restart cycle, not just the
        # startup-grace one - whichever is slowest sets the cycle time:
        #   * child killed at the plain startup grace
        #   * child killed for a wedged startup phase; its phase timer
        #     cannot start until the grace window ends, so these add
        #   * child killed for a stalled main loop: the in-process watchdog
        #     waits bot_progress_stall_seconds before it stops publishing,
        #     then the file must age past the stale limit, then the
        #     supervisor's poll interval has to come round
        # plus the time to actually kill it and the backoff before relaunch.
        worst_cycle = (
            max(
                self.cfg.supervisor_startup_grace_seconds,
                self.cfg.supervisor_startup_grace_seconds
                + self.cfg.supervisor_bootstrap_phase_stall_seconds,
                self.cfg.bot_progress_stall_seconds
                + self.cfg.supervisor_heartbeat_stale_seconds
                + self.cfg.supervisor_check_interval_seconds,
            )
            + self.cfg.supervisor_shutdown_timeout_seconds
            + self.cfg.supervisor_max_backoff_seconds
        )
        budget = worst_cycle * self.cfg.supervisor_max_restarts_per_hour
        if budget >= 3600:
            logger.warning(
                "Worst-case restart cycle is {:.0f}s and "
                "supervisor_max_restarts_per_hour is {}, so filling the "
                "restart budget would take {:.0f}s - at or beyond the "
                "3600s rolling window used to detect exhaustion. Old "
                "restarts will be pruned as fast as new ones accumulate, "
                "so a persistently-failing trading child can restart-loop "
                "forever without the critical 'bot stopped' alert ever "
                "firing. Lower supervisor_max_restarts_per_hour, or one of "
                "supervisor_startup_grace_seconds / "
                "supervisor_bootstrap_phase_stall_seconds / "
                "bot_progress_stall_seconds / supervisor_max_backoff_seconds.",
                worst_cycle,
                self.cfg.supervisor_max_restarts_per_hour,
                budget,
            )

    def _alert_exhausted(self, restart_count: int) -> None:
        alerter = Alerter(
            telegram_token=self.cfg.telegram_bot_token,
            telegram_chat_id=self.cfg.telegram_chat_id,
            discord_webhook=self.cfg.discord_webhook_url,
            queue_size=self.cfg.telegram_alert_queue_size,
            delivery_timeout=self.cfg.telegram_delivery_timeout_seconds,
        )
        if not alerter.enabled:
            logger.warning(
                "No alert channel configured; supervisor shutdown will not "
                "be reported outside this log"
            )
            return
        message = (
            "<b>Trading Bot STOPPED</b>\n"
            f"Environment: {self.cfg.binance_environment}\n"
            f"Restarted {restart_count} time(s) in the last hour and gave "
            "up. The process has exited - no new trades, no exposure "
            "checks, no position management until it is restarted "
            "manually. Any stop-loss/take-profit orders already resting "
            "on the exchange remain active."
        )
        # This call used to be an unbounded asyncio.run(alerter.send(...)):
        # with no timeout wrapper, a degraded network (exactly the
        # condition that exhausted the restart budget in the first place)
        # could make Telegram/Discord delivery retry for minutes before
        # run() could return - the opposite of "fail fast and loud" this
        # alert exists for. Bounding it means the process still exits
        # promptly even when the alert itself can't get out.
        timeout = self.cfg.supervisor_alert_timeout_seconds
        try:
            delivered = asyncio.run(_send_with_timeout(alerter, message, timeout))
        except Exception as exc:  # noqa: BLE001 - never let alerting block shutdown
            logger.error("Failed to send supervisor-exhausted alert: {}", exc)
            return
        if not delivered:
            # send() can return False without raising (bad token/webhook,
            # exhausted retries, or our own timeout above) - Alerter logs
            # the specifics internally, but this call site should not stay
            # silent about the single highest-priority alert in the system
            # having possibly never reached anyone.
            logger.error(
                "Supervisor-exhausted alert was not confirmed delivered; "
                "check Telegram/Discord configuration and connectivity"
            )

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
