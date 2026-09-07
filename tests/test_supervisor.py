import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings
from src.monitoring.heartbeat import HeartbeatSnapshot, RuntimeHeartbeat
from src.supervisor import TradingSupervisor


def test_runtime_heartbeat_round_trips_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_BOT_INSTANCE_ID", "instance-123")
    heartbeat = RuntimeHeartbeat(str(tmp_path / "heartbeat.json"))

    written = heartbeat.write("running", open_positions=2)
    loaded = heartbeat.read()

    assert loaded == written
    assert loaded.details["open_positions"] == 2
    assert loaded.details["instance_id"] == "instance-123"


def test_runtime_heartbeat_overwrites_while_readable(tmp_path):
    """Direct write succeeds while the file is open for reading (Windows compat)."""
    heartbeat = RuntimeHeartbeat(str(tmp_path / "heartbeat.json"))
    heartbeat.write("first")
    # Simulate supervisor holding a read handle while we write
    heartbeat.path.read_text(encoding="utf-8")
    heartbeat.write("second")
    assert heartbeat.path.exists()
    assert heartbeat.read().state == "second"


def test_runtime_heartbeat_tolerates_transient_read_lock(tmp_path):
    heartbeat = RuntimeHeartbeat(str(tmp_path / "heartbeat.json"))
    heartbeat.write("running")

    with patch.object(type(heartbeat.path), "read_text", side_effect=PermissionError):
        assert heartbeat.read() is None


def test_supervisor_terminates_stale_matching_child(tmp_path, monkeypatch):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        supervisor_check_interval_seconds=0.5,
        supervisor_startup_grace_seconds=10.0,
        supervisor_heartbeat_stale_seconds=10.0,
        supervisor_shutdown_timeout_seconds=1.0,
    )
    supervisor = TradingSupervisor(cfg=cfg)
    stale = HeartbeatSnapshot(
        pid=42,
        state="running",
        updated_at=(datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
        details={"instance_id": "instance-123"},
    )
    monkeypatch.setattr(supervisor.heartbeat, "read", lambda: stale)
    monkeypatch.setattr("src.supervisor.time.sleep", lambda seconds: None)
    monkeypatch.setattr("src.supervisor.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("src.supervisor.sys.platform", "linux")

    class Process:
        pid = 42
        return_code = None
        terminated = False

        def poll(self):
            return self.return_code

        def terminate(self):
            self.terminated = True
            self.return_code = -15

        def wait(self, timeout=None):
            return self.return_code

    process = Process()

    assert supervisor._monitor(process, started=0.0, instance_id="instance-123") is True
    assert process.terminated is True


def test_supervisor_terminates_child_without_matching_heartbeat(tmp_path, monkeypatch):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        supervisor_check_interval_seconds=0.5,
        supervisor_startup_grace_seconds=10.0,
        supervisor_shutdown_timeout_seconds=1.0,
    )
    supervisor = TradingSupervisor(cfg=cfg)
    monkeypatch.setattr(supervisor.heartbeat, "read", lambda: None)
    monkeypatch.setattr("src.supervisor.time.sleep", lambda seconds: None)
    monkeypatch.setattr("src.supervisor.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("src.supervisor.sys.platform", "linux")

    class Process:
        pid = 42
        return_code = None
        terminated = False

        def poll(self):
            return self.return_code

        def terminate(self):
            self.terminated = True
            self.return_code = -15

        def wait(self, timeout=None):
            return self.return_code

    process = Process()

    assert supervisor._monitor(process, started=0.0, instance_id="instance-123") is True
    assert process.terminated is True


def test_supervisor_accepts_heartbeat_from_interpreter_child(tmp_path, monkeypatch):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        supervisor_check_interval_seconds=0.5,
        supervisor_startup_grace_seconds=10.0,
        supervisor_heartbeat_stale_seconds=10.0,
    )
    supervisor = TradingSupervisor(cfg=cfg)
    snapshot = HeartbeatSnapshot(
        pid=99,
        state="running",
        updated_at=datetime.now(timezone.utc).isoformat(),
        details={"instance_id": "instance-123"},
    )
    monkeypatch.setattr(supervisor.heartbeat, "read", lambda: snapshot)
    monkeypatch.setattr("src.supervisor.time.sleep", lambda seconds: None)
    monkeypatch.setattr("src.supervisor.time.monotonic", lambda: 100.0)

    class Process:
        pid = 42
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    process = Process()

    assert (
        supervisor._monitor(process, started=0.0, instance_id="instance-123") is False
    )


def test_supervisor_allows_longer_staleness_while_child_bootstraps(
    tmp_path, monkeypatch
):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        supervisor_check_interval_seconds=0.5,
        supervisor_startup_grace_seconds=180.0,
        supervisor_heartbeat_stale_seconds=90.0,
    )
    supervisor = TradingSupervisor(cfg=cfg)
    snapshot = HeartbeatSnapshot(
        pid=99,
        state="bootstrapping",
        updated_at=(datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat(),
        details={"instance_id": "instance-123"},
    )
    monkeypatch.setattr(supervisor.heartbeat, "read", lambda: snapshot)
    monkeypatch.setattr("src.supervisor.time.sleep", lambda seconds: None)
    monkeypatch.setattr("src.supervisor.time.monotonic", lambda: 200.0)

    class Process:
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    assert (
        supervisor._monitor(Process(), started=0.0, instance_id="instance-123") is False
    )


def test_alert_exhausted_sends_critical_alert_when_channel_configured(
    tmp_path, monkeypatch
):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        telegram_bot_token="token-123",
        telegram_chat_id="chat-456",
    )
    supervisor = TradingSupervisor(cfg=cfg)
    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "src.supervisor.Alerter.send",
        send_mock,
    )

    supervisor._alert_exhausted(5)

    send_mock.assert_awaited_once()
    (message,), kwargs = send_mock.await_args
    assert "STOPPED" in message
    assert "5 time(s)" in message
    assert kwargs.get("level") == "critical"


def test_alert_exhausted_skips_silently_when_no_channel_configured(
    tmp_path, monkeypatch
):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        telegram_bot_token="",
        telegram_chat_id="",
        discord_webhook_url="",
    )
    supervisor = TradingSupervisor(cfg=cfg)
    send_mock = AsyncMock()
    monkeypatch.setattr("src.supervisor.Alerter.send", send_mock)

    supervisor._alert_exhausted(5)  # must not raise

    send_mock.assert_not_awaited()


def test_alert_exhausted_swallows_delivery_errors(tmp_path, monkeypatch):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        telegram_bot_token="token-123",
        telegram_chat_id="chat-456",
    )
    supervisor = TradingSupervisor(cfg=cfg)
    monkeypatch.setattr(
        "src.supervisor.Alerter.send",
        AsyncMock(side_effect=RuntimeError("network down")),
    )

    supervisor._alert_exhausted(5)  # must not raise, only log


def test_run_alerts_and_exits_when_restart_budget_exhausted(tmp_path, monkeypatch):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        supervisor_max_restarts_per_hour=1,
        supervisor_restart_backoff_seconds=0.0,
    )
    supervisor = TradingSupervisor(cfg=cfg)
    monkeypatch.setattr("src.supervisor.time.sleep", lambda seconds: None)
    monkeypatch.setattr(supervisor, "_monitor", lambda *a, **k: True)

    class Process:
        pid = 42

        def poll(self):
            return None

    monkeypatch.setattr(
        "src.supervisor.subprocess.Popen", lambda *a, **k: Process()
    )
    alert_mock = MagicMock()
    monkeypatch.setattr(supervisor, "_alert_exhausted", alert_mock)

    assert supervisor.run() == 1
    alert_mock.assert_called_once()


def test_supervisor_forces_child_tree_after_interrupt_timeout(tmp_path, monkeypatch):
    cfg = Settings(
        _env_file=None,
        runtime_heartbeat_path=str(tmp_path / "heartbeat.json"),
        supervisor_shutdown_timeout_seconds=1.0,
    )
    supervisor = TradingSupervisor(cfg=cfg)
    terminate = MagicMock()
    monkeypatch.setattr(supervisor, "_terminate", terminate)

    class TimedOutProcess:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("trade", timeout)

    process = TimedOutProcess()
    supervisor._wait_for_interrupt_shutdown(process)

    terminate.assert_called_once_with(process)
