import pytest

from src.config import settings
from src.main import _apply_cli_symbol_timeframe_override, parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.mode == "trade"
    # None (not a default symbol/timeframe) means "no CLI restriction" -
    # trade mode falls back to every symbol/timeframe from settings.
    assert args.symbol is None
    assert args.timeframe is None
    assert args.limit == 500


def test_parse_args_custom_options():
    args = parse_args(
        [
            "--mode",
            "trade",
            "--symbol",
            "ETHUSDT",
            "--timeframe",
            "4h",
            "--limit",
            "200",
        ]
    )
    assert args.mode == "trade"
    assert args.symbol == "ETHUSDT"
    assert args.timeframe == "4h"
    assert args.limit == 200


def test_parse_args_accepts_supervisor_mode():
    assert parse_args(["--mode", "supervisor"]).mode == "supervisor"


def test_parse_args_accepts_rollout_evidence_path():
    args = parse_args(["--mode", "rollout", "--rollout-evidence", "evidence.json"])
    assert args.mode == "rollout"
    assert args.rollout_evidence == "evidence.json"


def test_parse_args_accepts_demo_report_path():
    args = parse_args(
        [
            "--mode",
            "demo-report",
            "--demo-report",
            "demo-report.json",
            "--demo-report-window-hours",
            "24",
            "--demo-report-since",
            "2026-07-02T00:00:00+00:00",
        ]
    )
    assert args.mode == "demo-report"
    assert args.demo_report == "demo-report.json"
    assert args.demo_report_window_hours == 24
    assert args.demo_report_since == "2026-07-02T00:00:00+00:00"


def test_parse_args_rejects_removed_backtest_mode():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "backtest"])


@pytest.mark.parametrize("removed_mode", ["paper", "live"])
def test_parse_args_rejects_removed_split_trading_modes(removed_mode):
    with pytest.raises(SystemExit):
        parse_args(["--mode", removed_mode])


@pytest.mark.parametrize(
    "removed_flag", ["--offline", "--initial-capital", "--leverage"]
)
def test_parse_args_rejects_removed_backtest_flags(removed_flag):
    args = [removed_flag] if removed_flag == "--offline" else [removed_flag, "1"]
    with pytest.raises(SystemExit):
        parse_args(args)


def test_parse_args_admin_controls():
    args = parse_args(
        [
            "--mode",
            "admin",
            "--admin-action",
            "emergency-stop",
            "--reason",
            "test",
        ]
    )
    assert args.mode == "admin"
    assert args.admin_action == "emergency-stop"
    assert args.reason == "test"


def test_parse_args_admin_alert_test():
    args = parse_args(["--mode", "admin", "--admin-action", "test-alert"])
    assert args.admin_action == "test-alert"


def test_parse_args_history_options():
    args = parse_args(
        [
            "--mode",
            "history",
            "--history-limit",
            "25",
            "--history-status",
            "closed",
            "--history-symbol",
            "ETHUSDT",
            "--history-format",
            "csv",
            "--history-output",
            "history.csv",
        ]
    )
    assert args.mode == "history"
    assert args.history_limit == 25
    assert args.history_status == "closed"
    assert args.history_symbol == "ETHUSDT"
    assert args.history_format == "csv"
    assert args.history_output == "history.csv"


def test_cli_symbol_timeframe_override_restricts_settings(monkeypatch):
    monkeypatch.setattr(settings, "symbols", "BTCUSDT,ETHUSDT,BNBUSDT")
    monkeypatch.setattr(settings, "timeframes", "1m,3m,5m,15m,30m,1h,4h,1d")

    _apply_cli_symbol_timeframe_override("ethusdt", "15m")

    assert settings.symbols_list == ["ETHUSDT"]
    assert settings.timeframes_list == ["15m"]


def test_cli_symbol_timeframe_override_noop_when_not_passed(monkeypatch):
    monkeypatch.setattr(settings, "symbols", "BTCUSDT,ETHUSDT,BNBUSDT")
    monkeypatch.setattr(settings, "timeframes", "1m,3m,5m,15m,30m,1h,4h,1d")

    _apply_cli_symbol_timeframe_override(None, None)

    assert settings.symbols_list == ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    assert settings.timeframes_list == [
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "4h",
        "1d",
    ]


def test_cli_symbol_timeframe_override_rejects_invalid_timeframe(monkeypatch):
    monkeypatch.setattr(settings, "timeframes", "1h")

    with pytest.raises(ValueError):
        _apply_cli_symbol_timeframe_override(None, "99x")
