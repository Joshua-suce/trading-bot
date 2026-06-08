import pytest

from src.main import parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.mode == "trade"
    assert args.symbol == "BTCUSDT"
    assert args.timeframe == "1h"
    assert args.limit == 500


def test_parse_args_custom_options():
    args = parse_args(
        [
            "--mode",
            "train",
            "--symbol",
            "ETHUSDT",
            "--timeframe",
            "4h",
            "--limit",
            "200",
        ]
    )
    assert args.mode == "train"
    assert args.symbol == "ETHUSDT"
    assert args.timeframe == "4h"
    assert args.limit == 200


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
