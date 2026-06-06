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
