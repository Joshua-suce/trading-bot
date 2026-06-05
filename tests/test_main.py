from src.main import parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.mode == "backtest"
    assert args.symbol == "BTCUSDT"
    assert args.timeframe == "1h"
    assert args.limit == 500
    assert args.initial_capital == 10_000


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
            "--initial-capital",
            "15000",
        ]
    )
    assert args.mode == "train"
    assert args.symbol == "ETHUSDT"
    assert args.timeframe == "4h"
    assert args.limit == 200
    assert args.initial_capital == 15000.0


def test_parse_args_offline_flag():
    args = parse_args(["--mode", "backtest", "--offline"])
    assert args.mode == "backtest"
    assert args.offline is True


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
