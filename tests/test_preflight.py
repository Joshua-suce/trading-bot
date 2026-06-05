import pytest
from pydantic import ValidationError

from src.config import Settings
from src.preflight import run_preflight


def make_settings(**overrides) -> Settings:
    defaults = {
        "binance_api_key": "key",
        "binance_api_secret": "secret",
        "binance_testnet": True,
        "symbols": "btcusdt, ethusdt",
        "timeframes": "1h,4h",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def test_settings_normalizes_symbols_and_log_level():
    cfg = make_settings(log_level="warning")

    assert cfg.symbols == "BTCUSDT,ETHUSDT"
    assert cfg.symbols_list == ["BTCUSDT", "ETHUSDT"]
    assert cfg.log_level == "WARNING"


def test_settings_rejects_invalid_timeframes():
    with pytest.raises(ValidationError):
        make_settings(timeframes="1h,13m")


def test_settings_reads_secret_files(tmp_path):
    key_file = tmp_path / "key.txt"
    secret_file = tmp_path / "secret.txt"
    key_file.write_text("file-key\n", encoding="utf-8")
    secret_file.write_text("file-secret\n", encoding="utf-8")

    cfg = make_settings(
        binance_api_key="",
        binance_api_secret="",
        binance_api_key_file=str(key_file),
        binance_api_secret_file=str(secret_file),
    )

    assert cfg.binance_api_key == "file-key"
    assert cfg.binance_api_secret == "file-secret"


def test_settings_validates_position_scope():
    assert make_settings(position_scope="symbol_timeframe").position_scope == (
        "symbol_timeframe"
    )
    with pytest.raises(ValidationError):
        make_settings(position_scope="account")


def test_preflight_requires_credentials_for_paper_mode():
    cfg = make_settings(binance_api_key="", binance_api_secret="")

    with pytest.raises(RuntimeError, match="requires BINANCE_API_KEY"):
        run_preflight("paper", cfg=cfg)


def test_preflight_blocks_mainnet_paper_without_opt_in():
    cfg = make_settings(binance_testnet=False, allow_mainnet_paper=False)

    with pytest.raises(RuntimeError, match="Paper mode is pointed at Binance mainnet"):
        run_preflight("paper", cfg=cfg)


def test_preflight_blocks_mainnet_live_without_opt_in():
    cfg = make_settings(binance_testnet=False, allow_live_trading=False)

    with pytest.raises(RuntimeError, match="Live mainnet trading is disabled"):
        run_preflight("live", cfg=cfg)


def test_preflight_allows_demo_live_with_credentials():
    result = run_preflight("live", cfg=make_settings(require_strategy_approval=False))

    assert result.environment == "demo"
    assert result.credentials_required is True
