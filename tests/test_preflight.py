import pytest
from pydantic import ValidationError

from src.config import Settings
from src.preflight import run_preflight


def make_settings(**overrides) -> Settings:
    defaults = {
        "binance_api_key": "key",
        "binance_api_secret": "secret",
        "binance_api_url": "https://demo-fapi.binance.com",
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


def test_default_timeframes_include_short_and_long_intervals():
    cfg = Settings(
        _env_file=None,
        binance_api_key="key",
        binance_api_secret="secret",
    )

    assert cfg.timeframes_list == ["5m", "15m", "30m", "1h", "4h", "1d"]


def test_settings_rejects_invalid_timeframes():
    with pytest.raises(ValidationError):
        make_settings(timeframes="1h,13m")


def test_settings_requires_ema_200_warmup():
    with pytest.raises(ValidationError):
        make_settings(min_ohlcv_candles=199)


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
    assert make_settings(position_scope="symbol").position_scope == "symbol"
    with pytest.raises(ValidationError):
        make_settings(position_scope="symbol_timeframe")
    with pytest.raises(ValidationError):
        make_settings(position_scope="account")


@pytest.mark.parametrize(
    "url",
    [
        "http://demo-fapi.binance.com",
        "https://example.com",
        "https://fapi.binance.com/fapi/v1",
    ],
)
def test_settings_rejects_unsafe_or_unknown_exchange_urls(url):
    with pytest.raises(ValidationError):
        make_settings(binance_api_url=url)


def test_exchange_url_selects_environment():
    assert make_settings().binance_environment == "demo"
    assert (
        make_settings(binance_api_url="https://fapi.binance.com").binance_environment
        == "mainnet"
    )


def test_exchange_config_includes_request_timeout():
    cfg = make_settings(exchange_request_timeout_ms=45_000)

    assert cfg.exchange_config["timeout"] == 45_000


def test_reentry_cooldown_is_bounded():
    assert make_settings(reentry_cooldown_seconds=900).reentry_cooldown_seconds == 900
    with pytest.raises(ValidationError):
        make_settings(reentry_cooldown_seconds=86_401)


def test_consecutive_loss_circuit_breaker_is_configurable():
    cfg = make_settings(
        max_consecutive_losses=4,
        consecutive_loss_cooldown_seconds=3600,
        risk_block_alert_cooldown_seconds=600,
    )

    assert cfg.max_consecutive_losses == 4
    assert cfg.consecutive_loss_cooldown_seconds == 3600
    assert cfg.risk_block_alert_cooldown_seconds == 600


def test_default_consecutive_loss_cooldown_is_four_hours():
    assert make_settings().consecutive_loss_cooldown_seconds == 14_400


def test_preflight_requires_credentials_for_trade_mode():
    cfg = make_settings(binance_api_key="", binance_api_secret="")

    with pytest.raises(RuntimeError, match="requires BINANCE_API_KEY"):
        run_preflight("trade", cfg=cfg)


def test_preflight_blocks_mainnet_without_opt_in():
    cfg = make_settings(
        binance_api_url="https://fapi.binance.com",
        allow_mainnet_trading=False,
    )

    with pytest.raises(RuntimeError, match="Mainnet trading is disabled"):
        run_preflight("trade", cfg=cfg)


def test_preflight_allows_demo_trade_with_credentials():
    result = run_preflight("trade", cfg=make_settings())

    assert result.environment == "demo"
    assert result.credentials_required is True
