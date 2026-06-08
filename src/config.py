import re
from pathlib import Path
from typing import Any, ClassVar, List
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    allowed_timeframes: ClassVar[set[str]] = {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    }

    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_api_key_file: str = ""
    binance_api_secret_file: str = ""
    binance_api_url: str = "https://demo-fapi.binance.com"
    allow_mainnet_trading: bool = False
    exchange_request_timeout_ms: int = Field(default=30_000, ge=5_000, le=120_000)
    exchange_connect_attempts: int = Field(default=3, ge=1, le=10)
    exchange_connect_backoff_seconds: float = Field(default=2.0, ge=0.1, le=30.0)
    exchange_read_attempts: int = Field(default=3, ge=1, le=10)
    exchange_read_backoff_seconds: float = Field(default=1.0, ge=0.1, le=30.0)

    symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT"
    timeframes: str = "5m,15m,30m,1h,4h,1d"
    position_scope: str = "symbol"
    scan_sleep_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    account_refresh_interval_seconds: float = Field(default=60.0, ge=10.0, le=600.0)
    reconciliation_interval_seconds: float = Field(default=30.0, ge=10.0, le=3600.0)
    min_ohlcv_candles: int = Field(default=200, ge=200, le=1000)
    max_candle_delay_multiplier: float = Field(default=3.0, ge=1.0, le=20.0)
    allow_zero_volume_candles: bool = False
    max_entry_slippage_bps: float = Field(default=25.0, ge=0.0, le=1000.0)
    entry_fill_resolution_attempts: int = Field(default=5, ge=1, le=10)
    entry_fill_resolution_backoff_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
    )
    reentry_cooldown_seconds: int = Field(default=900, ge=0, le=86_400)
    min_order_notional: float = Field(default=5.0, ge=0.0, le=100000.0)
    max_leverage: int = Field(default=3, ge=1, le=20)
    max_position_size: float = Field(default=0.02, gt=0, le=0.10)
    max_open_positions: int = Field(default=6, ge=1, le=100)
    max_positions_per_symbol: int = Field(default=3, ge=1, le=20)
    max_total_open_notional_pct: float = Field(default=0.20, gt=0, le=1.0)
    max_symbol_open_notional_pct: float = Field(default=0.10, gt=0, le=1.0)
    daily_loss_limit: float = Field(default=0.05, gt=0, le=0.20)
    max_drawdown: float = Field(default=0.15, gt=0, le=0.50)
    risk_per_trade: float = Field(default=0.02, gt=0, le=0.05)
    max_consecutive_losses: int = Field(default=3, ge=1, le=20)
    consecutive_loss_cooldown_seconds: int = Field(default=1_800, ge=60, le=604_800)
    risk_block_alert_cooldown_seconds: int = Field(default=900, ge=60, le=86_400)

    model_update_interval_hours: int = Field(default=24, ge=1, le=168)
    feature_lookback: int = Field(default=100, ge=20, le=5000)
    prediction_horizon: int = Field(default=6, ge=1, le=500)

    log_level: str = "INFO"
    log_dir: str = "data/logs"
    json_logs: bool = True
    audit_db_path: str = "data/audit/trading_audit.db"
    trading_enabled: bool = True
    require_strategy_approval: bool = True
    strategy_approval_path: str = "data/governance/strategy_approval.json"
    strategy_approval_max_age_hours: int = Field(default=168, ge=1, le=8760)
    min_approval_trades: int = Field(default=20, ge=1, le=10000)
    min_approval_profit_factor: float = Field(default=1.1, ge=0.0, le=10.0)
    max_approval_drawdown_pct: float = Field(default=20.0, ge=0.0, le=100.0)

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_commands_enabled: bool = False
    telegram_allowed_user_ids: str = ""
    telegram_poll_timeout_seconds: int = Field(default=20, ge=1, le=50)
    telegram_alert_queue_size: int = Field(default=100, ge=10, le=10000)
    telegram_delivery_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    discord_webhook_url: str = ""
    telegram_bot_token_file: str = ""
    telegram_chat_id_file: str = ""
    discord_webhook_url_file: str = ""

    model_dir: str = "data/models"

    @field_validator("binance_api_key", "binance_api_secret", mode="before")
    @classmethod
    def strip_secret(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("binance_api_url")
    @classmethod
    def validate_binance_api_url(cls, value: str) -> str:
        normalized = (value or "").strip().rstrip("/")
        parsed = urlparse(normalized)
        allowed_hosts = {"demo-fapi.binance.com", "fapi.binance.com"}
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise ValueError(
                "BINANCE_API_URL must be https://demo-fapi.binance.com "
                "or https://fapi.binance.com"
            )
        if parsed.path or parsed.params or parsed.query or parsed.fragment:
            raise ValueError("BINANCE_API_URL must contain only the Binance host")
        return normalized

    @model_validator(mode="after")
    def load_file_secrets(self):
        self.binance_api_key = self._secret_from_file(
            self.binance_api_key, self.binance_api_key_file, "BINANCE_API_KEY_FILE"
        )
        self.binance_api_secret = self._secret_from_file(
            self.binance_api_secret,
            self.binance_api_secret_file,
            "BINANCE_API_SECRET_FILE",
        )
        self.telegram_bot_token = self._secret_from_file(
            self.telegram_bot_token,
            self.telegram_bot_token_file,
            "TELEGRAM_BOT_TOKEN_FILE",
        )
        self.telegram_chat_id = self._secret_from_file(
            self.telegram_chat_id,
            self.telegram_chat_id_file,
            "TELEGRAM_CHAT_ID_FILE",
        )
        self.discord_webhook_url = self._secret_from_file(
            self.discord_webhook_url,
            self.discord_webhook_url_file,
            "DISCORD_WEBHOOK_URL_FILE",
        )
        if self.telegram_commands_enabled and not (
            self.telegram_bot_token and self.telegram_chat_id
        ):
            raise ValueError(
                "TELEGRAM_COMMANDS_ENABLED requires TELEGRAM_BOT_TOKEN "
                "and TELEGRAM_CHAT_ID"
            )
        if (
            self.telegram_commands_enabled
            and self.telegram_chat_id.startswith("-")
            and not self.telegram_allowed_user_ids
        ):
            raise ValueError(
                "Group Telegram commands require TELEGRAM_ALLOWED_USER_IDS"
            )
        return self

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: str) -> str:
        symbols = cls._split_symbols(value)
        if not symbols:
            raise ValueError("SYMBOLS must include at least one market")
        invalid = [
            symbol for symbol in symbols if not re.fullmatch(r"[A-Z0-9]{6,20}", symbol)
        ]
        if invalid:
            raise ValueError(f"Invalid Binance symbols: {', '.join(invalid)}")
        return ",".join(symbols)

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, value: str) -> str:
        timeframes = cls._split_csv(value)
        if not timeframes:
            raise ValueError("TIMEFRAMES must include at least one timeframe")
        invalid = [tf for tf in timeframes if tf not in cls.allowed_timeframes]
        if invalid:
            raise ValueError(f"Invalid timeframes: {', '.join(invalid)}")
        return ",".join(timeframes)

    @field_validator("position_scope")
    @classmethod
    def validate_position_scope(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized != "symbol":
            raise ValueError(
                "POSITION_SCOPE must be 'symbol'; Binance USD-M positions are "
                "aggregated per symbol even when internal trade legs are tracked "
                "per timeframe"
            )
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = (value or "").strip().upper()
        if normalized not in {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be a valid loguru level")
        return normalized

    @field_validator("telegram_allowed_user_ids")
    @classmethod
    def validate_telegram_user_ids(cls, value: str) -> str:
        user_ids = cls._split_csv(value)
        invalid = [user_id for user_id in user_ids if not user_id.isdigit()]
        if invalid:
            raise ValueError("TELEGRAM_ALLOWED_USER_IDS must contain numeric IDs")
        return ",".join(user_ids)

    @staticmethod
    def _split_csv(value: str) -> List[str]:
        return [item.strip() for item in (value or "").split(",") if item.strip()]

    @staticmethod
    def _split_symbols(value: str) -> List[str]:
        return [
            item.strip().upper() for item in (value or "").split(",") if item.strip()
        ]

    @staticmethod
    def _secret_from_file(current: str, file_path: str, env_name: str) -> str:
        if current:
            return current.strip()
        if not file_path:
            return ""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"{env_name} points to a missing file: {file_path}")
        return path.read_text(encoding="utf-8").strip()

    @property
    def symbols_list(self) -> List[str]:
        return self._split_symbols(self.symbols)

    @property
    def timeframes_list(self) -> List[str]:
        return [item.strip() for item in self.timeframes.split(",") if item.strip()]

    @property
    def telegram_allowed_user_ids_list(self) -> List[str]:
        return self._split_csv(self.telegram_allowed_user_ids)

    @property
    def has_exchange_credentials(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    @property
    def binance_environment(self) -> str:
        host = urlparse(self.binance_api_url).hostname
        return "demo" if host == "demo-fapi.binance.com" else "mainnet"

    @property
    def binance_demo(self) -> bool:
        return self.binance_environment == "demo"

    @property
    def exchange_id(self) -> str:
        return "binanceusdm"

    @property
    def exchange_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "apiKey": self.binance_api_key or None,
            "secret": self.binance_api_secret or None,
            "enableRateLimit": True,
            "timeout": self.exchange_request_timeout_ms,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }
        if self.binance_demo:
            config["options"]["fetchCurrencies"] = False
        return config


settings = Settings()
