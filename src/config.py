import re
from pathlib import Path
from typing import Any, ClassVar, List

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
    binance_testnet: bool = True
    allow_live_trading: bool = False
    allow_mainnet_paper: bool = False

    symbols: str = "BTCUSDT,ETHUSDT"
    timeframes: str = "1h,4h,1d"
    position_scope: str = "symbol"
    scan_sleep_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    reconciliation_interval_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    min_ohlcv_candles: int = Field(default=50, ge=2, le=1000)
    max_candle_delay_multiplier: float = Field(default=3.0, ge=1.0, le=20.0)
    allow_zero_volume_candles: bool = False
    max_entry_slippage_bps: float = Field(default=25.0, ge=0.0, le=1000.0)
    min_order_notional: float = Field(default=5.0, ge=0.0, le=100000.0)
    max_leverage: int = Field(default=3, ge=1, le=20)
    max_position_size: float = Field(default=0.02, gt=0, le=0.10)
    max_open_positions: int = Field(default=3, ge=1, le=100)
    max_total_open_notional_pct: float = Field(default=0.20, gt=0, le=1.0)
    max_symbol_open_notional_pct: float = Field(default=0.10, gt=0, le=1.0)
    daily_loss_limit: float = Field(default=0.05, gt=0, le=0.20)
    max_drawdown: float = Field(default=0.15, gt=0, le=0.50)
    risk_per_trade: float = Field(default=0.02, gt=0, le=0.05)

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
    discord_webhook_url: str = ""
    telegram_bot_token_file: str = ""
    telegram_chat_id_file: str = ""
    discord_webhook_url_file: str = ""

    model_dir: str = "data/models"

    @field_validator("binance_api_key", "binance_api_secret", mode="before")
    @classmethod
    def strip_secret(cls, value: str) -> str:
        return (value or "").strip()

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
        if normalized not in {"symbol", "symbol_timeframe"}:
            raise ValueError("POSITION_SCOPE must be 'symbol' or 'symbol_timeframe'")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = (value or "").strip().upper()
        if normalized not in {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be a valid loguru level")
        return normalized

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
    def has_exchange_credentials(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    @property
    def exchange_id(self) -> str:
        return "binanceusdm"

    @property
    def exchange_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "apiKey": self.binance_api_key or None,
            "secret": self.binance_api_secret or None,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }
        if self.binance_testnet:
            config["options"]["fetchCurrencies"] = False
        return config


settings = Settings()
