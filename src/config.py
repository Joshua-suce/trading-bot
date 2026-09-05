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
    allowed_strategies: ClassVar[set[str]] = {
        "trend",
        "transition",
        "range",
        "breakout",
        "reversal",
        "countertrend",
        "scalp",
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
    exchange_dns_cache_ttl: int = Field(default=300, ge=30, le=86_400)
    exchange_connection_limit: int = Field(default=20, ge=1, le=100)
    exchange_connection_limit_per_host: int = Field(default=10, ge=1, le=50)
    exchange_keepalive_seconds: int = Field(default=30, ge=10, le=300)
    exchange_ws_reconnect_jitter: float = Field(default=5.0, ge=0.0, le=30.0)

    symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT"
    timeframes: str = "1m,3m,5m,15m,30m,1h,4h,1d"
    enabled_strategies: str = (
        "trend,transition,range,breakout,reversal,countertrend,scalp"
    )
    disabled_strategy_scopes: str = ""
    position_scope: str = "symbol"
    scan_sleep_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    network_outage_failure_threshold: int = Field(default=6, ge=1, le=100)
    network_outage_cooldown_seconds: float = Field(default=60.0, ge=5.0, le=3600.0)
    network_outage_alert_cooldown_seconds: float = Field(
        default=300.0,
        ge=30.0,
        le=86_400.0,
    )
    candle_close_grace_seconds: float = Field(default=2.0, ge=0.0, le=30.0)
    account_balance_cache_seconds: float = Field(default=10.0, ge=0.0, le=60.0)
    account_refresh_interval_seconds: float = Field(default=60.0, ge=10.0, le=600.0)
    reconciliation_interval_seconds: float = Field(default=30.0, ge=10.0, le=3600.0)
    reconciliation_alert_failure_threshold: int = Field(default=3, ge=1, le=20)
    protection_execution_match_bps: float = Field(
        default=10.0,
        ge=1.0,
        le=1000.0,
    )
    auto_close_unmanaged_positions: bool = True
    clock_sync_interval_seconds: int = Field(default=60, ge=60, le=86_400)
    min_ohlcv_candles: int = Field(default=200, ge=200, le=1000)
    max_candle_delay_multiplier: float = Field(default=3.0, ge=1.0, le=20.0)
    allow_zero_volume_candles: bool = False
    max_entry_slippage_bps: float = Field(default=45.0, ge=0.0, le=1000.0)
    max_entry_slippage_bps_per_symbol: dict[str, float] = {}
    entry_fill_resolution_attempts: int = Field(default=5, ge=1, le=10)
    entry_fill_resolution_backoff_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
    )
    reentry_cooldown_seconds: int = Field(default=0, ge=0, le=86_400)
    min_order_notional: float = Field(default=5.0, ge=0.0, le=100000.0)
    max_leverage: int = Field(default=3, ge=1, le=20)
    max_position_size: float = Field(default=0.02, gt=0, le=0.10)
    max_open_positions: int = Field(default=6, ge=1, le=100)
    max_same_direction_positions: int = Field(default=2, ge=1, le=100)
    reverse_on_opposite_signal: bool = True
    max_positions_per_symbol: int = Field(default=2, ge=1, le=20)
    restored_position_exposure_cleanup_enabled: bool = True
    max_total_open_notional_pct: float = Field(default=0.20, gt=0, le=1.0)
    max_symbol_open_notional_pct: float = Field(default=0.10, gt=0, le=1.0)
    correlated_symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT"
    max_correlated_open_notional_pct: float = Field(default=0.15, gt=0, le=1.0)
    max_unfavorable_funding_rate: float = Field(default=0.001, ge=0.0, le=0.10)
    market_depth_levels: int = Field(default=20, ge=1, le=100)
    max_market_depth_slippage_bps: float = Field(default=25.0, ge=0.0, le=1000.0)
    scalp_max_market_depth_slippage_bps: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
    )
    adaptive_limit_entry_enabled: bool = True
    limit_entry_timeout_seconds: float = Field(default=3.0, ge=0.1, le=30.0)
    limit_entry_poll_seconds: float = Field(default=0.5, ge=0.05, le=5.0)
    limit_entry_market_fallback: bool = True
    daily_loss_limit: float = Field(default=0.05, gt=0, le=0.20)
    max_drawdown: float = Field(default=0.15, gt=0, le=0.50)
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.05)
    max_consecutive_losses: int = Field(default=5, ge=1, le=20)
    consecutive_loss_cooldown_seconds: int = Field(default=900, ge=60, le=604_800)
    loss_cooldown_seconds: int = Field(default=0, ge=0, le=86_400)
    min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    strategy_trend_min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    strategy_transition_min_confidence: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )
    strategy_range_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    strategy_breakout_min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    strategy_reversal_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    strategy_countertrend_min_confidence: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
    )
    strategy_scalp_min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    strategy_trend_risk_per_trade: float = Field(default=0.005, gt=0.0, le=0.05)
    strategy_transition_risk_per_trade: float = Field(
        default=0.004,
        gt=0.0,
        le=0.05,
    )
    strategy_range_risk_per_trade: float = Field(default=0.004, gt=0.0, le=0.05)
    strategy_breakout_risk_per_trade: float = Field(default=0.004, gt=0.0, le=0.05)
    strategy_reversal_risk_per_trade: float = Field(default=0.003, gt=0.0, le=0.05)
    strategy_countertrend_risk_per_trade: float = Field(
        default=0.0025,
        gt=0.0,
        le=0.05,
    )
    strategy_min_net_edge_bps: float = Field(default=12.0, ge=0.0, le=100.0)
    strategy_max_spread_bps: float = Field(default=12.0, gt=0.0, le=500.0)
    strategy_max_consecutive_losses: int = Field(default=3, ge=1, le=20)
    strategy_loss_cooldown_seconds: int = Field(default=1800, ge=60, le=604_800)
    ml_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    ta_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    ml_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    require_signal_confluence: bool = False
    strategy_quality_min_score: float = Field(default=0.68, ge=0.0, le=1.0)
    strategy_quality_gate_enforced: bool = True
    # regime_appropriate_strategies() (src/signals/regime.py) was computed
    # but only logged, never enforced - every strategy could fire in every
    # market regime. Backtested via the replay engine on one month of
    # BTC/ETH/BNB 5m data for trend/range/countertrend: enforcing it cut
    # countertrend's trade count roughly in half while its win rate
    # improved on 2 of 3 symbols (e.g. ETHUSDT 23.3% -> 32.1%), and every
    # tested strategy's aggregate loss for the month shrank.
    #
    # Defaulting this to True initially surfaced two real bugs in the
    # allow-list itself (scalp's trend-pullback mode and reversal's
    # opposing-trend requirement were excluded from the "trending" regime
    # they actually need - see regime.py) via live/loop.py test failures.
    # Those are now fixed, but that also means only 3 of the 7 strategies
    # have actually been validated against the corrected table - scalp,
    # breakout, reversal, and transition have not. Defaulting to False
    # until those are backtested too; the mechanism is fully wired and
    # ready to flip once they are.
    regime_filter_enforced: bool = False
    same_symbol_reversal_guard_enabled: bool = True
    lower_timeframe_reversal_min_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
    )
    lower_timeframe_reversal_min_quality_score: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )
    lower_timeframe_reversal_min_hold_seconds: int = Field(
        default=300,
        ge=0,
        le=86_400,
    )
    lower_timeframe_reversal_max_timeframe_ratio: float = Field(
        default=3.0,
        ge=1.0,
        le=96.0,
    )
    strategy_min_adx: float = Field(default=18.0, ge=0.0, le=100.0)
    strategy_min_volume_ratio: float = Field(default=0.70, ge=0.0, le=10.0)
    strategy_min_atr_pct: float = Field(default=0.0005, ge=0.0, le=0.10)
    strategy_max_atr_pct: float = Field(default=0.025, gt=0.0, le=0.50)
    strategy_max_ema_extension_atr: float = Field(default=2.0, gt=0.0, le=10.0)
    strategy_range_max_adx: float = Field(default=25.0, ge=0.0, le=100.0)
    strategy_reversal_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    strategy_breakout_min_volume_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
    )
    strategy_breakout_max_extension_atr: float = Field(
        default=3.0,
        gt=0.0,
        le=10.0,
    )
    strategy_breakout_min_body_ratio: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )
    strategy_breakout_min_close_location: float = Field(
        default=0.60,
        ge=0.50,
        le=1.0,
    )
    strategy_breakout_max_close_location: float = Field(
        default=0.85,
        ge=0.50,
        le=1.0,
    )
    strategy_rejection_min_wick_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )
    scalp_risk_per_trade: float = Field(default=0.003, gt=0.0, le=0.01)
    scalp_atr_stop_multiplier: float = Field(default=1.20, gt=0.0, le=5.0)
    scalp_risk_reward_ratio: float = Field(default=2.00, ge=1.0, le=5.0)
    scalp_min_stop_loss_pct: float = Field(default=0.005, gt=0.0, le=0.02)
    scalp_max_stop_loss_pct: float = Field(default=0.008, gt=0.0, le=0.05)
    scalp_max_spread_bps: float = Field(default=4.0, ge=0.1, le=100.0)
    scalp_max_entry_slippage_bps: float = Field(default=5.0, ge=0.1, le=100.0)
    scalp_estimated_round_trip_fee_bps: float = Field(
        default=8.0,
        ge=0.0,
        le=100.0,
    )
    scalp_min_net_edge_bps: float = Field(default=15.0, ge=0.0, le=100.0)
    scalp_reentry_cooldown_seconds: int = Field(default=0, ge=0, le=3600)
    scalp_websocket_enabled: bool = True
    scalp_demo_websocket_enabled: bool = False
    scalp_stream_fallback_seconds: int = Field(default=20, ge=5, le=300)
    scalp_max_signal_latency_seconds: float = Field(default=12.0, ge=1.0, le=120.0)
    scalp_rest_max_signal_latency_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=180.0,
    )
    scalp_rest_candle_close_grace_seconds: float = Field(
        default=15.0,
        ge=0.0,
        le=45.0,
    )
    scalp_rest_freshness_attempts: int = Field(default=3, ge=1, le=10)
    scalp_rest_freshness_retry_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=30.0,
    )
    scalp_stream_failure_threshold: int = Field(default=3, ge=1, le=20)
    scalp_stream_circuit_seconds: int = Field(default=900, ge=30, le=86_400)
    market_data_concurrency: int = Field(default=4, ge=1, le=20)
    candle_cache_size: int = Field(default=350, ge=201, le=2000)
    scalp_management_interval_seconds: float = Field(default=2.0, ge=0.5, le=30.0)
    scalp_max_hold_seconds_1m: int = Field(default=900, ge=60, le=86_400)
    scalp_max_hold_seconds_3m: int = Field(default=1800, ge=180, le=86_400)
    scalp_max_hold_seconds_default: int = Field(default=3600, ge=300, le=86_400)
    scalp_break_even_trigger_r: float = Field(default=0.75, ge=0.25, le=5.0)
    scalp_break_even_offset_bps: float = Field(default=10.0, ge=0.0, le=100.0)
    scalp_trailing_trigger_r: float = Field(default=1.25, ge=0.5, le=10.0)
    scalp_trailing_distance_r: float = Field(default=0.75, ge=0.1, le=5.0)
    scalp_stop_update_min_bps: float = Field(default=2.0, ge=0.1, le=100.0)
    scalp_partial_profit_enabled: bool = True
    scalp_partial_profit_trigger_r: float = Field(default=1.0, ge=0.25, le=10.0)
    scalp_partial_profit_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)

    # Swing strategies (trend/range/breakout/reversal/countertrend/transition)
    # had no analogous position management at all: a fixed stop and target
    # set at entry, never adjusted, unlike scalp's break-even/trailing/
    # partial-profit handling in live/loop.py's _manage_scalp_position.
    # A trade that goes meaningfully favorable and then reverses all the way
    # back captures none of that unrealized move. These give swing trades
    # the same class of protection scalp already has, at slightly wider
    # triggers reflecting their larger stop distances and longer hold times.
    swing_break_even_trigger_r: float = Field(default=0.75, ge=0.25, le=5.0)
    swing_break_even_offset_bps: float = Field(default=5.0, ge=0.0, le=100.0)
    swing_trailing_trigger_r: float = Field(default=1.25, ge=0.5, le=10.0)
    swing_trailing_distance_r: float = Field(default=0.75, ge=0.1, le=5.0)
    swing_stop_update_min_bps: float = Field(default=2.0, ge=0.1, le=100.0)
    performance_governance_window_days: int = Field(default=30, ge=7, le=365)
    walk_forward_min_trades: int = Field(default=30, ge=5, le=10000)
    walk_forward_max_drawdown_pct: float = Field(default=15.0, gt=0.0, le=100.0)
    performance_governance_min_signals: int = Field(default=30, ge=10, le=10000)
    performance_governance_min_trades: int = Field(default=10, ge=3, le=10000)
    performance_promote_min_accuracy: float = Field(
        default=0.55,
        ge=0.50,
        le=1.0,
    )
    performance_disable_max_accuracy: float = Field(
        default=0.42,
        ge=0.0,
        le=0.50,
    )
    performance_promote_min_return_bps: float = Field(
        default=2.0,
        ge=0.0,
        le=1000.0,
    )
    execution_quality_window_hours: int = Field(default=24, ge=1, le=720)
    execution_quality_min_attempts: int = Field(default=10, ge=3, le=10000)
    execution_quality_warning_slippage_bps: float = Field(
        default=20.0,
        ge=0.0,
        le=1000.0,
    )
    execution_quality_critical_failure_rate: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
    )
    execution_quality_warning_protection_latency_ms: float = Field(
        default=5000.0,
        ge=100.0,
        le=120000.0,
    )
    min_stop_loss_pct: float = Field(default=0.0035, gt=0.0, le=0.10)
    max_stop_loss_pct: float = Field(default=0.02, gt=0.0, le=0.20)
    risk_block_alert_cooldown_seconds: int = Field(default=900, ge=60, le=86_400)

    force_ta_only: bool = True
    auto_retrain_enabled: bool = False
    auto_retrain_on_startup: bool = False
    model_update_interval_hours: int = Field(default=24, ge=1, le=168)
    auto_retrain_check_interval_seconds: int = Field(
        default=300,
        ge=30,
        le=86_400,
    )
    auto_retrain_candle_limit: int = Field(default=1000, ge=300, le=5000)
    auto_retrain_scope_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=60.0,
    )
    feature_lookback: int = Field(default=100, ge=20, le=5000)
    prediction_horizon: int = Field(default=6, ge=1, le=500)
    ml_label_atr_multiplier: float = Field(default=0.50, gt=0.0, le=5.0)
    ml_label_min_return: float = Field(default=0.001, ge=0.0, le=0.10)
    ml_drift_warning_psi: float = Field(default=0.20, ge=0.0, le=10.0)
    ml_drift_severe_psi: float = Field(default=0.30, ge=0.0, le=10.0)
    ml_drift_min_samples: int = Field(default=100, ge=20, le=10000)
    ml_candidate_min_accuracy: float = Field(default=0.45, ge=0.0, le=1.0)
    ml_candidate_min_macro_f1: float = Field(default=0.35, ge=0.0, le=1.0)
    signal_source_gate_enabled: bool = True
    signal_source_gate_min_samples: int = Field(default=3, ge=3, le=10000)
    signal_source_gate_min_accuracy: float = Field(default=0.45, ge=0.0, le=1.0)
    signal_source_gate_min_avg_bps: float = Field(default=2.0, ge=-1000.0, le=1000.0)
    signal_source_gate_lookback: int = Field(default=2000, ge=100, le=100000)

    log_level: str = "INFO"
    log_dir: str = "data/logs"
    json_logs: bool = True
    audit_db_path: str = "data/audit/trading_audit.db"
    runtime_heartbeat_path: str = "data/runtime/trading_heartbeat.json"
    supervisor_check_interval_seconds: float = Field(default=5.0, ge=0.5, le=60.0)
    supervisor_startup_grace_seconds: float = Field(
        default=180.0,
        ge=10.0,
        le=1800.0,
    )
    supervisor_heartbeat_stale_seconds: float = Field(
        default=90.0,
        ge=10.0,
        le=1800.0,
    )
    supervisor_shutdown_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
    )
    supervisor_max_restarts_per_hour: int = Field(default=5, ge=1, le=100)
    supervisor_restart_backoff_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=300.0,
    )
    supervisor_max_backoff_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
    )
    rollout_artifact_path: str = "data/governance/rollout.json"
    mainnet_canary_enabled: bool = True
    mainnet_canary_risk_per_trade: float = Field(default=0.0025, gt=0.0, le=0.05)
    mainnet_canary_max_position_size: float = Field(default=0.005, gt=0.0, le=0.10)
    trading_enabled: bool = True
    require_strategy_approval: bool = True
    strategy_approval_path: str = "data/governance/strategy_approval.json"
    strategy_approval_max_age_hours: int = Field(default=168, ge=1, le=8760)
    min_approval_trades: int = Field(default=20, ge=1, le=10000)
    min_approval_profit_factor: float = Field(default=1.1, ge=0.0, le=10.0)
    max_approval_drawdown_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    require_walk_forward_approval: bool = True
    min_approval_oos_folds: int = Field(default=3, ge=2, le=100)
    min_approval_profitable_fold_ratio: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

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
        if self.min_stop_loss_pct >= self.max_stop_loss_pct:
            raise ValueError("MIN_STOP_LOSS_PCT must be lower than MAX_STOP_LOSS_PCT")
        if self.scalp_min_stop_loss_pct >= self.scalp_max_stop_loss_pct:
            raise ValueError(
                "SCALP_MIN_STOP_LOSS_PCT must be lower than " "SCALP_MAX_STOP_LOSS_PCT"
            )
        if self.strategy_min_atr_pct >= self.strategy_max_atr_pct:
            raise ValueError(
                "STRATEGY_MIN_ATR_PCT must be lower than STRATEGY_MAX_ATR_PCT"
            )
        if (
            self.strategy_breakout_min_close_location
            >= self.strategy_breakout_max_close_location
        ):
            # With min >= max, quality_gate's min <= close_location <= max
            # check (and its mirrored short-side range) is unsatisfiable for
            # every possible candle, silently rejecting every breakout
            # signal regardless of price action even though the strategy
            # stays listed as enabled.
            raise ValueError(
                "STRATEGY_BREAKOUT_MIN_CLOSE_LOCATION must be lower than "
                "STRATEGY_BREAKOUT_MAX_CLOSE_LOCATION"
            )
        minimum_training_rows = (
            self.min_ohlcv_candles + self.prediction_horizon + self.feature_lookback + 1
        )
        if self.auto_retrain_candle_limit < minimum_training_rows:
            raise ValueError(
                "AUTO_RETRAIN_CANDLE_LIMIT must cover MIN_OHLCV_CANDLES + "
                "PREDICTION_HORIZON + FEATURE_LOOKBACK + 1"
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

    @field_validator("enabled_strategies")
    @classmethod
    def validate_enabled_strategies(cls, value: str) -> str:
        strategies = [item.lower() for item in cls._split_csv(value)]
        invalid = sorted(set(strategies) - cls.allowed_strategies)
        if invalid:
            raise ValueError(f"Invalid enabled strategies: {', '.join(invalid)}")
        return ",".join(dict.fromkeys(strategies))

    @field_validator("disabled_strategy_scopes")
    @classmethod
    def validate_disabled_strategy_scopes(cls, value: str) -> str:
        scopes = []
        for raw_scope in cls._split_csv(value):
            parts = raw_scope.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(
                    "DISABLED_STRATEGY_SCOPES entries must use SYMBOL:timeframe "
                    "or SYMBOL:timeframe:strategy"
                )
            symbol, timeframe = parts[0].upper(), parts[1]
            if not re.fullmatch(r"[A-Z0-9]{6,20}", symbol):
                raise ValueError(f"Invalid disabled strategy symbol: {symbol}")
            if timeframe not in cls.allowed_timeframes:
                raise ValueError(f"Invalid disabled strategy timeframe: {timeframe}")
            if len(parts) == 3:
                strategy = parts[2].lower()
                if strategy not in cls.allowed_strategies:
                    raise ValueError(f"Invalid disabled strategy name: {strategy}")
                scopes.append(f"{symbol}:{timeframe}:{strategy}")
            else:
                scopes.append(f"{symbol}:{timeframe}")
        return ",".join(scopes)

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
    def correlated_symbols_set(self) -> set[str]:
        return set(self._split_symbols(self.correlated_symbols))

    @property
    def timeframes_list(self) -> List[str]:
        return [item.strip() for item in self.timeframes.split(",") if item.strip()]

    @property
    def telegram_allowed_user_ids_list(self) -> List[str]:
        return self._split_csv(self.telegram_allowed_user_ids)

    @property
    def disabled_strategy_scopes_set(self) -> set[str]:
        return set(self._split_csv(self.disabled_strategy_scopes))

    @property
    def enabled_strategies_set(self) -> set[str]:
        return set(self._split_csv(self.enabled_strategies))

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
    def scalp_effective_round_trip_fee_bps(self) -> float:
        multiplier = 2.0 if self.binance_demo else 1.0
        return self.scalp_estimated_round_trip_fee_bps * multiplier

    @property
    def ml_effective_label_min_return(self) -> float:
        cost_floor = (
            self.scalp_effective_round_trip_fee_bps + self.strategy_min_net_edge_bps
        ) / 10_000
        return max(self.ml_label_min_return, cost_floor)

    @property
    def scalp_streaming_enabled(self) -> bool:
        return self.scalp_websocket_enabled and (
            not self.binance_demo or self.scalp_demo_websocket_enabled
        )

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
                "defaultType": "swap",
                "adjustForTimeDifference": True,
            },
        }
        if self.binance_demo:
            config["options"]["fetchCurrencies"] = False
        return config


settings = Settings()
