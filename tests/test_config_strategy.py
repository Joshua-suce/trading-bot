import pytest
from pydantic import ValidationError

from src.config import Settings


def test_disabled_strategy_scopes_are_normalized():
    cfg = Settings(
        _env_file=None,
        disabled_strategy_scopes="btcusdt:5m,ETHUSDT:1h",
    )

    assert cfg.disabled_strategy_scopes_set == {
        "BTCUSDT:5m",
        "ETHUSDT:1h",
    }


def test_disabled_strategy_scope_rejects_invalid_format():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, disabled_strategy_scopes="BTCUSDT")


def test_ml_confidence_threshold_is_configurable():
    cfg = Settings(_env_file=None, ml_confidence_threshold=0.55)

    assert cfg.ml_confidence_threshold == 0.55


def test_performance_governance_thresholds_are_validated():
    cfg = Settings(
        _env_file=None,
        performance_governance_min_signals=40,
        performance_promote_min_accuracy=0.60,
        performance_disable_max_accuracy=0.35,
    )

    assert cfg.performance_governance_min_signals == 40
    assert cfg.performance_promote_min_accuracy == 0.60
    assert cfg.performance_disable_max_accuracy == 0.35

    with pytest.raises(ValidationError):
        Settings(_env_file=None, performance_promote_min_accuracy=0.40)


def test_execution_quality_thresholds_are_validated():
    cfg = Settings(
        _env_file=None,
        execution_quality_window_hours=48,
        execution_quality_warning_slippage_bps=15.0,
        execution_quality_critical_failure_rate=0.20,
    )

    assert cfg.execution_quality_window_hours == 48
    assert cfg.execution_quality_warning_slippage_bps == 15.0
    assert cfg.execution_quality_critical_failure_rate == 0.20

    with pytest.raises(ValidationError):
        Settings(_env_file=None, execution_quality_critical_failure_rate=0.0)


def test_price_action_strategy_thresholds_are_validated():
    cfg = Settings(
        _env_file=None,
        strategy_breakout_min_body_ratio=0.60,
        strategy_breakout_min_close_location=0.80,
        strategy_rejection_min_wick_ratio=0.30,
    )

    assert cfg.strategy_breakout_min_body_ratio == 0.60
    assert cfg.strategy_breakout_min_close_location == 0.80
    assert cfg.strategy_rejection_min_wick_ratio == 0.30

    with pytest.raises(ValidationError):
        Settings(_env_file=None, strategy_breakout_min_close_location=0.40)


def test_demo_scalp_fee_estimate_uses_observed_demo_multiplier():
    demo = Settings(
        _env_file=None,
        binance_api_url="https://demo-fapi.binance.com",
        scalp_estimated_round_trip_fee_bps=8.0,
    )
    mainnet = Settings(
        _env_file=None,
        binance_api_url="https://fapi.binance.com",
        scalp_estimated_round_trip_fee_bps=8.0,
    )

    assert demo.scalp_effective_round_trip_fee_bps == 16.0
    assert mainnet.scalp_effective_round_trip_fee_bps == 8.0
    assert demo.ml_effective_label_min_return == 0.0021
    assert mainnet.ml_effective_label_min_return == 0.0013


def test_trade_defaults_keep_model_training_out_of_process():
    cfg = Settings(_env_file=None)

    assert cfg.auto_retrain_enabled is False
    assert cfg.auto_retrain_on_startup is False
    assert cfg.scalp_rest_candle_close_grace_seconds == 15.0
    assert cfg.scalp_rest_freshness_attempts == 3
    assert cfg.scalp_rest_freshness_retry_seconds == 5.0


def test_strategy_contract_risk_and_validation_settings_are_validated():
    cfg = Settings(
        _env_file=None,
        strategy_trend_risk_per_trade=0.004,
        strategy_max_consecutive_losses=4,
        walk_forward_min_trades=40,
    )

    assert cfg.strategy_trend_risk_per_trade == 0.004
    assert cfg.strategy_max_consecutive_losses == 4
    assert cfg.walk_forward_min_trades == 40
