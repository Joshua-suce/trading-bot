import json
from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.metrics import BacktestMetrics
from src.config import Settings
from src.governance import StrategyApprovalStore
from src.preflight import run_preflight


def metrics(
    total_trades: int = 25,
    profit_factor: float = 1.4,
    drawdown_pct: float = 8.0,
) -> BacktestMetrics:
    return BacktestMetrics(
        total_return=100.0,
        total_return_pct=1.0,
        annualized_return=0.1,
        total_trades=total_trades,
        win_rate=55.0,
        profit_factor=profit_factor,
        sharpe_ratio=1.0,
        sortino_ratio=1.2,
        calmar_ratio=1.0,
        max_drawdown=80.0,
        max_drawdown_pct=drawdown_pct,
        avg_win=10.0,
        avg_loss=-6.0,
        largest_win=20.0,
        largest_loss=-12.0,
        avg_holding_periods=3.0,
    )


def cfg(tmp_path, **overrides) -> Settings:
    defaults = {
        "binance_api_key": "key",
        "binance_api_secret": "secret",
        "binance_api_url": "https://demo-fapi.binance.com",
        "symbols": "BTCUSDT",
        "timeframes": "1h",
        "strategy_approval_path": str(tmp_path / "approval.json"),
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def test_strategy_approval_store_writes_and_validates(tmp_path):
    settings = cfg(tmp_path)
    store = StrategyApprovalStore(cfg=settings)

    store.write(
        metrics=metrics(),
        symbol="BTCUSDT",
        timeframe="1h",
        approved_by="tester",
        reason="unit test",
    )

    approved, reason = store.validate_for_mainnet()
    assert approved is True
    assert reason == "strategy approval valid"


def test_strategy_approval_rejects_weak_metrics(tmp_path):
    settings = cfg(tmp_path)
    store = StrategyApprovalStore(cfg=settings)
    store.write(
        metrics=metrics(total_trades=2, profit_factor=0.8, drawdown_pct=30.0),
        symbol="BTCUSDT",
        timeframe="1h",
        approved_by="tester",
        reason="weak",
    )

    approved, reason = store.validate_for_mainnet()
    assert approved is False
    assert "too few trades" in reason


def test_strategy_approval_rejects_expired_artifact(tmp_path):
    settings = cfg(tmp_path, strategy_approval_max_age_hours=1)
    store = StrategyApprovalStore(cfg=settings)
    approval = store.write(
        metrics=metrics(),
        symbol="BTCUSDT",
        timeframe="1h",
        approved_by="tester",
        reason="old",
    )
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    payload = {
        **approval.__dict__,
        "approved_at": old.isoformat(),
    }
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    approved, reason = store.validate_for_mainnet()
    assert approved is False
    assert "expired" in reason


def test_mainnet_preflight_requires_strategy_approval(tmp_path):
    with pytest.raises(RuntimeError, match="strategy governance"):
        run_preflight(
            "trade",
            cfg=cfg(
                tmp_path,
                binance_api_url="https://fapi.binance.com",
                allow_mainnet_trading=True,
            ),
        )


def test_mainnet_preflight_accepts_valid_strategy_approval(tmp_path):
    settings = cfg(
        tmp_path,
        binance_api_url="https://fapi.binance.com",
        allow_mainnet_trading=True,
    )
    StrategyApprovalStore(cfg=settings).write(
        metrics=metrics(),
        symbol="BTCUSDT",
        timeframe="1h",
        approved_by="tester",
        reason="valid",
    )

    result = run_preflight("trade", cfg=settings)

    assert result.mode == "trade"
