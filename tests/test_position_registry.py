from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.execution.position_registry import PositionRegistry
from src.risk.portfolio import PortfolioManager, TradeRecord


@pytest.fixture
def registry():
    portfolio = MagicMock(spec=PortfolioManager)
    audit_store = MagicMock()
    protection = MagicMock()
    protection.active_stops = {}
    protection.active_tps = {}
    return PositionRegistry(
        portfolio=portfolio,
        audit_store=audit_store,
        protection=protection,
    )


def make_trade(**kwargs):
    return TradeRecord(
        symbol=kwargs.get("symbol", "BTCUSDT"),
        side=kwargs.get("side", "long"),
        entry_price=kwargs.get("entry_price", 100.0),
        quantity=kwargs.get("quantity", 1.0),
        timestamp=kwargs.get("timestamp", datetime.now()),
    )


class TestNormalizeSymbol:
    def test_removes_slash(self, registry):
        assert registry.normalize_symbol("BTC/USDT") == "BTCUSDT"

    def test_removes_futures_suffix(self, registry):
        assert registry.normalize_symbol("BTCUSDT:USDT") == "BTCUSDT"

    def test_uppercases(self, registry):
        assert registry.normalize_symbol("btcusdt") == "BTCUSDT"

    def test_handles_none(self, registry):
        assert registry.normalize_symbol(None) == ""


class TestPositionKey:
    def test_without_timeframe(self, registry):
        assert registry.position_key("BTCUSDT") == "BTCUSDT"

    def test_with_timeframe(self, registry):
        assert registry.position_key("BTCUSDT", "5m") == "BTCUSDT:5m"


class TestTradesForSymbol:
    def test_returns_matching_trades(self, registry):
        t1 = make_trade(symbol="BTCUSDT")
        t2 = make_trade(symbol="ETHUSDT")
        registry.open_trades["BTCUSDT"] = t1
        registry.open_trades["ETHUSDT"] = t2
        result = registry.trades_for_symbol("BTCUSDT")
        assert len(result) == 1
        assert result[0][1] == t1

    def test_returns_empty_when_no_match(self, registry):
        assert registry.trades_for_symbol("BTCUSDT") == []


class TestOpenNotional:
    def test_all_symbols(self, registry):
        registry.open_trades["BTC"] = make_trade(
            symbol="BTCUSDT", entry_price=100.0, quantity=2.0
        )
        registry.open_trades["ETH"] = make_trade(
            symbol="ETHUSDT", entry_price=50.0, quantity=3.0
        )
        assert registry.open_notional() == 350.0

    def test_single_symbol(self, registry):
        registry.open_trades["BTC"] = make_trade(
            symbol="BTCUSDT", entry_price=100.0, quantity=2.0
        )
        registry.open_trades["ETH"] = make_trade(
            symbol="ETHUSDT", entry_price=50.0, quantity=3.0
        )
        assert registry.open_notional(symbol="BTCUSDT") == 200.0

    def test_empty(self, registry):
        assert registry.open_notional() == 0.0


class TestReentryCooldown:
    def test_no_cooldown_when_disabled(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "reentry_cooldown_seconds", 0)
        assert registry.reentry_cooldown_reason("BTCUSDT") == ""

    def test_no_cooldown_when_no_exit(self, registry):
        assert registry.reentry_cooldown_reason("BTCUSDT") == ""

    def test_cooldown_active(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "reentry_cooldown_seconds", 300)
        registry.last_symbol_exit_at["BTCUSDT"] = datetime.now(timezone.utc)
        reason = registry.reentry_cooldown_reason("BTCUSDT")
        assert "cooldown" in reason

    def test_cooldown_expired(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "reentry_cooldown_seconds", 0)
        registry.last_symbol_exit_at["BTCUSDT"] = datetime.now(timezone.utc)
        assert registry.reentry_cooldown_reason("BTCUSDT") == ""

    def test_scalp_uses_shorter_reentry_cooldown(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "reentry_cooldown_seconds", 900)
        monkeypatch.setattr(settings, "scalp_reentry_cooldown_seconds", 60)
        registry.last_symbol_exit_at["BTCUSDT"] = datetime.now(
            timezone.utc
        ) - timedelta(seconds=120)

        assert registry.reentry_cooldown_reason("BTCUSDT", "scalp") == ""
        assert "cooldown active" in registry.reentry_cooldown_reason("BTCUSDT")


class TestPolicyAlertDue:
    def test_returns_true_on_first_call(self, registry):
        assert registry.policy_alert_due("max open positions")

    def test_returns_false_within_cooldown(self, registry, monkeypatch):
        monkeypatch.setattr(settings, "risk_block_alert_cooldown_seconds", 60)
        registry.policy_alert_due("max open positions")
        assert not registry.policy_alert_due("max open positions")

    def test_separate_categories_independent(self, registry):
        assert registry.policy_alert_due("max open positions")
        assert registry.policy_alert_due("equity limit")


class TestRecordExitTime:
    def test_records_iso_timestamp(self, registry):
        registry.record_exit_time("BTCUSDT")
        assert "BTCUSDT" in registry.last_symbol_exit_at
        assert registry.last_symbol_exit_at["BTCUSDT"].tzinfo is not None


class TestRestoreOpenTradesFromAudit:
    def test_restores_audited_trades(self, registry):
        registry.audit_store.load_open_trades.return_value = [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "entry_price": 100.0,
                "quantity": 1.0,
                "opened_at": "2026-06-10T00:00:00+00:00",
                "correlation_id": "corr-1",
                "stop_order_id": "sl-1",
                "take_profit_order_id": "tp-1",
                "strategy": "breakout",
            },
        ]
        count = registry.restore_open_trades_from_audit()
        assert count == 1
        assert "BTCUSDT" in registry.open_trades
        assert registry.trade_correlation_ids["BTCUSDT"] == "corr-1"
        assert registry.protection.active_stops["BTCUSDT"] == "sl-1"
        assert registry.protection.active_tps["BTCUSDT"] == "tp-1"
        assert registry.open_trades["BTCUSDT"].strategy == "breakout"

    def test_skips_duplicate_trades(self, registry):
        existing = make_trade()
        registry.open_trades["BTCUSDT"] = existing
        registry.audit_store.load_open_trades.return_value = [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "entry_price": 100.0,
                "quantity": 1.0,
                "opened_at": "2026-06-10T00:00:00+00:00",
                "correlation_id": "corr-1",
            },
        ]
        count = registry.restore_open_trades_from_audit()
        assert count == 0
        registry.audit_store.activate_emergency_stop.assert_called_once()


class TestRestoreRecentExitCooldowns:
    def test_restores_closed_trades(self, registry):
        registry.audit_store.load_closed_trades.return_value = [
            {"symbol": "BTCUSDT", "closed_at": "2026-06-10T00:00:00+00:00"},
        ]
        count = registry.restore_recent_exit_cooldowns()
        assert count == 1
        assert "BTCUSDT" in registry.last_symbol_exit_at

    def test_skips_missing_symbol(self, registry):
        registry.audit_store.load_closed_trades.return_value = [
            {"closed_at": "2026-06-10T00:00:00+00:00"},
        ]
        assert registry.restore_recent_exit_cooldowns() == 0
