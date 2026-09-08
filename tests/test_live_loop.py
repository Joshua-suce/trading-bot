import asyncio
import inspect
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.exchange.account import AccountInfo
from src.exchange.client import BinanceDemoAccountInactiveError
from src.live import loop as live_loop_module
from src.live.loop import LiveTradingLoop
from src.risk.portfolio import TradeRecord
from src.signals.aggregator import FinalSignal
from src.signals.quality_gate import StrategyQuality
from src.signals.signal_gate import SignalGateResult


@pytest.fixture(autouse=True)
def isolate_live_loop_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "audit_db_path",
        str(tmp_path / "live-loop-audit.db"),
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "model_dir",
        str(tmp_path / "models"),
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "runtime_heartbeat_path",
        str(tmp_path / "heartbeat.json"),
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "disabled_strategy_scopes",
        "",
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "auto_retrain_enabled",
        False,
    )


def test_live_loop_uses_injected_audit_store(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "injected.db")
    bot = LiveTradingLoop(audit_store=audit)

    assert bot.audit_store is audit
    assert bot.order_mgr.audit_store is audit
    assert bot.pos_mgr.audit_store is audit


def test_signal_observation_records_evaluated_strategy(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "strategy-observation.db")
    bot = LiveTradingLoop(audit_store=audit)
    signal = FinalSignal(
        direction=1,
        confidence=0.8,
        ta_source="trend_structure_bull+structure_score_1.00",
        strategy="trend",
    )
    candle = {
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
        "close": 100.0,
    }

    bot._record_signal_observation(
        candle,
        signal,
        strategy="breakout",
        minimum_confidence=0.5,
        decision="accepted",
        reason="test",
    )

    rows = audit.load_signal_observations(limit=1)
    assert rows[0]["strategy"] == "breakout"


def test_candle_cache_deduplicates_and_keeps_latest_rows():
    bot = LiveTradingLoop()
    first = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="1min"),
    )
    update = pd.DataFrame(
        {"close": [102.0, 103.0]},
        index=pd.date_range("2026-01-01 00:01", periods=2, freq="1min"),
    )

    cached = bot._cache_market_frame("BTCUSDT", "1m", first)
    cached = bot._cache_market_frame("BTCUSDT", "1m", update)

    assert len(cached) == 3
    assert cached.iloc[1]["close"] == 102.0


def test_healthy_scalp_stream_suppresses_rest_scan(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "symbols", "BTCUSDT")
    monkeypatch.setattr(live_loop_module.settings, "timeframes", "1m,5m")
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_demo_websocket_enabled",
        True,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_websocket_enabled",
        True,
    )
    bot = LiveTradingLoop()
    bot._scalp_stream_heartbeat["BTCUSDT_1m"] = time.monotonic()

    pairs = bot._due_scan_pairs(time.monotonic())

    assert ("BTCUSDT", "1m") not in pairs
    assert ("BTCUSDT", "5m") in pairs


def test_stale_scalp_stream_restores_rest_fallback(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "symbols", "BTCUSDT")
    monkeypatch.setattr(live_loop_module.settings, "timeframes", "1m")
    bot = LiveTradingLoop()
    bot._scalp_stream_heartbeat["BTCUSDT_1m"] = (
        time.monotonic() - live_loop_module.settings.scalp_stream_fallback_seconds - 1
    )

    assert ("BTCUSDT", "1m") in bot._due_scan_pairs(time.monotonic())


def test_demo_defaults_to_rest_scalp_fast_lane(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "binance_api_url",
        "https://demo-fapi.binance.com",
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_websocket_enabled",
        True,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_demo_websocket_enabled",
        False,
    )
    bot = LiveTradingLoop()

    bot._start_scalp_streams()

    assert bot._scalp_stream_tasks == []


def test_mainnet_can_enable_scalp_websockets(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "binance_api_url",
        "https://fapi.binance.com",
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_websocket_enabled",
        True,
    )

    assert live_loop_module.settings.scalp_streaming_enabled


def test_closed_candle_latency_uses_timeframe_close():
    candle = {
        "timestamp": pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=65),
        "timeframe": "1m",
    }

    latency = LiveTradingLoop._closed_candle_latency_seconds(candle)

    assert 4 <= latency <= 7


def test_due_scan_pairs_prioritizes_scalp_timeframes(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "symbols", "BTCUSDT,ETHUSDT")
    monkeypatch.setattr(live_loop_module.settings, "timeframes", "1h,3m,1m")
    monkeypatch.setattr(live_loop_module.settings, "scalp_websocket_enabled", False)
    bot = LiveTradingLoop()

    pairs = bot._due_scan_pairs(100.0)

    assert pairs[:4] == [
        ("BTCUSDT", "1m"),
        ("ETHUSDT", "1m"),
        ("BTCUSDT", "3m"),
        ("ETHUSDT", "3m"),
    ]


@pytest.mark.asyncio
async def test_rest_scalp_uses_fallback_latency_budget(monkeypatch):
    bot = LiveTradingLoop()
    monkeypatch.setattr(
        live_loop_module.settings,
        "strategy_quality_gate_enforced",
        True,
    )
    signal = FinalSignal(
        direction=1,
        confidence=0.9,
        ta_source="scalp",
        strategy="scalp",
    )
    bot._entry_runtime_blocked = MagicMock(return_value=False)
    bot.aggregator = MagicMock()
    bot.aggregator.generate.return_value = [signal]
    bot._record_signal_observation = MagicMock()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(False, 0.0, "stop after latency check", {})
    )
    monkeypatch.setattr(
        bot,
        "_closed_candle_latency_seconds",
        MagicMock(return_value=30.0),
    )
    candle = {
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "timestamp": pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1),
        "close": 100.0,
        "market_data_source": "rest",
    }

    await bot._execute_trade(
        candle,
        df_ind=pd.DataFrame({"close": [100.0], "atr": [1.0]}),
    )

    assert bot.strategy_quality.evaluate.called


def test_responsive_scalp_stop_advances_to_break_even(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_break_even_offset_bps",
        10.0,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_estimated_round_trip_fee_bps",
        8.0,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_min_net_edge_bps",
        3.0,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "binance_api_url",
        "https://demo-fapi.binance.com",
    )
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(timezone.utc),
        timeframe="1m",
        strategy="scalp",
    )

    stop = LiveTradingLoop._responsive_scalp_stop(
        trade,
        price=101.0,
        peak=101.0,
        current_stop=99.0,
        initial_risk=1.0,
        favorable_r=1.0,
    )

    assert stop == 100.19


@pytest.mark.asyncio
async def test_scalp_max_hold_exits_position(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "scalp_max_hold_seconds_1m", 60)
    bot = LiveTradingLoop()
    bot.pos_mgr.exit_position = AsyncMock()
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=61),
        timeframe="1m",
        strategy="scalp",
    )

    await bot._manage_scalp_position("BTCUSDT:1m", trade)

    bot.pos_mgr.exit_position.assert_awaited_once_with(
        "BTCUSDT:1m",
        "scalp maximum hold",
    )


def test_responsive_swing_stop_advances_to_break_even(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "swing_break_even_offset_bps", 5.0)
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(timezone.utc),
        timeframe="5m",
        strategy="trend",
    )

    stop = LiveTradingLoop._responsive_swing_stop(
        trade,
        price=101.0,
        peak=101.0,
        current_stop=99.0,
        initial_risk=1.0,
        favorable_r=1.0,
    )

    assert stop == 100.05


@pytest.mark.asyncio
async def test_manage_scalp_positions_if_due_also_manages_swing_positions():
    bot = LiveTradingLoop()
    now = datetime.now(timezone.utc)
    bot.pos_mgr.open_trades["BTCUSDT:1m:scalp"] = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=now,
        timeframe="1m",
        strategy="scalp",
    )
    bot.pos_mgr.open_trades["ETHUSDT:5m:trend"] = TradeRecord(
        symbol="ETHUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=now,
        timeframe="5m",
        strategy="trend",
    )
    bot.pos_mgr.open_trades["BNBUSDT:5m:trend"] = TradeRecord(
        symbol="BNBUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=now - timedelta(hours=6),
        timeframe="5m",
        strategy="trend",
    )
    bot._manage_scalp_position = AsyncMock()
    bot._manage_swing_position = AsyncMock()
    bot._exit_position_safely = AsyncMock(return_value=True)

    await bot._manage_scalp_positions_if_due()

    bot._manage_scalp_position.assert_awaited_once()
    assert bot._manage_scalp_position.await_args.args[0] == "BTCUSDT:1m:scalp"
    bot._manage_swing_position.assert_awaited_once()
    assert bot._manage_swing_position.await_args.args[0] == "ETHUSDT:5m:trend"
    bot._exit_position_safely.assert_awaited_once_with(
        "BNBUSDT:5m:trend", "maximum hold"
    )


def test_non_scalp_holding_period_is_timeframe_aware():
    bot = LiveTradingLoop()
    now = datetime.now(timezone.utc)
    five_minute_trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=now - timedelta(minutes=61),
        timeframe="5m",
        strategy="trend",
    )
    hourly_trade = TradeRecord(
        symbol="ETHUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=now - timedelta(minutes=61),
        timeframe="1h",
        strategy="trend",
    )

    assert bot._strategy_max_hold_reached(five_minute_trade)
    assert not bot._strategy_max_hold_reached(hourly_trade)


class NoopAlerter:
    pending_messages = 0

    async def start(self, command_handler=None):
        return None

    async def initializing_alert(self, mode, environment):
        return None

    async def startup_alert(self, mode, environment, symbols):
        return None

    async def trade_failed_alert(self, mode, symbol, reason):
        return None

    async def data_feed_alert(self, mode, symbol, timeframe, reason):
        return None

    async def error_alert(self, error):
        return None

    async def shutdown_alert(self, mode, environment, reason):
        return None

    async def stop(self):
        return None


class NoopClient:
    async def connect(self):
        return None

    async def close(self):
        return None


class LifecycleAlerter(NoopAlerter):
    def __init__(self):
        self.shutdown_reasons = []

    async def shutdown_alert(self, mode, environment, reason):
        self.shutdown_reasons.append(reason)


@pytest.mark.asyncio
async def test_trade_account_refresh_failure_is_non_fatal_before_threshold(monkeypatch):
    bot = LiveTradingLoop()

    async def fail_account(_client):
        raise RuntimeError("temporary demo account outage")

    monkeypatch.setattr(live_loop_module, "get_account_info", fail_account)

    await bot._refresh_account()

    assert bot._account_refresh_failures == 1


@pytest.mark.asyncio
async def test_manual_request_processor_completes_successful_request(monkeypatch):
    bot = LiveTradingLoop()
    request_id = bot.audit_store.create_manual_trade_request(
        action="open",
        symbol="BTCUSDT",
        side="long",
        timeframe="1h",
        reason="test request",
    )
    execute = AsyncMock(
        return_value={
            "success": True,
            "reason": "opened",
            "correlation_id": "manual-1",
        }
    )
    monkeypatch.setattr(bot, "_execute_manual_trade_request", execute)

    await bot._process_manual_trade_requests()

    request = bot.audit_store.load_manual_trade_requests(1)[0]
    assert request["request_id"] == request_id
    assert request["status"] == "completed"
    assert request["result"]["correlation_id"] == "manual-1"


@pytest.mark.asyncio
async def test_manual_result_persistence_failure_does_not_crash_loop(monkeypatch):
    bot = LiveTradingLoop()
    bot.audit_store.create_manual_trade_request(
        action="open",
        symbol="BTCUSDT",
        side="long",
        timeframe="1h",
        reason="test request",
    )
    monkeypatch.setattr(
        bot,
        "_execute_manual_trade_request",
        AsyncMock(return_value={"success": True, "correlation_id": "manual-2"}),
    )
    monkeypatch.setattr(
        bot.audit_store,
        "finish_manual_trade_request",
        MagicMock(side_effect=RuntimeError("database unavailable")),
    )

    await bot._process_manual_trade_requests()

    events = bot.audit_store.load_recent_events(5)
    assert any(
        event["event_type"] == "manual_trade_result_persistence_failed"
        for event in events
    )


@pytest.mark.asyncio
async def test_manual_open_rejects_disabled_scope(monkeypatch):
    bot = LiveTradingLoop()
    monkeypatch.setattr(
        live_loop_module.settings,
        "disabled_strategy_scopes",
        "BTCUSDT:1h",
    )

    result = await bot._execute_manual_trade_request(
        {
            "action": "open",
            "symbol": "BTCUSDT",
            "side": "long",
            "timeframe": "1h",
            "reason": "test",
            "options": {"require_quality": False},
        }
    )

    assert result["success"] is False
    assert "disabled" in str(result["reason"])


@pytest.mark.asyncio
async def test_manual_open_uses_validated_closed_candle(monkeypatch):
    bot = LiveTradingLoop()
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [10.0, 11.0, 12.0],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
    )
    bot.client.fetch_ohlcv = AsyncMock(return_value=frame)
    bot.data_quality.validate = MagicMock(
        return_value=MagicMock(valid=True, reason="ok")
    )

    def fake_indicators(closed):
        result = closed.copy()
        result["atr"] = 2.5
        return result

    monkeypatch.setattr(
        "src.indicators.compute.compute_all_indicators",
        fake_indicators,
    )

    async def enter_long(
        symbol,
        price,
        atr,
        leverage,
        timeframe,
        signal_timestamp=None,
    ):
        assert (symbol, price, atr, leverage, timeframe) == (
            "BTCUSDT",
            102.0,
            2.5,
            live_loop_module.settings.max_leverage,
            "1h",
        )
        assert signal_timestamp == frame.index[-2]
        bot.pos_mgr.trades.trade_correlation_ids["BTCUSDT:1h"] = "manual-open-1"
        return True

    bot.pos_mgr.enter_long = enter_long

    result = await bot._execute_manual_trade_request(
        {
            "action": "open",
            "symbol": "BTCUSDT",
            "side": "long",
            "timeframe": "1h",
            "reason": "operator entry",
            "options": {"leverage": 2, "require_quality": False},
        }
    )

    assert result["success"] is True
    assert result["correlation_id"] == "manual-open-1"
    assert result["exchange_leverage"] == live_loop_module.settings.max_leverage


@pytest.mark.asyncio
async def test_manual_entry_quality_rejection_blocks_order(monkeypatch):
    bot = LiveTradingLoop()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(
            accepted=False,
            score=0.4,
            reason="trend strength too low",
            metrics={"adx": 10.0},
        )
    )

    result = await bot._manual_entry_quality(
        "BTCUSDT",
        "1h",
        "long",
        pd.DataFrame({"close": [100.0]}),
        True,
    )

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["quality_score"] == 0.4


@pytest.mark.asyncio
async def test_manual_bulk_close_all_reports_remaining_positions():
    bot = LiveTradingLoop()
    bot.pos_mgr.close_all = AsyncMock()
    bot.pos_mgr.trades.open_trades["BTCUSDT:1h"] = MagicMock()

    result = await bot._execute_manual_bulk_close(
        "close-all",
        "ALL",
        "operator close",
    )

    assert result["success"] is False
    assert result["remaining_positions"] == 1


@pytest.mark.asyncio
async def test_manual_close_targets_exact_correlation_id():
    from src.risk.portfolio import TradeRecord

    bot = LiveTradingLoop()
    position_key = "BTCUSDT:1h"
    bot.pos_mgr.trades.open_trades[position_key] = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=pd.Timestamp.now(),
        timeframe="1h",
    )
    bot.pos_mgr.trades.trade_correlation_ids[position_key] = "close-me"

    async def successful_exit(key, reason):
        assert key == position_key
        assert reason == "dashboard: operator close"
        del bot.pos_mgr.trades.open_trades[key]

    bot.pos_mgr.exit_position = successful_exit

    result = await bot._execute_manual_close(
        {
            "symbol": "BTCUSDT",
            "correlation_id": "close-me",
        },
        "operator close",
    )

    assert result["success"] is True
    assert result["correlation_id"] == "close-me"


@pytest.mark.asyncio
async def test_trade_start_stops_after_initial_account_refresh_failure(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = NoopAlerter()
    reached = []

    async def fail_account(_client):
        raise RuntimeError(
            "binanceusdm GET https://demo-fapi.binance.com/fapi/v3/account"
            "?timestamp=1&signature=secret"
        )

    async def reconcile():
        reached.append("reconcile")
        return True

    async def cancel_scan():
        reached.append("scan")
        raise asyncio.CancelledError

    async def stop(reason):
        reached.append("stop")

    monkeypatch.setattr(live_loop_module, "get_account_info", fail_account)
    monkeypatch.setattr(bot.pos_mgr.trades, "restore_open_trades_from_audit", lambda: 0)
    monkeypatch.setattr(bot.pos_mgr, "reconcile_exchange_state", reconcile)
    monkeypatch.setattr(bot, "_scan_due_timeframes_once", cancel_scan)
    monkeypatch.setattr(bot, "stop", stop)

    await bot.start()

    assert reached == ["stop"]
    assert bot.portfolio.account is None
    assert bot._account_refresh_failures == 1
    assert "secret" not in bot._describe_exception(RuntimeError("signature=secret"))


@pytest.mark.asyncio
async def test_startup_reconciliation_unavailable_preserves_positions(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = NoopAlerter()
    stopped = []

    async def refresh_account(required=False):
        return None

    async def reconcile(*args, **kwargs):
        return None

    async def stop(reason, close_positions=None):
        stopped.append((reason, close_positions))

    async def cancel_scan():
        raise asyncio.CancelledError

    monkeypatch.setattr(bot, "_refresh_account", refresh_account)
    monkeypatch.setattr(bot.pos_mgr.trades, "restore_open_trades_from_audit", lambda: 1)
    monkeypatch.setattr(bot.pos_mgr, "reconcile_exchange_state", reconcile)
    monkeypatch.setattr(bot, "_scan_due_timeframes_once", cancel_scan)
    monkeypatch.setattr(bot, "stop", stop)

    await bot.start()

    assert stopped == [("cancelled", None)]


@pytest.mark.asyncio
async def test_restored_position_exposure_cleanup_closes_oldest_excess_position(
    tmp_path,
    monkeypatch,
):
    from src.audit import AuditStore

    monkeypatch.setattr(live_loop_module.settings, "max_positions_per_symbol", 2)
    monkeypatch.setattr(live_loop_module.settings, "max_same_direction_positions", 2)
    monkeypatch.setattr(
        live_loop_module.settings,
        "restored_position_exposure_cleanup_enabled",
        True,
    )
    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "restore-cleanup.db"))
    now = datetime.now(timezone.utc)
    bot.pos_mgr.open_trades["BTCUSDT:old:trend"] = TradeRecord(
        symbol="BTCUSDT",
        side="short",
        entry_price=100.0,
        quantity=0.1,
        timestamp=now - timedelta(hours=3),
        timeframe="5m",
        strategy="trend",
    )
    bot.pos_mgr.open_trades["ETHUSDT:mid:trend"] = TradeRecord(
        symbol="ETHUSDT",
        side="short",
        entry_price=100.0,
        quantity=0.1,
        timestamp=now - timedelta(hours=2),
        timeframe="15m",
        strategy="trend",
    )
    bot.pos_mgr.open_trades["BNBUSDT:new:trend"] = TradeRecord(
        symbol="BNBUSDT",
        side="short",
        entry_price=100.0,
        quantity=0.1,
        timestamp=now - timedelta(hours=1),
        timeframe="15m",
        strategy="trend",
    )

    async def exit_position(position_key, _reason):
        bot.pos_mgr.open_trades.pop(position_key, None)

    bot.pos_mgr.exit_position = AsyncMock(side_effect=exit_position)

    await bot._cleanup_restored_position_exposure()

    bot.pos_mgr.exit_position.assert_awaited_once_with(
        "BTCUSDT:old:trend",
        "startup restored exposure limit",
    )
    assert set(bot.pos_mgr.open_trades) == {
        "ETHUSDT:mid:trend",
        "BNBUSDT:new:trend",
    }


@pytest.mark.asyncio
async def test_directional_entry_reservation_counts_pending_entries(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "max_same_direction_positions", 2)
    bot = LiveTradingLoop()
    bot.pos_mgr.open_trades["BTCUSDT:5m:trend"] = TradeRecord(
        symbol="BTCUSDT",
        side="short",
        entry_price=100.0,
        quantity=0.1,
        timestamp=datetime.now(timezone.utc),
        timeframe="5m",
        strategy="trend",
    )

    first_reserved, first_reason = await bot._reserve_directional_entry("short")
    second_reserved, second_reason = await bot._reserve_directional_entry("short")

    assert first_reserved is True
    assert first_reason == ""
    assert second_reserved is False
    assert "existing/pending short" in second_reason

    await bot._release_directional_entry("short")
    assert bot._pending_entry_sides["short"] == 0


@pytest.mark.asyncio
async def test_trade_start_stops_cleanly_when_demo_account_is_inactive(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = NoopAlerter()
    stopped = []

    async def refresh_account(required=False):
        account = AccountInfo(
            total_equity=100.0,
            wallet_balance=100.0,
            available_balance=100.0,
            unrealized_pnl=0.0,
            margin_ratio=0.0,
        )
        bot.portfolio.update_account(account)

    async def reconcile(*args, **kwargs):
        return True

    async def set_leverage(symbol, leverage):
        raise BinanceDemoAccountInactiveError("inactive")

    async def stop(reason, close_positions=None):
        stopped.append((reason, close_positions))

    monkeypatch.setattr(bot, "_refresh_account", refresh_account)
    monkeypatch.setattr(bot.pos_mgr.trades, "restore_open_trades_from_audit", lambda: 0)
    monkeypatch.setattr(bot.pos_mgr, "reconcile_exchange_state", reconcile)
    monkeypatch.setattr(bot.client, "set_leverage", set_leverage, raising=False)
    monkeypatch.setattr(bot, "stop", stop)

    await bot.start()

    assert stopped == [("demo account inactive", None)]


@pytest.mark.asyncio
async def test_graceful_shutdown_preserves_protected_positions(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = LifecycleAlerter()
    bot.pos_mgr.open_trades["BTCUSDT"] = object()
    closed = []

    async def close_all():
        closed.append(True)

    monkeypatch.setattr(bot.pos_mgr, "close_all", close_all)

    await bot.stop("cancelled")

    assert closed == []
    assert bot.pos_mgr.open_trades
    events = bot.audit_store.load_recent_events(2)
    assert any(
        event["event_type"] == "positions_preserved_on_shutdown" for event in events
    )


@pytest.mark.asyncio
async def test_fatal_shutdown_flattens_positions(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = LifecycleAlerter()
    closed = []

    async def close_all():
        closed.append(True)
        return True

    monkeypatch.setattr(bot.pos_mgr, "close_all", close_all)

    await bot.stop("fatal error")

    assert closed == [True]
    events = bot.audit_store.load_recent_events(5)
    assert not any(event["event_type"] == "shutdown_close_all_failed" for event in events)


@pytest.mark.asyncio
async def test_fatal_shutdown_preserves_positions_when_close_all_returns_false(
    monkeypatch,
):
    # TradeExecutor.close_all() catches per-symbol failures internally and
    # returns False instead of raising - this is the path that actually
    # fires in real operation now, not an exception. Previously
    # PositionManager.close_all() discarded this bool entirely, so
    # loop.stop()'s failure handling (the shutdown_close_all_failed audit
    # event) could never fire for a real failure, only for the exception
    # path covered by the sibling test below.
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = LifecycleAlerter()
    bot.pos_mgr.open_trades["BTCUSDT:1m:scalp"] = object()

    async def close_all():
        return False

    monkeypatch.setattr(bot.pos_mgr, "close_all", close_all)

    await bot.stop("fatal error")

    assert bot.pos_mgr.open_trades
    events = bot.audit_store.load_recent_events(5)
    assert any(event["event_type"] == "shutdown_close_all_failed" for event in events)


@pytest.mark.asyncio
async def test_fatal_shutdown_preserves_positions_when_close_all_fails(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = LifecycleAlerter()
    bot.pos_mgr.open_trades["BTCUSDT:1m:scalp"] = object()

    async def close_all():
        raise RuntimeError("exchange unavailable")

    monkeypatch.setattr(bot.pos_mgr, "close_all", close_all)

    await bot.stop("fatal error")

    assert bot.pos_mgr.open_trades
    events = bot.audit_store.load_recent_events(5)
    assert any(event["event_type"] == "shutdown_close_all_failed" for event in events)


@pytest.mark.asyncio
async def test_heartbeat_failure_does_not_escape_live_loop(monkeypatch):
    bot = LiveTradingLoop()

    async def fail_write(*args, **kwargs):
        raise PermissionError("temporarily locked")

    monkeypatch.setattr(bot._heartbeat, "write_async", fail_write)

    await bot._write_heartbeat("running", open_positions=0)


@pytest.mark.asyncio
async def test_managed_exit_failure_does_not_escape_loop(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "managed-exit-failed.db")
    bot = LiveTradingLoop(audit_store=audit)

    async def exit_position(_position_key, _reason):
        raise RuntimeError("exchange unavailable")

    bot.pos_mgr.exit_position = AsyncMock(side_effect=exit_position)

    result = await bot._exit_position_safely("ETHUSDT:15m:trend", "maximum hold")

    assert result is False
    events = audit.load_recent_events(1)
    assert events[0]["event_type"] == "managed_exit_failed"


@pytest.mark.asyncio
async def test_heartbeat_publisher_runs_independently_of_scan_loop(monkeypatch):
    bot = LiveTradingLoop()
    writes = []

    def record_write(state, **details):
        writes.append((state, details))

    async def stop_after_first_interval(_seconds):
        bot._stopped = True

    monkeypatch.setattr(bot, "_write_heartbeat", record_write)
    monkeypatch.setattr(live_loop_module.asyncio, "sleep", stop_after_first_interval)

    await bot._heartbeat_publisher()

    assert writes[0][0] == "running"
    assert writes[0][1]["open_positions"] == 0


@pytest.mark.asyncio
async def test_slow_pair_marks_progress_without_finishing_a_sweep(monkeypatch):
    # A merely-slow sweep must not look like a hang: each completed
    # (symbol, timeframe) pair refreshes the watchdog's clock, so a sweep
    # that legitimately runs for minutes never trips the stall detector.
    bot = LiveTradingLoop()
    monkeypatch.setattr(bot, "_network_outage_active", lambda: False)
    monkeypatch.setattr(bot, "_process_timeframe", AsyncMock())
    bot._last_loop_progress_at = time.monotonic() - 10_000

    assert await bot._bounded_process_timeframe("BTCUSDT", "1h") is True
    assert time.monotonic() - bot._last_loop_progress_at < 1.0


@pytest.mark.asyncio
async def test_outage_short_circuit_does_not_mark_progress(monkeypatch):
    # Zero-work short-circuit, not a completed unit - must not refresh.
    bot = LiveTradingLoop()
    monkeypatch.setattr(bot, "_network_outage_active", lambda: True)
    stale = time.monotonic() - 10_000
    bot._last_loop_progress_at = stale

    assert await bot._bounded_process_timeframe("BTCUSDT", "1h") is False
    assert bot._last_loop_progress_at == stale


@pytest.mark.asyncio
async def test_hung_pair_freezes_progress_once_siblings_drain(monkeypatch):
    # The watchdog must still catch a genuine deadlock even though healthy
    # siblings in the same gather keep marking progress for a while. The
    # sibling pool is finite and drains, after which the clock freezes.
    bot = LiveTradingLoop()
    monkeypatch.setattr(bot, "_network_outage_active", lambda: False)
    hang = asyncio.Event()  # never set

    async def process(_symbol, timeframe):
        if timeframe == "1h":
            await hang.wait()

    monkeypatch.setattr(bot, "_process_timeframe", process)
    sweep = asyncio.gather(
        bot._bounded_process_timeframe("BTCUSDT", "1h"),
        bot._bounded_process_timeframe("ETHUSDT", "15m"),
    )
    await asyncio.sleep(0.05)
    drained_at = bot._last_loop_progress_at
    await asyncio.sleep(0.2)

    assert bot._last_loop_progress_at == drained_at  # marker frozen
    assert not sweep.done()
    sweep.cancel()
    with suppress(asyncio.CancelledError):
        await sweep


def test_progress_is_not_marked_on_the_shared_stream_path():
    # _process_market_frame and _on_candle beneath it are reachable from
    # the fire-and-forget scalp stream tasks, which the main loop never
    # awaits. A progress mark in either would let a healthy WebSocket
    # stream mask a deadlocked main loop forever, defeating the watchdog.
    # (_process_timeframe is deliberately NOT in this list - it is only
    # reached from the awaited scan gather, so marking there is safe and
    # is what covers the order-placement chain.)
    for method in (
        LiveTradingLoop._process_market_frame,
        LiveTradingLoop._on_candle,
    ):
        assert "_mark_loop_progress" not in inspect.getsource(method)


def test_process_timeframe_marks_progress_around_the_order_chain():
    # Placing an entry issues four sequential exchange round trips inside
    # _process_market_frame; without marks on both sides of it the whole
    # order chain is invisible to the watchdog. This is the exact window
    # in which the 2026-09-08 03:36:26 false stall fired.
    source = inspect.getsource(LiveTradingLoop._process_timeframe)
    assert source.count("_mark_loop_progress") == 2


def test_bootstrap_publisher_withholds_heartbeat_on_a_wedged_phase(monkeypatch):
    # A startup wedged on one await must stop heartbeating so the
    # supervisor still restarts it - the phase gate is what keeps the
    # thread-based publisher honest.
    bot = LiveTradingLoop()
    writes = []
    monkeypatch.setattr(
        bot._heartbeat,
        "write",
        lambda state, **details: writes.append((state, details)),
    )
    # Stop after the first pass. Must not pre-set the event: the publisher
    # checks it as the loop condition, so pre-setting would skip the body
    # entirely and make this assertion pass vacuously.
    monkeypatch.setattr(
        bot._bootstrap_stop, "wait", lambda _timeout: bot._bootstrap_stop.set()
    )

    bot._set_bootstrap_phase("exchange_connect")
    # Wedge it: phase last advanced longer ago than the allowed bound.
    bot._bootstrap_phase_at = (
        time.monotonic()
        - live_loop_module.settings.supervisor_bootstrap_phase_stall_seconds
        - 1
    )
    bot._publish_bootstrap_heartbeat()

    assert writes == []


def test_bootstrap_publisher_heartbeats_while_phases_advance(monkeypatch):
    bot = LiveTradingLoop()
    writes = []
    monkeypatch.setattr(
        bot._heartbeat,
        "write",
        lambda state, **details: writes.append((state, details)),
    )

    # Stop after the first pass (see note in the wedged-phase test above).
    monkeypatch.setattr(
        bot._bootstrap_stop, "wait", lambda _timeout: bot._bootstrap_stop.set()
    )

    bot._set_bootstrap_phase("exchange_connect")
    bot._publish_bootstrap_heartbeat()

    assert len(writes) == 1
    state, details = writes[0]
    # Must be a state the supervisor grants extended startup grace to.
    assert state == "bootstrapping"
    assert details["phase"] == "exchange_connect"
    # Never pass instance_id explicitly - RuntimeHeartbeat reads it from
    # the env the supervisor injected, and the supervisor matches on that.
    assert "instance_id" not in details


@pytest.mark.asyncio
async def test_stop_cancels_heartbeat_before_writing_stopped(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = LifecycleAlerter()
    states = []

    async def publisher():
        try:
            await asyncio.Event().wait()
        finally:
            states.append("publisher_cancelled")

    monkeypatch.setattr(
        bot,
        "_write_heartbeat",
        lambda state, **details: states.append(state),
    )
    bot._heartbeat_task = asyncio.create_task(publisher())
    await asyncio.sleep(0)

    await bot.stop("cancelled")

    assert states == ["publisher_cancelled", "stopping", "stopped"]


@pytest.mark.asyncio
async def test_telegram_commands_update_durable_trading_controls(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "telegram-controls.db"))
    bot.alerter = NoopAlerter()

    paused = await bot._handle_telegram_command("/pause", "maintenance", "42")
    allowed, reason = bot.audit_store.trading_allowed()
    assert not allowed
    assert "pause" in reason
    assert "Trading Paused" in paused

    resumed = await bot._handle_telegram_command("/resume", "", "42")
    assert bot.audit_store.trading_allowed() == (True, "ok")
    assert "Trading Resumed" in resumed

    await bot._handle_telegram_command("/emergency_stop", "incident", "42")
    blocked, reason = bot.audit_store.trading_allowed()
    assert not blocked
    assert "emergency" in reason

    confirmation = await bot._handle_telegram_command("/clear_emergency", "", "42")
    assert "Confirmation Required" in confirmation
    assert not bot.audit_store.trading_allowed()[0]

    cleared = await bot._handle_telegram_command("/clear_emergency", "CONFIRM", "42")
    assert bot.audit_store.trading_allowed() == (True, "ok")
    assert "Emergency Stop Cleared" in cleared


@pytest.mark.asyncio
async def test_telegram_status_and_help_commands(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "telegram-status.db"))
    bot.alerter = NoopAlerter()

    status = await bot._handle_telegram_command("/status", "", "42")
    help_text = await bot._handle_telegram_command("/help", "", "42")
    unknown = await bot._handle_telegram_command("/nope", "", "42")

    assert "Trading: ENABLED" in status
    assert "Open trades: 0" in status
    assert "/emergency_stop" in help_text
    assert "Unknown Command" in unknown


@pytest.mark.asyncio
async def test_telegram_command_replay_is_ignored(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "telegram-replay.db"))
    bot.alerter = NoopAlerter()

    first = await bot._handle_telegram_command(
        "/pause", "maintenance", "42", update_id=100
    )
    replay = await bot._handle_telegram_command(
        "/resume", "stale replay", "42", update_id=100
    )

    assert "Trading Paused" in first
    assert replay == ""
    assert not bot.audit_store.trading_allowed()[0]


@pytest.mark.asyncio
async def test_trade_account_refresh_degrades_without_stopping_after_three_attempts(
    monkeypatch,
):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()

    async def fail_account(_client):
        raise RuntimeError("temporary demo account outage")

    monkeypatch.setattr(live_loop_module, "get_account_info", fail_account)

    await bot._refresh_account()
    await bot._refresh_account()
    await bot._refresh_account()

    assert bot._account_refresh_failures == 3
    assert bot._account_degradation_alerted is True
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "account_refresh_degraded"


@pytest.mark.asyncio
async def test_account_refresh_recovery_clears_degraded_state(monkeypatch):
    bot = LiveTradingLoop()
    bot._account_refresh_failures = 3
    bot._account_degradation_alerted = True
    account = AccountInfo(
        total_equity=1234.0,
        wallet_balance=1200.0,
        available_balance=1000.0,
        unrealized_pnl=34.0,
        margin_ratio=0.1,
    )

    async def get_account(_client):
        return account

    monkeypatch.setattr(live_loop_module, "get_account_info", get_account)

    await bot._refresh_account()

    assert bot._account_refresh_failures == 0
    assert bot._account_degradation_alerted is False
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "account_refresh_recovered"


@pytest.mark.asyncio
async def test_account_refresh_updates_portfolio(monkeypatch):
    bot = LiveTradingLoop()
    account = AccountInfo(
        total_equity=1234.0,
        wallet_balance=1200.0,
        available_balance=1000.0,
        unrealized_pnl=34.0,
        margin_ratio=0.1,
    )

    async def get_account(_client):
        return account

    monkeypatch.setattr(live_loop_module, "get_account_info", get_account)

    await bot._refresh_account()

    assert bot._account_refresh_failures == 0
    assert bot.portfolio.account == account
    assert bot.audit_store.get_control("peak_equity") == "1234.0"


@pytest.mark.asyncio
async def test_account_refresh_if_due_is_throttled(monkeypatch):
    bot = LiveTradingLoop()
    calls = []

    async def refresh(required=False):
        calls.append(required)
        bot._last_account_refresh_at = 100.0

    monkeypatch.setattr(bot, "_refresh_account", refresh)
    monkeypatch.setattr(live_loop_module.time, "monotonic", lambda: 100.0)

    await bot._refresh_account_if_due()
    await bot._refresh_account_if_due()

    assert calls == [False]


@pytest.mark.asyncio
async def test_failed_account_refresh_blocks_new_entries(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "stale-account.db"))
    bot._account_refresh_failures = 1
    generated = []

    class Aggregator:
        def generate(self, _df):
            generated.append(True)

    bot.aggregator = Aggregator()
    await bot._execute_trade(
        {"symbol": "BTCUSDT", "timeframe": "5m", "close": 100.0},
        df_ind=pd.DataFrame({"close": [100.0]}),
    )

    assert generated == []
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "entry_blocked_account_stale"


@pytest.mark.asyncio
async def test_disabled_strategy_scope_skips_signal_generation(
    tmp_path,
    monkeypatch,
):
    from src.audit import AuditStore
    from src.live import loop as live_loop_module

    monkeypatch.setattr(
        live_loop_module.settings,
        "disabled_strategy_scopes",
        "BTCUSDT:5m",
    )
    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "scope-disabled.db"))
    generated = []

    class Aggregator:
        def generate(self, _df):
            generated.append(True)

    bot.aggregator = Aggregator()
    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "close": 100.0,
        },
        df_ind=pd.DataFrame({"close": [100.0]}),
    )

    assert generated == []
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "strategy_scope_disabled"


@pytest.mark.asyncio
async def test_strategy_specific_disabled_scope_does_not_block_other_strategies(
    tmp_path,
    monkeypatch,
):
    from src.audit import AuditStore
    from src.live import loop as live_loop_module

    monkeypatch.setattr(
        live_loop_module.settings,
        "disabled_strategy_scopes",
        "BTCUSDT:30m:breakout",
    )
    audit = AuditStore(tmp_path / "strategy-specific-disabled.db")
    bot = LiveTradingLoop(audit_store=audit)

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=1,
                    confidence=0.90,
                    ta_source="donchian_breakout_bull",
                    strategy="breakout",
                ),
                FinalSignal(
                    direction=1,
                    confidence=0.80,
                    ta_source="trend_structure_bull",
                    strategy="trend",
                ),
            ]

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(True, 0.90, "confirmed", {})
    )
    bot.pos_mgr.enter_long = AsyncMock(return_value=True)

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "30m",
            "close": 100.0,
            "timestamp": pd.Timestamp("2026-06-20T10:00:00Z"),
        },
        df_ind=pd.DataFrame(
            {
                "close": [100.0],
                "atr": [2.0],
                "ema_50": [98.0],
                "ema_200": [95.0],
                "ema_50_slope": [0.01],
                "plus_di": [28.0],
                "minus_di": [12.0],
                "adx": [30.0],
                "bb_upper": [110.0],
                "bb_lower": [90.0],
                "vol_ratio": [1.2],
            }
        ),
    )

    bot.pos_mgr.enter_long.assert_awaited_once()
    assert bot.pos_mgr.enter_long.await_args.kwargs["strategy"] == "trend"
    observations = audit.load_signal_observations(10)
    decisions = {row["strategy"]: row["decision"] for row in observations}
    assert decisions["breakout"] == "disabled"
    assert decisions["trend"] == "accepted"


def _regime_mismatch_bot(tmp_path, *, db_name: str) -> LiveTradingLoop:
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / db_name)
    bot = LiveTradingLoop(audit_store=audit)

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=1,
                    confidence=0.85,
                    ta_source="bb_lower_bounce",
                    strategy="range",
                )
            ]

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(True, 0.90, "range confirmed", {})
    )
    bot.pos_mgr.enter_long = AsyncMock(return_value=True)
    return bot


REGIME_MISMATCH_DF_IND = pd.DataFrame(
    {
        "close": [100.0],
        "atr": [2.0],
        "ema_50": [98.0],
        "ema_200": [95.0],
        "ema_50_slope": [0.01],
        "plus_di": [28.0],
        "minus_di": [12.0],
        "adx": [30.0],
        "bb_upper": [110.0],
        "bb_lower": [90.0],
        "vol_ratio": [1.2],
    }
)


@pytest.mark.asyncio
async def test_mismatched_regime_signal_is_rejected_when_regime_filter_enforced(
    tmp_path, monkeypatch
):
    # A "range" signal fires while the regime detector reads a strong,
    # clearly-directional trend (adx=30, close>ema_50>ema_200) - range is
    # not in regime_appropriate_strategies() for a trending market. With
    # regime_filter_enforced on, this must be rejected before quality-gate
    # evaluation or entry, not just logged and let through. (Defaults to
    # off pending backtest validation for scalp/breakout/reversal/
    # transition - see config.py - so this test sets it explicitly.)
    monkeypatch.setattr(live_loop_module.settings, "regime_filter_enforced", True)
    bot = _regime_mismatch_bot(tmp_path, db_name="regime-enforced.db")

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "30m",
            "close": 100.0,
            "timestamp": pd.Timestamp("2026-06-20T11:00:00Z"),
        },
        df_ind=REGIME_MISMATCH_DF_IND,
    )

    bot.strategy_quality.evaluate.assert_not_called()
    bot.pos_mgr.enter_long.assert_not_awaited()


@pytest.mark.asyncio
async def test_mismatched_regime_signal_is_advisory_when_filter_disabled(
    tmp_path, monkeypatch
):
    # Same setup as above, but with the enforcement toggle off: this
    # preserves the old advisory-only behavior (logged, not blocked) for
    # anyone who needs to revert it.
    monkeypatch.setattr(live_loop_module.settings, "regime_filter_enforced", False)
    bot = _regime_mismatch_bot(tmp_path, db_name="regime-advisory.db")

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "30m",
            "close": 100.0,
            "timestamp": pd.Timestamp("2026-06-20T11:00:00Z"),
        },
        df_ind=REGIME_MISMATCH_DF_IND,
    )

    bot.strategy_quality.evaluate.assert_called_once()
    bot.pos_mgr.enter_long.assert_awaited_once()
    assert bot.pos_mgr.enter_long.await_args.kwargs["strategy"] == "range"


@pytest.mark.asyncio
async def test_source_gate_hard_block_skips_quality_and_entry(tmp_path, monkeypatch):
    from src.audit import AuditStore

    monkeypatch.setattr(live_loop_module.settings, "signal_source_gate_enabled", True)
    audit = AuditStore(tmp_path / "source-gate-block.db")
    bot = LiveTradingLoop(audit_store=audit)

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=-1,
                    confidence=0.90,
                    ta_source="trend_structure_bear+structure_score_0.90",
                    strategy="trend",
                )
            ]

    bot.aggregator = Aggregator()
    bot.signal_gate.evaluate = MagicMock(
        return_value=SignalGateResult(
            passed=False,
            confidence_multiplier=0.70,
            win_rate=0.25,
            total_samples=8,
            avg_directional_bps=-6.0,
            reason="win rate and edge below threshold",
        )
    )
    bot.strategy_quality.evaluate = MagicMock()
    bot.pos_mgr.enter_short = AsyncMock(return_value=True)

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "30m",
            "close": 100.0,
            "timestamp": pd.Timestamp("2026-06-20T11:30:00Z"),
        },
        df_ind=pd.DataFrame(
            {
                "close": [100.0],
                "atr": [2.0],
                "ema_50": [102.0],
                "ema_200": [105.0],
                "ema_50_slope": [-0.01],
                "plus_di": [12.0],
                "minus_di": [28.0],
                "adx": [30.0],
                "bb_upper": [110.0],
                "bb_lower": [90.0],
                "vol_ratio": [1.2],
            }
        ),
    )

    bot.strategy_quality.evaluate.assert_not_called()
    bot.pos_mgr.enter_short.assert_not_awaited()
    observations = audit.load_signal_observations(10)
    assert observations[0]["decision"] == "source_gate_rejected"


@pytest.mark.asyncio
async def test_accepted_opposite_signal_closes_existing_position_before_entry(
    tmp_path,
):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "reverse-opposite.db")
    bot = LiveTradingLoop(audit_store=audit)
    bot.pos_mgr.open_trades["BTCUSDT:old:trend"] = TradeRecord(
        symbol="BTCUSDT",
        side="short",
        entry_price=101.0,
        quantity=0.1,
        timestamp=datetime.now(timezone.utc),
        timeframe="1h",
        strategy="trend",
    )

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=1,
                    confidence=0.90,
                    ta_source="trend_structure_bull",
                    strategy="trend",
                )
            ]

    async def exit_position(position_key, _reason):
        bot.pos_mgr.open_trades.pop(position_key, None)

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(True, 0.90, "trend confirmed", {})
    )
    bot.pos_mgr.exit_position = AsyncMock(side_effect=exit_position)
    bot.pos_mgr.enter_long = AsyncMock(return_value=True)

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "close": 100.0,
            "timestamp": pd.Timestamp("2026-06-20T12:00:00Z"),
        },
        df_ind=pd.DataFrame(
            {
                "close": [100.0],
                "atr": [2.0],
                "ema_50": [98.0],
                "ema_200": [95.0],
                "ema_50_slope": [0.01],
                "plus_di": [28.0],
                "minus_di": [12.0],
                "adx": [30.0],
                "bb_upper": [110.0],
                "bb_lower": [90.0],
                "vol_ratio": [1.2],
            }
        ),
    )

    bot.pos_mgr.exit_position.assert_awaited_once()
    bot.pos_mgr.enter_long.assert_awaited_once()
    assert bot.pos_mgr.enter_long.await_args.kwargs["ignore_reentry_cooldown"] is True
    assert "BTCUSDT:old:trend" not in bot.pos_mgr.open_trades


@pytest.mark.asyncio
async def test_opposite_reversal_close_is_serialized_per_symbol(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "reverse-lock.db"))
    bot.pos_mgr.open_trades["BTCUSDT:4h:trend"] = TradeRecord(
        symbol="BTCUSDT",
        side="short",
        entry_price=101.0,
        quantity=0.1,
        timestamp=datetime.now(timezone.utc),
        timeframe="4h",
        strategy="trend",
    )
    signal = FinalSignal(
        direction=1,
        confidence=0.90,
        ta_source="trend_structure_bull",
        strategy="trend",
    )

    async def exit_position(position_key, _reason):
        await asyncio.sleep(0)
        bot.pos_mgr.open_trades.pop(position_key, None)

    bot.pos_mgr.exit_position = AsyncMock(side_effect=exit_position)

    results = await asyncio.gather(
        bot._close_opposite_symbol_trades_if_needed(
            "BTCUSDT",
            signal,
            "trend",
            "BTCUSDT:5m",
        ),
        bot._close_opposite_symbol_trades_if_needed(
            "BTCUSDT",
            signal,
            "trend",
            "BTCUSDT:15m",
        ),
    )

    assert bot.pos_mgr.exit_position.await_count == 1
    assert results == [(True, True), (True, False)]


@pytest.mark.asyncio
async def test_reversal_clears_stale_risk_state_for_reused_position_key(tmp_path):
    # The old and new side of a same-key reversal share one position_key
    # with no gap where it's absent from open_trades, so the periodic
    # _cleanup_scalp_state() sweep never sees the transition. Without an
    # explicit clear here, a fresh reversed trade could inherit the
    # previous trade's stale initial_risk/peak price.
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "reverse-state.db"))
    position_key = "BTCUSDT:4h:trend"
    bot.pos_mgr.open_trades[position_key] = TradeRecord(
        symbol="BTCUSDT",
        side="short",
        entry_price=101.0,
        quantity=0.1,
        timestamp=datetime.now(timezone.utc),
        timeframe="4h",
        strategy="trend",
    )
    bot._swing_initial_risk[position_key] = 2.0
    bot._swing_peak_prices[position_key] = 99.0

    signal = FinalSignal(
        direction=1,
        confidence=0.90,
        ta_source="trend_structure_bull",
        strategy="trend",
    )

    async def exit_position(pk, _reason):
        bot.pos_mgr.open_trades.pop(pk, None)

    bot.pos_mgr.exit_position = AsyncMock(side_effect=exit_position)

    await bot._close_opposite_symbol_trades_if_needed(
        "BTCUSDT", signal, "trend", "BTCUSDT:4h"
    )

    assert position_key not in bot._swing_initial_risk
    assert position_key not in bot._swing_peak_prices


@pytest.mark.asyncio
async def test_lower_timeframe_signal_rejects_higher_timeframe_reversal_without_quality(
    tmp_path,
):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "lower-timeframe-reversal-deferred.db")
    bot = LiveTradingLoop(audit_store=audit)
    bot.pos_mgr.open_trades["BTCUSDT:1d:countertrend"] = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=0.1,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
        timeframe="1d",
        strategy="countertrend",
    )

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=-1,
                    confidence=0.92,
                    ta_source="scalp_momentum_bear",
                    strategy="scalp",
                )
            ]

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(False, 0.20, "weak bearish quality", {})
    )
    bot.pos_mgr.exit_position = AsyncMock()
    bot.pos_mgr.enter_short = AsyncMock(return_value=True)

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "close": 99.0,
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "market_data_source": "websocket",
        },
        df_ind=pd.DataFrame({"close": [99.0], "atr": [1.0]}),
    )

    bot.pos_mgr.exit_position.assert_not_awaited()
    bot.pos_mgr.enter_short.assert_not_awaited()
    observation = audit.load_signal_observations(1)[0]
    assert observation["decision"] == "quality_rejected"
    assert observation["reason"] == "weak bearish quality"


@pytest.mark.asyncio
async def test_lower_timeframe_signal_cannot_reverse_distant_higher_timeframe_position(
    tmp_path,
):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "lower-timeframe-reversal-confirmed.db")
    bot = LiveTradingLoop(audit_store=audit)
    bot.pos_mgr.open_trades["BTCUSDT:1d:countertrend"] = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=0.1,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
        timeframe="1d",
        strategy="countertrend",
    )

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=-1,
                    confidence=0.92,
                    ta_source="scalp_momentum_bear",
                    strategy="scalp",
                )
            ]

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(True, 0.80, "bearish reversal confirmed", {})
    )
    bot.pos_mgr.exit_position = AsyncMock()
    bot.pos_mgr.enter_short = AsyncMock(return_value=True)

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "close": 99.0,
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "market_data_source": "websocket",
        },
        df_ind=pd.DataFrame({"close": [99.0], "atr": [1.0]}),
    )

    bot.pos_mgr.exit_position.assert_not_awaited()
    bot.pos_mgr.enter_short.assert_not_awaited()
    observation = audit.load_signal_observations(1)[0]
    assert observation["decision"] == "reversal_deferred"
    assert "timeframe ratio" in observation["reason"]


def test_fifteen_minute_signal_cannot_reverse_four_hour_position(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "15m-vs-4h.db"))
    bot.pos_mgr.open_trades["BTCUSDT:4h:trend"] = TradeRecord(
        symbol="BTCUSDT",
        side="short",
        entry_price=59_500.0,
        quantity=0.001,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=8),
        timeframe="4h",
        strategy="trend",
    )

    allowed, reason = bot._same_symbol_reversal_allowed(
        symbol="BTCUSDT",
        timeframe="15m",
        signal=FinalSignal(
            direction=1,
            confidence=0.95,
            ta_source="trend_structure_bull",
            strategy="trend",
        ),
        quality=StrategyQuality(True, 0.95, "trend confirmed", {}),
    )

    assert allowed is False
    assert "timeframe ratio" in reason


def test_reversal_guard_requires_accepted_quality_even_when_gate_is_advisory(
    tmp_path,
    monkeypatch,
):
    from src.audit import AuditStore

    monkeypatch.setattr(
        live_loop_module.settings,
        "strategy_quality_gate_enforced",
        False,
    )
    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "reversal-quality.db"))
    bot.pos_mgr.open_trades["BTCUSDT:5m:trend"] = TradeRecord(
        symbol="BTCUSDT",
        side="short",
        entry_price=100.0,
        quantity=0.1,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
        timeframe="5m",
        strategy="trend",
    )

    allowed, reason = bot._same_symbol_reversal_allowed(
        symbol="BTCUSDT",
        timeframe="5m",
        signal=FinalSignal(
            direction=1,
            confidence=0.95,
            ta_source="countertrend",
            strategy="countertrend",
        ),
        quality=StrategyQuality(False, 0.0, "countertrend quality below threshold", {}),
    )

    assert allowed is False
    assert "requires accepted strategy quality" in reason


@pytest.mark.asyncio
async def test_reversal_does_not_close_position_when_target_entry_policy_is_blocked(
    tmp_path,
):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "reversal-entry-policy.db")
    bot = LiveTradingLoop(audit_store=audit)
    bot.pos_mgr.open_trades["BTCUSDT:15m:reversal"] = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=59_400.0,
        quantity=0.001,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=20),
        timeframe="15m",
        strategy="reversal",
    )
    bot.portfolio.strategy_consecutive_losses["trend"] = 10
    bot.portfolio.strategy_last_loss_at["trend"] = datetime.now(timezone.utc)

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=-1,
                    confidence=0.92,
                    ta_source="trend_structure_bear",
                    strategy="trend",
                )
            ]

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(True, 0.90, "trend confirmed", {})
    )
    bot.pos_mgr.exit_position = AsyncMock()
    bot.pos_mgr.enter_short = AsyncMock(return_value=True)

    await bot._execute_trade(
        {
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "close": 59_458.80,
            "timestamp": pd.Timestamp("2026-06-28T20:26:10Z"),
        },
        df_ind=pd.DataFrame({"close": [59_458.80], "atr": [300.0]}),
    )

    bot.pos_mgr.exit_position.assert_not_awaited()
    bot.pos_mgr.enter_short.assert_not_awaited()
    observation = audit.load_signal_observations(1)[0]
    assert observation["decision"] == "risk_rejected"
    assert "entry policy blocks reversal" in observation["reason"]


@pytest.mark.asyncio
async def test_accepted_signal_records_decision_context(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "signal-context.db")
    bot = LiveTradingLoop(audit_store=audit)

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=1,
                    confidence=0.72,
                    ta_source="ema_fibonacci",
                    decision_reason="aligned signal ready",
                    strategy="breakout",
                )
            ]

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(
            accepted=True,
            score=0.85,
            reason="confirmed",
            metrics={"adx": 30.0},
        )
    )
    bot.pos_mgr.enter_long = AsyncMock(return_value=True)
    timestamp = pd.Timestamp("2026-06-13T10:00:00Z")

    await bot._execute_trade(
        {
            "symbol": "ETHUSDT",
            "timeframe": "30m",
            "close": 1700.0,
            "timestamp": timestamp,
        },
        df_ind=pd.DataFrame({"close": [1700.0], "atr": [12.0]}),
    )

    event = audit.load_recent_events(1)[0]
    assert event["event_type"] == "signal_accepted"
    assert event["payload"]["scope"] == "ETHUSDT:30m"
    assert event["payload"]["confidence"] == 0.72
    assert event["payload"]["strategy"] == "breakout"
    assert event["payload"]["atr"] == 12.0
    assert event["payload"]["quality_score"] == 0.85
    assert bot.pos_mgr.enter_long.await_args.kwargs["strategy"] == "breakout"
    observation = audit.load_signal_observations(1)[0]
    assert observation["decision"] == "accepted"
    assert observation["execution_status"] == "opened"
    assert observation["strategy"] == "breakout"
    assert observation["quality_score"] == 0.85


@pytest.mark.asyncio
async def test_same_direction_limit_is_recorded_as_risk_rejected(tmp_path, monkeypatch):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "same-direction-limit.db")
    bot = LiveTradingLoop(audit_store=audit)
    monkeypatch.setattr(live_loop_module.settings, "max_same_direction_positions", 1)

    bot.pos_mgr.open_trades["BTCUSDT:old:trend"] = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(timezone.utc),
        timeframe="1h",
        strategy="trend",
    )

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=1,
                    confidence=0.90,
                    ta_source="trend_structure_bull+structure_score_1.00",
                    decision_reason="aligned signal ready",
                    strategy="trend",
                )
            ]

    bot.aggregator = Aggregator()
    bot.strategy_quality.evaluate = MagicMock(
        return_value=StrategyQuality(
            accepted=True,
            score=0.90,
            reason="confirmed",
            metrics={"adx": 30.0},
        )
    )
    bot.pos_mgr.enter_long = AsyncMock(return_value=True)

    await bot._execute_trade(
        {
            "symbol": "ETHUSDT",
            "timeframe": "1h",
            "close": 1700.0,
            "timestamp": pd.Timestamp("2026-06-13T10:00:00Z"),
        },
        df_ind=pd.DataFrame(
            {
                "close": [1700.0],
                "atr": [12.0],
                "ema_50": [1680.0],
                "ema_200": [1650.0],
                "ema_50_slope": [0.01],
                "plus_di": [30.0],
                "minus_di": [12.0],
                "trend_regime": [1],
                "adx": [30.0],
            }
        ),
    )

    bot.pos_mgr.enter_long.assert_not_awaited()
    observation = audit.load_signal_observations(1)[0]
    assert observation["decision"] == "risk_rejected"
    assert "same-direction limit reached" in observation["reason"]
    assert all(
        event["event_type"] != "signal_accepted"
        for event in audit.load_recent_events(10)
    )


@pytest.mark.asyncio
async def test_skipped_signal_is_recorded_for_forward_analysis(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "skipped-signal.db")
    bot = LiveTradingLoop(audit_store=audit)

    class Aggregator:
        def generate(self, _df):
            return [
                FinalSignal(
                    direction=-1,
                    confidence=0.10,
                    ta_source="ema_fibonacci",
                    decision_reason="confidence below threshold",
                    strategy="trend",
                )
            ]

    bot.aggregator = Aggregator()
    await bot._execute_trade(
        {
            "symbol": "BNBUSDT",
            "timeframe": "1h",
            "close": 600.0,
            "timestamp": pd.Timestamp("2026-06-15T10:00:00Z"),
        },
        df_ind=pd.DataFrame({"close": [600.0]}),
    )

    observation = audit.load_signal_observations(1)[0]
    assert observation["decision"] == "skipped"
    assert observation["direction"] == -1
    assert observation["execution_status"] == "not_attempted"
    assert observation["reason"] == "confidence below threshold"


def test_strategy_specific_confidence_thresholds(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "min_confidence", 0.45)
    monkeypatch.setattr(
        live_loop_module.settings,
        "strategy_breakout_min_confidence",
        0.55,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "strategy_scalp_min_confidence",
        0.60,
    )

    assert LiveTradingLoop._strategy_minimum_confidence("trend") == 0.45
    assert LiveTradingLoop._strategy_minimum_confidence("breakout") == 0.55
    assert LiveTradingLoop._strategy_minimum_confidence("scalp") == 0.60


def test_live_loop_restores_persisted_peak_equity(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "risk-state.db")
    audit.set_control("peak_equity", "6000.0", "test")
    bot = LiveTradingLoop(audit_store=audit)

    bot._restore_persisted_risk_state()

    assert bot.portfolio.peak_equity == 6000.0


class FakeSequentialClient:
    def __init__(self):
        self.calls = []
        self.last_index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"), periods=201, freq="5min"
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=200):
        self.calls.append((symbol, timeframe, limit))
        index = self.last_index
        return pd.DataFrame(
            {
                "open": [100.0 + i * 0.1 for i in range(len(index))],
                "high": [101.0 + i * 0.1 for i in range(len(index))],
                "low": [99.0 + i * 0.1 for i in range(len(index))],
                "close": [100.5 + i * 0.1 for i in range(len(index))],
                "volume": [10.0 + i for i in range(len(index))],
            },
            index=index,
        )


def test_due_scan_pairs_only_returns_due_items(monkeypatch):
    bot = LiveTradingLoop()
    monkeypatch.setattr(live_loop_module.settings, "symbols", "BTCUSDT")
    monkeypatch.setattr(live_loop_module.settings, "timeframes", "5m,15m")
    bot._next_scan_due["BTCUSDT_15m"] = 999.0

    due = bot._due_scan_pairs(100.0)

    assert due == [("BTCUSDT", "5m")]


def test_timeframe_seconds_parses_supported_units():
    assert LiveTradingLoop._timeframe_seconds("5m") == 300
    assert LiveTradingLoop._timeframe_seconds("1h") == 3600
    assert LiveTradingLoop._timeframe_seconds("1d") == 86_400


def test_next_scan_aligns_to_exchange_candle_boundary(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "candle_close_grace_seconds",
        2.0,
    )
    monkeypatch.setattr(live_loop_module.settings, "scan_sleep_seconds", 5.0)

    due = LiveTradingLoop._next_candle_scan_due(
        "15m",
        monotonic_now=100.0,
        epoch_now=12 * 3600 + 7 * 60,
    )

    assert due == 100.0 + 8 * 60 + 2.0


def test_rest_scalp_scan_waits_for_exchange_publication(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "candle_close_grace_seconds", 2.0)
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_rest_candle_close_grace_seconds",
        15.0,
    )
    monkeypatch.setattr(live_loop_module.settings, "scalp_websocket_enabled", False)
    monkeypatch.setattr(live_loop_module.settings, "scan_sleep_seconds", 5.0)

    due = LiveTradingLoop._next_candle_scan_due(
        "1m",
        monotonic_now=100.0,
        epoch_now=12 * 3600 + 30.0,
    )

    assert due == 100.0 + 30.0 + 15.0


def test_streamed_scalp_scan_keeps_standard_close_grace(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "binance_api_url",
        "https://fapi.binance.com",
    )
    monkeypatch.setattr(live_loop_module.settings, "candle_close_grace_seconds", 2.0)
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_rest_candle_close_grace_seconds",
        15.0,
    )
    monkeypatch.setattr(live_loop_module.settings, "scalp_websocket_enabled", True)
    monkeypatch.setattr(live_loop_module.settings, "scan_sleep_seconds", 5.0)

    due = LiveTradingLoop._next_candle_scan_due(
        "1m",
        monotonic_now=100.0,
        epoch_now=12 * 3600 + 30.0,
    )

    assert due == 100.0 + 30.0 + 2.0


@pytest.mark.asyncio
async def test_scan_timeframes_runs_in_configured_sequence(monkeypatch):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    fake_client = FakeSequentialClient()
    bot.client = fake_client
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append((candle["symbol"], candle["timeframe"]))

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    await bot._scan_timeframes_once(
        symbols=["BTCUSDT"],
        timeframes=["5m", "15m", "30m"],
    )

    # Fetches are dispatched in the configured sequence.
    assert fake_client.calls == [
        ("BTCUSDT", "5m", 201),
        ("BTCUSDT", "15m", 201),
        ("BTCUSDT", "30m", 201),
    ]
    # Every pair is processed exactly once. Deliberately order-insensitive:
    # _scan_timeframes_once gathers pairs behind a semaphore of
    # market_data_concurrency (>1), so completion order follows whichever
    # fetch resolves first, never the configured order. This assertion used
    # to be an ordered one, which only held because FakeSequentialClient
    # returns without awaiting - so each task ran fetch -> compute ->
    # _on_candle without ever yielding. Real network latency broke that
    # long before indicator computation moved to a worker thread; the
    # ordering was an artifact of the fake, not an invariant of the code.
    assert sorted(processed) == sorted(
        [
            ("BTCUSDT", "5m"),
            ("BTCUSDT", "15m"),
            ("BTCUSDT", "30m"),
        ]
    )


@pytest.mark.asyncio
async def test_scan_timeframes_skips_duplicate_candle(monkeypatch):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    bot.client = FakeSequentialClient()
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append((candle["symbol"], candle["timeframe"]))

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    await bot._scan_timeframes_once(symbols=["BTCUSDT"], timeframes=["5m"])
    await bot._scan_timeframes_once(symbols=["BTCUSDT"], timeframes=["5m"])

    assert processed == [("BTCUSDT", "5m")]


@pytest.mark.asyncio
async def test_process_timeframe_uses_last_closed_candle(monkeypatch):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    bot.client = FakeSequentialClient()
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append(candle)

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    await bot._process_timeframe("BTCUSDT", "5m")

    assert processed[0]["timestamp"] == bot.client.last_index[-2]
    assert processed[0]["close"] == 100.5 + 199 * 0.1


@pytest.mark.asyncio
async def test_process_timeframe_rejects_bad_data(monkeypatch):
    monkeypatch.setattr(live_loop_module.settings, "allow_zero_volume_candles", False)
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    bot.client = FakeSequentialClient()
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append(candle)

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    async def bad_fetch(symbol, timeframe, limit=200):
        df = await FakeSequentialClient().fetch_ohlcv(symbol, timeframe, limit)
        df.loc[df.index[-2], "volume"] = 0.0
        return df

    monkeypatch.setattr(bot.client, "fetch_ohlcv", bad_fetch)

    await bot._process_timeframe("BTCUSDT", "5m")

    assert processed == []
    events = bot.audit_store.load_recent_events(5)
    assert any(event["event_type"] == "data_quality_rejected" for event in events)


@pytest.mark.asyncio
async def test_process_timeframe_reports_market_data_failure(monkeypatch):
    bot = LiveTradingLoop()
    bot.client.fetch_ohlcv = AsyncMock(side_effect=RuntimeError("feed unavailable"))
    bot.alerter.data_feed_alert = AsyncMock()
    bot.alerter.trade_failed_alert = AsyncMock()

    await bot._process_timeframe("BTCUSDT", "5m")

    bot.alerter.data_feed_alert.assert_awaited_once_with(
        "trade",
        "BTCUSDT",
        "5m",
        "RuntimeError: feed unavailable",
    )
    bot.alerter.trade_failed_alert.assert_not_awaited()
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "market_data_scan_failed"
    assert event["payload"]["timeframe"] == "5m"


@pytest.mark.asyncio
async def test_network_outage_circuit_breaker_suppresses_feed_alerts(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "network_outage_failure_threshold",
        1,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "network_outage_cooldown_seconds",
        60.0,
    )
    bot = LiveTradingLoop()
    bot.client.fetch_ohlcv = AsyncMock(side_effect=RuntimeError("dns unavailable"))
    bot.alerter.data_feed_alert = AsyncMock()
    bot.alerter.error_alert = AsyncMock()

    await bot._process_timeframe("BTCUSDT", "1m")
    await bot._process_timeframe("ETHUSDT", "1m")

    assert bot._network_outage_active()
    bot.alerter.error_alert.assert_awaited_once()
    bot.alerter.data_feed_alert.assert_not_awaited()
    events = bot.audit_store.load_recent_events(10)
    assert any(event["event_type"] == "network_outage_detected" for event in events)


@pytest.mark.asyncio
async def test_network_outage_blocks_new_entries(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "network_outage_cooldown_seconds",
        60.0,
    )
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    bot._connectivity_outage_until = time.monotonic() + 60.0
    called = False

    async def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(bot.aggregator, "generate", fail_if_called)

    await bot._execute_trade(
        {"symbol": "BTCUSDT", "timeframe": "1m", "close": 100.0},
        df_ind=pd.DataFrame({"close": [100.0]}),
    )

    assert called is False


def test_scalp_cost_floor_delays_partial_until_fees_and_edge_are_covered(
    monkeypatch,
):
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_break_even_offset_bps",
        10.0,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_estimated_round_trip_fee_bps",
        8.0,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_min_net_edge_bps",
        3.0,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "binance_api_url",
        "https://demo-fapi.binance.com",
    )
    trade = TradeRecord(
        symbol="BTCUSDT",
        side="long",
        entry_price=100.0,
        quantity=1.0,
        timestamp=datetime.now(timezone.utc),
        timeframe="1m",
        strategy="scalp",
    )

    trigger_r = LiveTradingLoop._scalp_cost_floor_r(
        trade,
        initial_risk=0.15,
    )

    assert trigger_r == pytest.approx(19.0 / 15.0)


@pytest.mark.asyncio
async def test_stale_rest_scalp_candle_is_refetched(monkeypatch):
    bot = LiveTradingLoop()
    now = pd.Timestamp.now(tz="UTC").floor("min")
    stale = pd.DataFrame(
        {"close": [100.0, 100.0, 100.0]},
        index=[now - pd.Timedelta(minutes=4), now - pd.Timedelta(minutes=3), now],
    )
    fresh = pd.DataFrame(
        {"close": [100.0, 100.0, 100.0]},
        index=[now - pd.Timedelta(minutes=1), now, now + pd.Timedelta(minutes=1)],
    )
    bot.client.fetch_ohlcv = AsyncMock(side_effect=[stale, fresh])
    sleep = AsyncMock()
    monkeypatch.setattr(live_loop_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_rest_freshness_attempts",
        3,
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_rest_freshness_retry_seconds",
        5.0,
    )

    result = await bot._fetch_rest_market_frame("BNBUSDT", "1m")

    assert result is fresh
    assert bot.client.fetch_ohlcv.await_count == 2
    sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_current_expected_rest_candle_is_not_refetched_for_age_alone(
    monkeypatch,
):
    bot = LiveTradingLoop()
    now = pd.Timestamp("2026-06-21 21:04:29", tz="UTC")
    frame = pd.DataFrame(
        {"close": [100.0, 100.0, 100.0]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-06-21 20:57:00", tz="UTC"),
                pd.Timestamp("2026-06-21 21:00:00", tz="UTC"),
                pd.Timestamp("2026-06-21 21:03:00", tz="UTC"),
            ]
        ),
    )
    bot.client.fetch_ohlcv = AsyncMock(return_value=frame)
    monkeypatch.setattr(
        bot,
        "_latest_expected_closed_candle_timestamp",
        lambda timeframe: LiveTradingLoop._latest_expected_closed_candle_timestamp(
            timeframe, now=now
        ),
    )

    result = await bot._fetch_rest_market_frame("BTCUSDT", "3m")

    assert result is frame
    bot.client.fetch_ohlcv.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_rest_scalp_refetch_is_bounded(monkeypatch):
    bot = LiveTradingLoop()
    now = pd.Timestamp.now(tz="UTC").floor("min")
    stale = pd.DataFrame(
        {"close": [100.0, 100.0, 100.0]},
        index=[
            now - pd.Timedelta(minutes=4),
            now - pd.Timedelta(minutes=3),
            now - pd.Timedelta(minutes=2),
        ],
    )
    bot.client.fetch_ohlcv = AsyncMock(return_value=stale)
    sleep = AsyncMock()
    monkeypatch.setattr(live_loop_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(
        bot,
        "_latest_expected_closed_candle_timestamp",
        lambda timeframe: now - pd.Timedelta(minutes=1),
    )
    monkeypatch.setattr(
        live_loop_module.settings,
        "scalp_rest_freshness_attempts",
        3,
    )

    result = await bot._fetch_rest_market_frame("BNBUSDT", "1m")

    assert result is stale
    assert bot.client.fetch_ohlcv.await_count == 3
    assert sleep.await_count == 2
