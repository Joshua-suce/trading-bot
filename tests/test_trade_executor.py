from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import settings
from src.execution.trade_executor import TradeExecutor
from src.risk.portfolio import TradeRecord


@pytest.fixture
def mock_deps():
    client = MagicMock()
    client.fetch_ticker = AsyncMock(
        return_value={
            "bid": 100.0,
            "ask": 100.0,
            "last": 100.0,
            "mark": 100.0,
        }
    )
    client.fetch_order_book = AsyncMock(
        return_value={
            "bids": [[99.99, 1.0]],
            "asks": [[100.01, 1.0]],
        }
    )
    client.fetch_funding_rate = AsyncMock(return_value=0.0)

    orders = MagicMock()
    orders.market_order = AsyncMock(
        return_value={"id": "m1", "filled": 1.0, "average": 100.0}
    )
    orders.cancel_order = AsyncMock(return_value=None)

    fill_resolver = MagicMock()
    fill_resolver.resolve_entry_fill_price = AsyncMock(return_value=100.0)
    fill_resolver.confirmed_full_fill = AsyncMock(
        return_value={"id": "m1", "filled": 1.0, "average": 100.0}
    )
    fill_resolver.resolve_exit_price = AsyncMock(return_value=101.0)
    fill_resolver.order_fee = AsyncMock(return_value=0.0)
    fill_resolver.entry_fill_slippage_bps = MagicMock(return_value=0.0)

    sl_manager = MagicMock()

    def risk_levels(entry_price, side, atr, strategy=None):
        if side == "long":
            return MagicMock(stop_loss=98.0, take_profit=104.0)
        return MagicMock(stop_loss=102.0, take_profit=96.0)

    sl_manager.calculate.side_effect = risk_levels

    sizer = MagicMock()
    sizer.calculate.return_value = MagicMock(quantity=1.0)

    portfolio = MagicMock()
    portfolio.can_trade = MagicMock(return_value=(True, "ok"))
    portfolio.account = MagicMock(total_equity=5000.0)
    portfolio.loss_cooldown_seconds = 0
    portfolio.last_symbol_loss_at = {}

    alerter = MagicMock()
    alerter.trade_opened_alert = AsyncMock()
    alerter.trade_completed_alert = AsyncMock()
    alerter.trade_failed_alert = AsyncMock()

    audit_store = MagicMock()
    audit_store.trading_allowed.return_value = (True, "ok")
    audit_store.safe_record_event = MagicMock()

    trades = MagicMock()
    trades.position_key.return_value = "BTCUSDT-5m"
    trades.open_trades = {}
    trades.trade_correlation_ids = {}
    trades.last_symbol_exit_at = {}
    trades.policy_alert_due = MagicMock(return_value=False)
    trades.reentry_cooldown_reason = MagicMock(return_value="")
    trades.normalize_symbol.side_effect = lambda s: s.replace("/", "")
    trades.trades_for_symbol = MagicMock(return_value=[])

    protection = MagicMock()
    protection.place_entry_protection = AsyncMock(
        return_value=({"id": "sl-1"}, {"id": "tp-1"})
    )
    protection.cancel_protection = AsyncMock()
    protection.active_stops = {}
    protection.active_tps = {}

    check_exposure_limits = MagicMock(return_value=(True, "ok"))
    fail_reconciliation = AsyncMock()
    finalize_trade_leg = AsyncMock()

    executor = TradeExecutor(
        client=client,
        orders=orders,
        fill_resolver=fill_resolver,
        sl_manager=sl_manager,
        sizer=sizer,
        portfolio=portfolio,
        alerter=alerter,
        audit_store=audit_store,
        mode="trade",
        trades=trades,
        protection=protection,
        check_exposure_limits=check_exposure_limits,
        fail_reconciliation=fail_reconciliation,
        finalize_trade_leg=finalize_trade_leg,
    )
    return {
        "executor": executor,
        "client": client,
        "orders": orders,
        "fill_resolver": fill_resolver,
        "sl_manager": sl_manager,
        "sizer": sizer,
        "portfolio": portfolio,
        "alerter": alerter,
        "audit_store": audit_store,
        "trades": trades,
        "protection": protection,
        "check_exposure_limits": check_exposure_limits,
        "fail_reconciliation": fail_reconciliation,
        "finalize_trade_leg": finalize_trade_leg,
    }


class TestEnterPosition:
    @pytest.mark.asyncio
    async def test_enter_long_success(self, mock_deps):
        assert await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_called_once_with("BTCUSDT", "buy", 1.0)
        mock_deps["protection"].place_entry_protection.assert_awaited_once()
        mock_deps["audit_store"].record_open_trade.assert_called_once()
        metrics = mock_deps["audit_store"].record_execution_attempt.call_args.kwargs
        assert metrics["phase"] == "entry"
        assert metrics["status"] == "completed"
        assert metrics["actual_price"] == 100.0
        mock_deps["portfolio"].add_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_enter_short_success(self, mock_deps):
        assert await mock_deps["executor"].enter_short("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_called_once_with("BTCUSDT", "sell", 1.0)

    @pytest.mark.asyncio
    async def test_unfavorable_funding_blocks_entry(self, mock_deps, monkeypatch):
        monkeypatch.setattr(settings, "max_unfavorable_funding_rate", 0.001)
        mock_deps["client"].fetch_funding_rate.return_value = 0.002

        opened = await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)

        assert opened is False
        mock_deps["orders"].market_order.assert_not_awaited()
        assert any(
            call.args[0] == "trade_blocked_market_risk"
            for call in mock_deps["audit_store"].safe_record_event.call_args_list
        )

    @pytest.mark.asyncio
    async def test_order_book_depth_impact_blocks_entry(self, mock_deps, monkeypatch):
        monkeypatch.setattr(settings, "scalp_max_market_depth_slippage_bps", 5.0)
        mock_deps["client"].fetch_order_book.return_value = {
            "bids": [[99.0, 1.0]],
            "asks": [[100.0, 0.4], [101.0, 0.6]],
        }

        opened = await mock_deps["executor"].enter_long(
            "BTCUSDT",
            100.0,
            2.0,
            strategy="scalp",
        )

        assert opened is False
        mock_deps["orders"].market_order.assert_not_awaited()
        assert any(
            "depth impact" in str(call.args[1])
            for call in mock_deps["audit_store"].safe_record_event.call_args_list
        )

    @pytest.mark.asyncio
    async def test_enter_trading_disabled(self, mock_deps):
        mock_deps["audit_store"].trading_allowed.return_value = (
            False,
            "emergency stop",
        )
        assert not await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_enter_position_size_zero(self, mock_deps):
        mock_deps["sizer"].calculate.return_value = MagicMock(quantity=0.0)
        assert not await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_enter_exposure_limit_blocks(self, mock_deps):
        mock_deps["check_exposure_limits"].return_value = (False, "max open positions")
        assert not await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_enter_market_order_not_filled(self, mock_deps):
        mock_deps["orders"].market_order.return_value = {
            "id": "m1",
            "filled": 0,
            "average": None,
        }
        assert not await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        metrics = mock_deps["audit_store"].record_execution_attempt.call_args.kwargs
        assert metrics["status"] == "order_failed"

    @pytest.mark.asyncio
    async def test_enter_fill_slippage_exceeds_limit(self, mock_deps, monkeypatch):
        monkeypatch.setattr(settings, "max_entry_slippage_bps", 25.0)
        mock_deps["fill_resolver"].resolve_entry_fill_price.return_value = 100.5
        assert not await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_any_call("BTCUSDT", "buy", 1.0)
        mock_deps["orders"].market_order.assert_any_call(
            "BTCUSDT", "sell", 1.0, reduce_only=True
        )

    @pytest.mark.asyncio
    async def test_post_fill_cost_recheck_flattens_weak_edge(
        self,
        mock_deps,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "max_entry_slippage_bps", 100.0)
        monkeypatch.setattr(settings, "adaptive_limit_entry_enabled", False)
        mock_deps["fill_resolver"].resolve_entry_fill_price.return_value = 100.5

        def risk_levels(entry_price, side, atr, strategy=None):
            if entry_price == 100.0:
                return MagicMock(stop_loss=99.0, take_profit=104.0)
            return MagicMock(stop_loss=98.0, take_profit=100.6)

        mock_deps["sl_manager"].calculate.side_effect = risk_levels

        opened = await mock_deps["executor"].enter_long(
            "BTCUSDT",
            100.0,
            2.0,
            strategy="trend",
        )

        assert opened is False
        mock_deps["protection"].place_entry_protection.assert_not_awaited()
        mock_deps["orders"].market_order.assert_any_call(
            "BTCUSDT",
            "sell",
            1.0,
            reduce_only=True,
        )
        metrics = mock_deps["audit_store"].record_execution_attempt.call_args.kwargs
        assert metrics["status"] == "post_fill_cost_rejected"
        assert "target edge too small" in metrics["reason"]

    @pytest.mark.asyncio
    async def test_scalp_fill_slippage_uses_strict_limit(
        self,
        mock_deps,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "max_entry_slippage_bps", 45.0)
        monkeypatch.setattr(settings, "scalp_max_entry_slippage_bps", 5.0)
        mock_deps["fill_resolver"].resolve_entry_fill_price.return_value = 100.07

        result = await mock_deps["executor"].enter_long(
            "BTCUSDT",
            100.0,
            2.0,
            timeframe="1m",
            strategy="scalp",
        )

        assert not result
        mock_deps["orders"].market_order.assert_any_call(
            "BTCUSDT",
            "sell",
            1.0,
            reduce_only=True,
        )

    @pytest.mark.asyncio
    async def test_enter_protection_failure_triggers_emergency_flatten(self, mock_deps):
        mock_deps["protection"].place_entry_protection.return_value = None
        assert not await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_any_call("BTCUSDT", "buy", 1.0)
        mock_deps["orders"].market_order.assert_any_call(
            "BTCUSDT", "sell", 1.0, reduce_only=True
        )

    @pytest.mark.asyncio
    async def test_enter_audit_failure_triggers_emergency_flatten(self, mock_deps):
        mock_deps["audit_store"].record_open_trade.side_effect = RuntimeError(
            "db error"
        )
        assert not await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["orders"].market_order.assert_any_call("BTCUSDT", "buy", 1.0)

    @pytest.mark.asyncio
    async def test_enter_with_timeframe_sets_position_key(self, mock_deps):
        mock_deps["trades"].position_key.return_value = "BTCUSDT-15m"
        assert await mock_deps["executor"].enter_long(
            "BTCUSDT", 100.0, 2.0, timeframe="15m"
        )
        mock_deps["trades"].position_key.assert_called_with("BTCUSDT", "15m", None)

    @pytest.mark.asyncio
    async def test_enter_sends_notifications(self, mock_deps):
        assert await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)
        mock_deps["alerter"].trade_opened_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_favorable_fill_records_zero_adverse_slippage(self, mock_deps):
        mock_deps["fill_resolver"].resolve_entry_fill_price.return_value = 99.0

        assert await mock_deps["executor"].enter_long("BTCUSDT", 100.0, 2.0)

        completed = [
            call.kwargs
            for call in mock_deps["audit_store"].record_execution_attempt.call_args_list
            if call.kwargs.get("status") == "completed"
        ]
        assert completed[-1]["slippage_bps"] == 0.0

    @pytest.mark.asyncio
    async def test_scalp_passes_strategy_to_risk_components(self, mock_deps):
        assert await mock_deps["executor"].enter_long(
            "BTCUSDT",
            100.0,
            2.0,
            timeframe="1m",
            strategy="scalp",
        )

        mock_deps["sl_manager"].calculate.assert_any_call(
            100.0,
            "long",
            2.0,
            strategy="scalp",
        )
        mock_deps["sizer"].calculate.assert_called_once_with(
            100.0,
            98.0,
            1,
            "long",
            strategy="scalp",
        )

    @pytest.mark.asyncio
    async def test_scalp_wide_spread_is_blocked_before_order(self, mock_deps):
        mock_deps["client"].fetch_ticker.return_value = {
            "bid": 99.0,
            "ask": 101.0,
            "last": 100.0,
            "mark": 100.0,
        }

        assert not await mock_deps["executor"].enter_long(
            "BTCUSDT",
            100.0,
            2.0,
            timeframe="1m",
            strategy="scalp",
        )

        mock_deps["orders"].market_order.assert_not_called()
        event = mock_deps["audit_store"].safe_record_event.call_args
        assert event.args[0] == "scalp_entry_blocked"
        assert "spread above limit" in event.args[1]

    @pytest.mark.asyncio
    async def test_scalp_uses_order_book_when_ticker_has_no_bid_ask(
        self,
        mock_deps,
    ):
        mock_deps["client"].fetch_ticker.return_value = {
            "bid": None,
            "ask": None,
            "last": 100.0,
            "mark": 100.0,
        }

        assert await mock_deps["executor"].enter_long(
            "BTCUSDT",
            100.0,
            2.0,
            timeframe="1m",
            strategy="scalp",
        )

        assert mock_deps["client"].fetch_order_book.await_count == 3
        mock_deps["client"].fetch_order_book.assert_any_await("BTCUSDT", limit=5)
        mock_deps["client"].fetch_order_book.assert_any_await(
            "BTCUSDT",
            limit=settings.market_depth_levels,
        )


class TestExitPosition:
    @pytest.mark.asyncio
    async def test_exit_successful(self, mock_deps):
        trade = TradeRecord(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            quantity=1.0,
            timestamp=datetime.now(),
        )
        mock_deps["trades"].open_trades["BTCUSDT"] = trade
        mock_deps["trades"].trade_correlation_ids["BTCUSDT"] = "corr-1"
        mock_deps["protection"].active_stops["BTCUSDT"] = "sl-1"

        await mock_deps["executor"].exit_position("BTCUSDT", "manual")
        mock_deps["orders"].market_order.assert_called_once_with(
            "BTCUSDT", "sell", 1.0, reduce_only=True
        )
        mock_deps["portfolio"].close_trade.assert_called_once_with(
            trade,
            101.0,
            "manual",
            exit_fee=0.0,
        )
        mock_deps["orders"].cancel_order.assert_any_call(
            "BTCUSDT", "sl-1", conditional=True
        )
        metrics = mock_deps["audit_store"].record_execution_attempt.call_args.kwargs
        assert metrics["phase"] == "exit"
        assert metrics["status"] == "completed"

    @pytest.mark.asyncio
    async def test_exit_trade_not_found_is_noop(self, mock_deps):
        await mock_deps["executor"].exit_position("NONEXISTENT", "manual")
        mock_deps["orders"].market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_exit_order_fails(self, mock_deps):
        trade = TradeRecord(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            quantity=1.0,
            timestamp=datetime.now(),
        )
        mock_deps["trades"].open_trades["BTCUSDT"] = trade
        mock_deps["fill_resolver"].confirmed_full_fill.return_value = None

        await mock_deps["executor"].exit_position("BTCUSDT", "manual")
        mock_deps["orders"].market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_cancels_protection_orders(self, mock_deps):
        trade = TradeRecord(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            quantity=1.0,
            timestamp=datetime.now(),
        )
        mock_deps["trades"].open_trades["BTCUSDT"] = trade
        mock_deps["trades"].trade_correlation_ids["BTCUSDT"] = "corr-1"
        mock_deps["protection"].active_stops["BTCUSDT"] = "sl-1"
        mock_deps["protection"].active_tps["BTCUSDT"] = "tp-1"

        await mock_deps["executor"].exit_position("BTCUSDT", "manual")
        mock_deps["orders"].cancel_order.assert_any_call(
            "BTCUSDT", "sl-1", conditional=True
        )
        mock_deps["orders"].cancel_order.assert_any_call("BTCUSDT", "tp-1")

    @pytest.mark.asyncio
    async def test_partial_exit_persists_remaining_protected_quantity(
        self,
        mock_deps,
    ):
        trade = TradeRecord(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            quantity=1.0,
            timestamp=datetime.now(),
            timeframe="1m",
            strategy="scalp",
            entry_fee=0.4,
        )
        mock_deps["trades"].open_trades["BTCUSDT:1m"] = trade
        mock_deps["trades"].trade_correlation_ids["BTCUSDT:1m"] = "corr-1"
        mock_deps["fill_resolver"].confirmed_full_fill.return_value = {
            "id": "partial-1",
            "filled": 0.5,
            "average": 101.0,
        }
        mock_deps["fill_resolver"].order_fee.return_value = 0.2
        mock_deps["audit_store"].get_trade_protection_levels.return_value = (
            99.0,
            102.0,
        )
        mock_deps["protection"].clear_protection.return_value = ("sl-old", "tp-old")
        mock_deps["protection"].place_entry_protection.return_value = (
            {"id": "sl-new"},
            {"id": "tp-new"},
        )

        result = await mock_deps["executor"].partial_exit_position(
            "BTCUSDT:1m",
            0.5,
            "scalp partial profit",
        )

        assert result
        assert trade.quantity == 0.5
        assert trade.entry_fee == 0.2
        closed_trade = mock_deps["portfolio"].close_trade.call_args.args[0]
        assert closed_trade.entry_fee == 0.2
        assert mock_deps["portfolio"].close_trade.call_args.kwargs == {"exit_fee": 0.2}
        mock_deps["orders"].market_order.assert_awaited_once_with(
            "BTCUSDT",
            "sell",
            0.5,
            reduce_only=True,
        )
        mock_deps["audit_store"].record_closed_trade.assert_called_once()
        mock_deps["audit_store"].update_open_trade_state.assert_called_once()
        mock_deps["protection"].track_protection.assert_called_once_with(
            "BTCUSDT:1m",
            "sl-new",
            "tp-new",
        )


class TestEntryPreflight:
    @pytest.mark.asyncio
    async def test_policy_blocks_can_trade(self, mock_deps):
        mock_deps["portfolio"].can_trade.return_value = (False, "manual pause")
        mock_deps["trades"].policy_alert_due.return_value = True
        assert await mock_deps["executor"]._entry_policy_blocks("BTCUSDT")
        mock_deps["alerter"].trade_failed_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_policy_blocks_reentry_cooldown(self, mock_deps):
        mock_deps["trades"].reentry_cooldown_reason.return_value = "cooldown active"
        assert await mock_deps["executor"]._entry_policy_blocks("BTCUSDT")

    @pytest.mark.asyncio
    async def test_policy_bypasses_reentry_cooldown_for_reversal(self, mock_deps):
        mock_deps["trades"].reentry_cooldown_reason.return_value = "cooldown active"
        blocked = await mock_deps["executor"]._entry_policy_blocks(
            "BTCUSDT",
            ignore_reentry_cooldown=True,
        )

        assert blocked is False

    @pytest.mark.asyncio
    async def test_policy_passes_when_no_block(self, mock_deps):
        assert not await mock_deps["executor"]._entry_policy_blocks("BTCUSDT")

    @pytest.mark.asyncio
    async def test_price_drift_blocks_when_excessive(self, mock_deps, monkeypatch):
        monkeypatch.setattr(settings, "max_entry_slippage_bps", 25.0)
        mock_deps["client"].fetch_ticker.return_value["ask"] = 100.5
        reason = await mock_deps["executor"]._entry_price_drift_reason(
            "BTCUSDT", "long", 100.0
        )
        assert "drift above limit" in reason

    @pytest.mark.asyncio
    async def test_scalp_price_drift_uses_strict_limit(
        self,
        mock_deps,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "max_entry_slippage_bps", 45.0)
        monkeypatch.setattr(settings, "scalp_max_entry_slippage_bps", 5.0)
        mock_deps["client"].fetch_ticker.return_value["bid"] = 100.07
        mock_deps["client"].fetch_ticker.return_value["ask"] = 100.07

        reason = await mock_deps["executor"]._entry_price_drift_reason(
            "BTCUSDT",
            "long",
            100.0,
            strategy="scalp",
        )

        assert "7.00bps > 5.00bps" in reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("side", "quote_key", "quote"),
        [
            ("long", "ask", 99.0),
            ("short", "bid", 101.0),
        ],
    )
    async def test_small_favorable_price_drift_can_pass(
        self,
        mock_deps,
        monkeypatch,
        side,
        quote_key,
        quote,
    ):
        monkeypatch.setattr(settings, "max_entry_slippage_bps", 150.0)
        mock_deps["client"].fetch_ticker.return_value[quote_key] = quote

        reason = await mock_deps["executor"]._entry_price_drift_reason(
            "BTCUSDT",
            side,
            100.0,
        )

        assert reason == ""

    @pytest.mark.asyncio
    async def test_excessive_favorable_price_drift_is_stale(
        self,
        mock_deps,
        monkeypatch,
    ):
        monkeypatch.setattr(settings, "max_entry_slippage_bps", 45.0)
        mock_deps["client"].fetch_ticker.return_value["bid"] = 101.0

        reason = await mock_deps["executor"]._entry_price_drift_reason(
            "BTCUSDT",
            "short",
            100.0,
        )

        assert "signal price drift above limit" in reason
        assert "100.00bps > 45.00bps" in reason

    @pytest.mark.parametrize(
        ("side", "signal_price", "fill_price", "expected"),
        [
            ("long", 100.0, 99.0, 0.0),
            ("long", 100.0, 101.0, 100.0),
            ("short", 100.0, 101.0, 0.0),
            ("short", 100.0, 99.0, 100.0),
        ],
    )
    def test_adverse_entry_drift_is_side_aware(
        self,
        side,
        signal_price,
        fill_price,
        expected,
    ):
        assert TradeExecutor._adverse_entry_drift_bps(
            side,
            signal_price,
            fill_price,
        ) == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_price_drift_passes_within_limit(self, mock_deps):
        reason = await mock_deps["executor"]._entry_price_drift_reason(
            "BTCUSDT", "long", 100.0
        )
        assert reason == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "take_profit"),
    [
        ("long", 99.0),
        ("short", 101.0),
    ],
)
async def test_scalp_cost_rejects_take_profit_on_losing_side(
    mock_deps,
    side,
    take_profit,
):
    levels = MagicMock(take_profit=take_profit)

    reason = await mock_deps["executor"]._scalp_cost_reason(
        "BTCUSDT",
        side,
        100.0,
        levels,
    )

    assert "not profitable" in reason


@pytest.mark.asyncio
async def test_trend_cost_gate_requires_edge_after_fees_and_spread(mock_deps):
    levels = MagicMock(take_profit=100.1, stop_loss=99.0)

    reason = await mock_deps["executor"]._strategy_cost_reason(
        "BTCUSDT",
        "long",
        100.0,
        levels,
        "trend",
    )

    assert "target edge too small" in reason


@pytest.mark.asyncio
async def test_trend_cost_gate_accepts_sufficient_after_cost_edge(mock_deps):
    levels = MagicMock(take_profit=104.0, stop_loss=99.0)

    reason = await mock_deps["executor"]._strategy_cost_reason(
        "BTCUSDT",
        "long",
        100.0,
        levels,
        "trend",
    )

    assert reason == ""


@pytest.mark.asyncio
async def test_trend_cost_gate_rejects_weak_after_cost_reward_risk(mock_deps):
    levels = MagicMock(take_profit=104.0, stop_loss=97.0)

    reason = await mock_deps["executor"]._strategy_cost_reason(
        "BTCUSDT",
        "long",
        100.0,
        levels,
        "trend",
    )

    assert "after-cost reward/risk too weak" in reason


class TestReportEntryLimitBlock:
    @pytest.mark.asyncio
    async def test_policy_block_logs_info(self, mock_deps):
        await mock_deps["executor"]._report_entry_limit_block(
            "BTCUSDT",
            "trade leg already open",
            payload={},
        )
        mock_deps["audit_store"].safe_record_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_exposure_block_notifies_failure(self, mock_deps):
        await mock_deps["executor"]._report_entry_limit_block(
            "BTCUSDT",
            "max open positions",
            payload={},
        )
        mock_deps["alerter"].trade_failed_alert.assert_awaited_once()


class TestCloseAll:
    @pytest.mark.asyncio
    async def test_close_all_exits_all_symbols(self, mock_deps):
        trade1 = TradeRecord(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            quantity=1.0,
            timestamp=datetime.now(),
        )
        trade2 = TradeRecord(
            symbol="ETHUSDT",
            side="short",
            entry_price=200.0,
            quantity=2.0,
            timestamp=datetime.now(),
        )
        mock_deps["trades"].open_trades["BTCUSDT"] = trade1
        mock_deps["trades"].open_trades["ETHUSDT"] = trade2
        mock_deps["trades"].trades_for_symbol.side_effect = lambda s: [
            (k, t)
            for k, t in [("BTCUSDT", trade1), ("ETHUSDT", trade2)]
            if t.symbol == s
        ]

        await mock_deps["executor"].close_all()
        assert mock_deps["orders"].market_order.call_count == 2

    @pytest.mark.asyncio
    async def test_close_all_records_symbol_failure_without_raising(self, mock_deps):
        trade = TradeRecord(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            quantity=1.0,
            timestamp=datetime.now(),
        )
        mock_deps["trades"].open_trades["BTCUSDT"] = trade
        mock_deps["trades"].trades_for_symbol.return_value = [("BTCUSDT", trade)]
        mock_deps["orders"].market_order.side_effect = RuntimeError(
            "exchange unavailable"
        )

        result = await mock_deps["executor"].close_all()

        assert result is False
        mock_deps["audit_store"].activate_emergency_stop.assert_called_once()
        mock_deps["audit_store"].safe_record_event.assert_called()
