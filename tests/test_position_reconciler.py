import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.position_reconciler import PositionReconciler
from src.risk.portfolio import TradeRecord


@pytest.fixture
def reconciler():
    orders = MagicMock()
    orders.cancel_order = AsyncMock(return_value=None)
    fill_resolver = MagicMock()
    fill_resolver.order_fee = AsyncMock(return_value=0.0)
    fill_resolver.execution_fee.side_effect = lambda _trade: 0.0
    return PositionReconciler(
        client=MagicMock(),
        orders=orders,
        fill_resolver=fill_resolver,
        protection=MagicMock(),
        trades=MagicMock(),
        portfolio=MagicMock(),
        alerter=MagicMock(),
        audit_store=MagicMock(),
        mode="trade",
    )


def make_trade(**kwargs):
    return TradeRecord(
        symbol=kwargs.get("symbol", "BTCUSDT"),
        side=kwargs.get("side", "long"),
        entry_price=kwargs.get("entry_price", 100.0),
        quantity=kwargs.get("quantity", 1.0),
        timestamp=kwargs.get("timestamp", datetime.datetime.now()),
    )


class TestPositionSize:
    def test_from_contracts_key(self, reconciler):
        assert reconciler._position_size({"contracts": "1.5"}) == 1.5

    def test_from_info_positionAmt(self, reconciler):
        assert reconciler._position_size({"info": {"positionAmt": "-2.0"}}) == -2.0

    def test_returns_zero_when_no_size(self, reconciler):
        assert reconciler._position_size({}) == 0.0


class TestPositionSide:
    def test_from_side_key(self, reconciler):
        assert reconciler._position_side({"side": "long"}) == "long"

    def test_from_info_positionSide(self, reconciler):
        assert reconciler._position_side({"info": {"positionSide": "short"}}) == "short"

    def test_from_info_positionAmt_positive(self, reconciler):
        assert reconciler._position_side({"info": {"positionAmt": "1.5"}}) == "long"

    def test_from_info_positionAmt_negative(self, reconciler):
        assert reconciler._position_side({"info": {"positionAmt": "-2.0"}}) == "short"

    def test_from_size_positive(self, reconciler):
        assert reconciler._position_side({"contracts": "1.0"}) == "long"

    def test_from_size_negative(self, reconciler):
        assert reconciler._position_side({"contracts": "-1.0"}) == "short"

    def test_returns_none_when_no_side(self, reconciler):
        assert reconciler._position_side({}) is None


class TestClassifyExchangePositions:
    def test_separates_unmanaged_positions(self, reconciler):
        reconciler.trades.normalize_symbol.side_effect = lambda s: s
        reconciler.trades.has_open_trade_for_symbol.return_value = False
        positions = [
            {"symbol": "BTCUSDT", "contracts": "1.0"},
            {"symbol": "ETHUSDT", "contracts": "2.0"},
        ]
        exchange_positions, unmanaged = reconciler._classify_exchange_positions(
            positions
        )
        assert "BTCUSDT" in exchange_positions
        assert len(unmanaged) == 2

    def test_filters_zero_size_positions(self, reconciler):
        positions = [{"symbol": "BTCUSDT", "contracts": "0"}]
        exchange_positions, unmanaged = reconciler._classify_exchange_positions(
            positions
        )
        assert exchange_positions == {}
        assert unmanaged == []


class TestReconciliationConnectivity:
    @pytest.mark.asyncio
    async def test_transient_outage_is_audited_without_immediate_alert(
        self, reconciler, monkeypatch
    ):
        monkeypatch.setattr(
            "src.execution.position_reconciler.settings."
            "reconciliation_alert_failure_threshold",
            3,
        )
        reconciler.trades.open_trades = {}
        reconciler.client.fetch_positions = AsyncMock(
            side_effect=RuntimeError("temporary outage")
        )
        reconciler.alerter.error_alert = AsyncMock()

        assert await reconciler._fetch_positions_for_reconciliation() is None

        reconciler.alerter.error_alert.assert_not_awaited()
        assert reconciler._position_fetch_failures == 1

    @pytest.mark.asyncio
    async def test_repeated_outage_alerts_and_recovery_resets_counter(
        self, reconciler, monkeypatch
    ):
        monkeypatch.setattr(
            "src.execution.position_reconciler.settings."
            "reconciliation_alert_failure_threshold",
            2,
        )
        reconciler.trades.open_trades = {}
        reconciler.client.fetch_positions = AsyncMock(
            side_effect=[RuntimeError("one"), RuntimeError("two"), []]
        )
        reconciler.alerter.error_alert = AsyncMock()

        assert await reconciler._fetch_positions_for_reconciliation() is None
        assert await reconciler._fetch_positions_for_reconciliation() is None
        assert await reconciler._fetch_positions_for_reconciliation() == []

        reconciler.alerter.error_alert.assert_awaited_once()
        assert reconciler._position_fetch_failures == 0


class TestInferredProtectiveExitReason:
    def test_long_profitable_is_take_profit(self, reconciler):
        trade = make_trade(side="long", entry_price=100.0)
        reason = reconciler._inferred_protective_exit_reason("BTCUSDT", trade, 110.0)
        assert reason == "take_profit"

    def test_long_unprofitable_is_stop_loss(self, reconciler):
        trade = make_trade(side="long", entry_price=100.0)
        reason = reconciler._inferred_protective_exit_reason("BTCUSDT", trade, 90.0)
        assert reason == "stop_loss"

    def test_short_profitable_is_take_profit(self, reconciler):
        trade = make_trade(side="short", entry_price=100.0)
        reason = reconciler._inferred_protective_exit_reason("BTCUSDT", trade, 90.0)
        assert reason == "take_profit"

    def test_short_unprofitable_is_stop_loss(self, reconciler):
        trade = make_trade(side="short", entry_price=100.0)
        reason = reconciler._inferred_protective_exit_reason("BTCUSDT", trade, 110.0)
        assert reason == "stop_loss"


class TestExitExecutionGroups:
    @pytest.mark.asyncio
    async def test_groups_by_order_id(self, reconciler):
        reconciler.client.fetch_my_trades = AsyncMock(
            return_value=[
                {
                    "order": "o1",
                    "amount": 0.5,
                    "price": 100.0,
                    "side": "sell",
                    "timestamp": 2000,
                },
                {
                    "order": "o1",
                    "amount": 0.5,
                    "price": 101.0,
                    "side": "sell",
                    "timestamp": 3000,
                },
                {
                    "order": "o2",
                    "amount": 1.0,
                    "price": 102.0,
                    "side": "sell",
                    "timestamp": 1000,
                },
            ]
        )
        groups = await reconciler._exit_execution_groups("BTCUSDT")
        assert len(groups) == 2
        o1 = [g for g in groups if g["order_id"] == "o1"][0]
        assert o1["price"] == 100.5
        assert o1["quantity"] == 1.0
        assert o1["timestamp"] == 2000

    @pytest.mark.asyncio
    async def test_skips_invalid_entries(self, reconciler):
        reconciler.client.fetch_my_trades = AsyncMock(
            return_value=[
                {
                    "order": "",
                    "amount": 0.0,
                    "price": 0,
                    "side": "sell",
                    "timestamp": 1000,
                },
            ]
        )
        groups = await reconciler._exit_execution_groups("BTCUSDT")
        assert len(groups) == 0

    @pytest.mark.asyncio
    async def test_returns_empty_on_fetch_failure(self, reconciler):
        reconciler.client.fetch_my_trades = AsyncMock(
            side_effect=Exception("no connection")
        )
        groups = await reconciler._exit_execution_groups("BTCUSDT")
        assert groups == []


class TestProtectiveExecutionAttribution:
    def test_requires_execution_price_near_audited_protection(
        self,
        reconciler,
        monkeypatch,
    ):
        reconciler.trades.trade_correlation_ids = {"BTCUSDT:1d": "corr-1d"}
        reconciler.audit_store.get_trade_protection_levels.return_value = (
            90.0,
            120.0,
        )
        monkeypatch.setattr(
            "src.execution.position_reconciler.settings."
            "protection_execution_match_bps",
            50.0,
        )

        assert reconciler._execution_matches_protection(
            "BTCUSDT:1d",
            "stop_loss",
            90.2,
        )
        assert not reconciler._execution_matches_protection(
            "BTCUSDT:1d",
            "stop_loss",
            99.0,
        )

    @pytest.mark.asyncio
    async def test_one_exit_execution_cannot_close_two_timeframe_legs(
        self,
        reconciler,
        monkeypatch,
    ):
        first = make_trade(entry_price=100.0)
        second = make_trade(entry_price=100.0)
        reconciler.trades.open_trades = {
            "BTCUSDT:5m": first,
            "BTCUSDT:1d": second,
        }
        reconciler.trades.trade_correlation_ids = {
            "BTCUSDT:5m": "corr-5m",
            "BTCUSDT:1d": "corr-1d",
        }
        reconciler.protection.active_tps = {}
        reconciler.protection.active_stops = {
            "BTCUSDT:5m": "sl-5m",
            "BTCUSDT:1d": "sl-1d",
        }
        reconciler._fill_resolver.resolved_order = AsyncMock(return_value=None)
        exit_order = {"id": "exit-1", "average": 90.0}

        async def latest_exit_order(trade, skip_order_ids):
            if "exit-1" in skip_order_ids:
                return None
            return exit_order

        reconciler._fill_resolver.latest_exit_order = latest_exit_order
        reconciler._fill_resolver.positive_order_price.side_effect = (
            lambda order: order.get("average")
        )
        reconciler.audit_store.get_trade_protection_levels.side_effect = (
            lambda correlation_id: (
                (90.0, 120.0) if correlation_id == "corr-5m" else (80.0, 140.0)
            )
        )
        monkeypatch.setattr(
            "src.execution.position_reconciler.settings."
            "protection_execution_match_bps",
            50.0,
        )

        first_details = await reconciler._filled_protective_exit_details("BTCUSDT:5m")
        second_details = await reconciler._filled_protective_exit_details("BTCUSDT:1d")

        assert first_details == (90.0, "stop_loss", 0.0)
        assert second_details is None
        assert reconciler._consumed_exit_order_ids == {"exit-1"}

    @pytest.mark.asyncio
    async def test_partial_reduction_closes_only_nearest_bnb_leg(
        self,
        reconciler,
        monkeypatch,
    ):
        timestamp = datetime.datetime(2026, 6, 12, tzinfo=datetime.timezone.utc)
        trades = {
            "BNBUSDT:5m": make_trade(
                symbol="BNBUSDT", entry_price=605.97, quantity=0.16, timestamp=timestamp
            ),
            "BNBUSDT:15m": make_trade(
                symbol="BNBUSDT", entry_price=606.00, quantity=0.16, timestamp=timestamp
            ),
            "BNBUSDT:30m": make_trade(
                symbol="BNBUSDT", entry_price=606.06, quantity=0.16, timestamp=timestamp
            ),
        }
        reconciler.trades.open_trades = trades
        reconciler.trades.trades_for_symbol.return_value = list(trades.items())
        reconciler.trades.trade_correlation_ids = {
            "BNBUSDT:5m": "corr-5m",
            "BNBUSDT:15m": "corr-15m",
            "BNBUSDT:30m": "corr-30m",
        }
        reconciler.protection.missing_protection_candidates.return_value = [
            ("BNBUSDT:30m", trades["BNBUSDT:30m"]),
            ("BNBUSDT:15m", trades["BNBUSDT:15m"]),
            ("BNBUSDT:5m", trades["BNBUSDT:5m"]),
        ]
        reconciler._filled_protective_exit_details = AsyncMock(return_value=None)
        reconciler._exit_execution_groups = AsyncMock(
            return_value=[
                {
                    "order_id": "exit-1",
                    "timestamp": int(timestamp.timestamp() * 1000),
                    "side": "sell",
                    "quantity": 0.16,
                    "price": 604.80125,
                }
            ]
        )
        reconciler._fill_resolver.quantity_tolerance = AsyncMock(return_value=0.001)
        reconciler.audit_store.get_trade_protection_levels.side_effect = {
            "corr-5m": (604.85, 608.21),
            "corr-15m": (603.72, 610.56),
            "corr-30m": (603.05, 612.08),
        }.get
        reconciler._finalize_trade_leg = AsyncMock()
        monkeypatch.setattr(
            "src.execution.position_reconciler.settings.protection_execution_match_bps",
            10.0,
        )

        changed = await reconciler._reconcile_partial_exchange_exits(
            {"BNBUSDT": {"contracts": 0.32}},
            {"BNBUSDT": (set(), set())},
            AsyncMock(),
        )

        assert changed is True
        reconciler._finalize_trade_leg.assert_awaited_once()
        assert reconciler._finalize_trade_leg.await_args.args[0] == "BNBUSDT:5m"


class TestEmergencyRecovery:
    @pytest.mark.asyncio
    async def test_verified_flat_state_clears_stale_red_lock(self, reconciler):
        reconciler.trades.open_trades = {}
        reconciler._reconciliation_state = AsyncMock(return_value=({}, {}))
        reconciler._reconcile_partial_exchange_exits = AsyncMock(return_value=False)
        reconciler._clear_missing_exchange_positions = AsyncMock()
        reconciler._verify_exchange_position_details = AsyncMock(return_value=True)
        reconciler.protection.verify_all_protective_orders = AsyncMock(
            return_value=True
        )
        reconciler.audit_store.TRADING_LEVEL_RED = "3"
        reconciler.audit_store.TRADING_LEVEL_GREEN = "0"
        reconciler.audit_store.get_trading_level.side_effect = ["3", "0"]
        notify = AsyncMock()

        assert await reconciler.reconcile_exchange_state(notify) is True

        reconciler.audit_store.clear_emergency_stop.assert_called_once_with(
            "verified consistent exchange, audit, and protection state"
        )
        events = reconciler.audit_store.safe_record_event.call_args_list
        assert any(call.args[0] == "emergency_stop_auto_cleared" for call in events)


class TestFinalizeTradeLeg:
    @pytest.mark.asyncio
    async def test_closes_trade_and_updates_state(self, reconciler):
        trade = MagicMock(symbol="BTCUSDT", side="long")
        reconciler.trades.open_trades = {"BTCUSDT": trade}
        reconciler.trades.trade_correlation_ids = {"BTCUSDT": "corr-1"}
        reconciler.trades.last_symbol_exit_at = {}
        reconciler.protection.active_stops = {"BTCUSDT": "sl-1"}
        reconciler.protection.active_tps = {"BTCUSDT": "tp-1"}
        notify = AsyncMock()

        await reconciler._finalize_trade_leg(
            "BTCUSDT",
            exit_price=101.0,
            reason="take_profit",
            correlation_id="corr-1",
            notify_trade_completed=notify,
        )
        reconciler.portfolio.close_trade.assert_called_once_with(
            trade,
            101.0,
            "take_profit",
            exit_fee=0.0,
        )
        notify.assert_awaited_once_with(trade, 101.0, "take_profit")
        assert "BTCUSDT" not in reconciler.trades.open_trades
        assert "BTCUSDT" not in reconciler.trades.trade_correlation_ids
        assert "BTCUSDT" not in reconciler.protection.active_stops
        assert "BTCUSDT" not in reconciler.protection.active_tps

    @pytest.mark.asyncio
    async def test_skips_cancelling_protection_when_reason_matches(self, reconciler):
        trade = MagicMock(symbol="BTCUSDT", side="long")
        reconciler.trades.open_trades = {"BTCUSDT": trade}
        reconciler.trades.trade_correlation_ids = {"BTCUSDT": "corr-1"}
        reconciler.trades.last_symbol_exit_at = {}
        reconciler.protection.active_stops = {"BTCUSDT": "sl-1"}
        reconciler.protection.active_tps = {}
        notify = AsyncMock()

        await reconciler._finalize_trade_leg(
            "BTCUSDT",
            exit_price=98.0,
            reason="stop_loss",
            correlation_id="",
            notify_trade_completed=notify,
        )
        reconciler.orders.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_audit_when_no_correlation_id(self, reconciler):
        trade = MagicMock(symbol="BTCUSDT", side="long")
        reconciler.trades.open_trades = {"BTCUSDT": trade}
        reconciler.trades.trade_correlation_ids = {"BTCUSDT": ""}
        reconciler.trades.last_symbol_exit_at = {}
        reconciler.protection.active_stops = {}
        notify = AsyncMock()

        await reconciler._finalize_trade_leg(
            "BTCUSDT",
            exit_price=101.0,
            reason="take_profit",
            correlation_id="",
            notify_trade_completed=notify,
        )
        reconciler.audit_store.record_closed_trade.assert_not_called()
