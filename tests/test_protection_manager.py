from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.protection_manager import ProtectionManager
from src.risk.portfolio import TradeRecord
from src.risk.stop_loss import StopLossLevels


@pytest.fixture
def pm():
    client = MagicMock()
    client.fetch_open_orders = AsyncMock(return_value=[])
    orders = MagicMock()
    orders.stop_loss_order = AsyncMock(return_value={"id": "sl-1"})
    orders.take_profit_order = AsyncMock(return_value={"id": "tp-1"})
    orders.cancel_order = AsyncMock(return_value=None)
    return ProtectionManager(
        client=client,
        orders=orders,
        audit_store=MagicMock(),
        fill_resolver=MagicMock(),
        alerter=MagicMock(),
        mode="trade",
    )


class TestPlaceEntryProtection:
    @pytest.mark.asyncio
    async def test_places_sl_and_tp(self, pm):
        levels = StopLossLevels(stop_loss=98.0, take_profit=104.0)
        result = await pm.place_entry_protection("BTCUSDT", "long", 1.0, levels)
        assert result == ({"id": "sl-1"}, {"id": "tp-1"})
        pm.orders.stop_loss_order.assert_awaited_once_with("BTCUSDT", "sell", 1.0, 98.0)
        pm.orders.take_profit_order.assert_awaited_once_with(
            "BTCUSDT", "sell", 1.0, 104.0
        )

    @pytest.mark.asyncio
    async def test_places_sl_only_when_tp_is_none(self, pm):
        levels = StopLossLevels(stop_loss=98.0, take_profit=None)
        result = await pm.place_entry_protection("BTCUSDT", "long", 1.0, levels)
        assert result == ({"id": "sl-1"}, None)
        pm.orders.take_profit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancels_sl_when_tp_fails(self, pm):
        pm.orders.take_profit_order.return_value = None
        levels = StopLossLevels(stop_loss=98.0, take_profit=104.0)
        result = await pm.place_entry_protection("BTCUSDT", "long", 1.0, levels)
        assert result is None
        pm.orders.cancel_order.assert_awaited_once_with(
            "BTCUSDT", "sl-1", conditional=True
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_sl_fails(self, pm):
        pm.orders.stop_loss_order.return_value = None
        levels = StopLossLevels(stop_loss=98.0, take_profit=104.0)
        result = await pm.place_entry_protection("BTCUSDT", "long", 1.0, levels)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_both_fail(self, pm):
        pm.orders.stop_loss_order.return_value = None
        levels = StopLossLevels(stop_loss=98.0, take_profit=104.0)
        result = await pm.place_entry_protection("BTCUSDT", "long", 1.0, levels)
        assert result is None

    @pytest.mark.asyncio
    async def test_replace_stop_places_new_stop_before_cancelling_old(self, pm):
        trade = TradeRecord(
            symbol="BTCUSDT",
            side="long",
            entry_price=100.0,
            quantity=1.0,
            timestamp=MagicMock(),
        )
        pm.active_stops["BTCUSDT:1m"] = "sl-old"
        pm.orders.stop_loss_order.return_value = {"id": "sl-new"}

        result = await pm.replace_stop("BTCUSDT:1m", trade, 100.1)

        assert result == "sl-new"
        assert pm.active_stops["BTCUSDT:1m"] == "sl-new"
        pm.orders.stop_loss_order.assert_awaited_with(
            "BTCUSDT",
            "sell",
            1.0,
            100.1,
        )
        pm.orders.cancel_order.assert_awaited_with(
            "BTCUSDT",
            "sl-old",
            conditional=True,
        )


class TestTrackAndClearProtection:
    def test_track_sl_and_tp(self, pm):
        pm.track_protection("BTCUSDT-5m", "sl-1", "tp-1")
        assert pm.active_stops["BTCUSDT-5m"] == "sl-1"
        assert pm.active_tps["BTCUSDT-5m"] == "tp-1"

    def test_track_skips_tp_if_none(self, pm):
        pm.track_protection("BTCUSDT-5m", "sl-1", None)
        assert "BTCUSDT-5m" in pm.active_stops
        assert "BTCUSDT-5m" not in pm.active_tps

    def test_clear_removes_both(self, pm):
        pm.active_stops["BTCUSDT-5m"] = "sl-1"
        pm.active_tps["BTCUSDT-5m"] = "tp-1"
        stop_id, tp_id = pm.clear_protection("BTCUSDT-5m")
        assert stop_id == "sl-1"
        assert tp_id == "tp-1"
        assert "BTCUSDT-5m" not in pm.active_stops
        assert "BTCUSDT-5m" not in pm.active_tps

    def test_clear_returns_none_when_not_found(self, pm):
        stop_id, tp_id = pm.clear_protection("NONEXISTENT")
        assert stop_id is None
        assert tp_id is None


class TestVerifyProtectiveOrders:
    def test_returns_true_when_both_found(self, pm):
        pm.active_stops["BTCUSDT"] = "sl-1"
        pm.active_tps["BTCUSDT"] = "tp-1"
        assert pm.verify_protective_orders("BTCUSDT", ({"tp-1"}, {"sl-1"}))

    def test_returns_false_when_stop_missing(self, pm):
        pm.active_tps["BTCUSDT"] = "tp-1"
        assert not pm.verify_protective_orders("BTCUSDT", ({"tp-1"}, {"sl-1"}))

    def test_returns_false_when_sl_not_in_conditional(self, pm):
        pm.active_stops["BTCUSDT"] = "sl-1"
        assert not pm.verify_protective_orders("BTCUSDT", (set(), set()))

    def test_returns_false_when_tp_not_in_open(self, pm):
        pm.active_stops["BTCUSDT"] = "sl-1"
        pm.active_tps["BTCUSDT"] = "tp-1"
        assert not pm.verify_protective_orders("BTCUSDT", (set(), {"sl-1"}))

    def test_returns_true_when_tp_not_tracked(self, pm):
        pm.active_stops["BTCUSDT"] = "sl-1"
        assert pm.verify_protective_orders("BTCUSDT", (set(), {"sl-1"}))


class TestCancelProtection:
    @pytest.mark.asyncio
    async def test_cancels_both(self, pm):
        await pm.cancel_protection("BTCUSDT", "sl-1", "tp-1")
        pm.orders.cancel_order.assert_any_call("BTCUSDT", "sl-1", conditional=True)
        pm.orders.cancel_order.assert_any_call("BTCUSDT", "tp-1")

    @pytest.mark.asyncio
    async def test_skips_when_none(self, pm):
        await pm.cancel_protection("BTCUSDT", None, None)
        pm.orders.cancel_order.assert_not_called()


class TestMissingProtectionCandidates:
    def test_identifies_missing_stop(self, pm):
        pm.active_stops["ETHUSDT"] = "sl-1"
        order_ids = (set(), {"sl-2"})
        trade = MagicMock(symbol="ETHUSDT")
        candidates = pm.missing_protection_candidates(
            [("ETHUSDT", trade)],
            order_ids,
        )
        assert len(candidates) == 1

    def test_no_candidates_when_protected(self, pm):
        pm.active_stops["ETHUSDT"] = "sl-1"
        order_ids = (set(), {"sl-1"})
        trade = MagicMock(symbol="ETHUSDT")
        candidates = pm.missing_protection_candidates(
            [("ETHUSDT", trade)],
            order_ids,
        )
        assert len(candidates) == 0


class TestOrphanProtectionCleanup:
    @pytest.mark.asyncio
    async def test_cancels_only_untracked_bot_orders(self, pm):
        orders = [
            {"id": "sl-active", "clientOrderId": "tb_sl_active"},
            {"id": "sl-orphan", "info": {"clientAlgoId": "tb_sl_orphan"}},
            {"id": "manual", "clientOrderId": "manual-stop"},
        ]

        retained = await pm._cancel_orphan_bot_orders(
            "BTCUSDT",
            orders,
            {"sl-active"},
            conditional=True,
        )

        assert {order["id"] for order in retained} == {"sl-active", "manual"}
        pm.orders.cancel_order.assert_awaited_once_with(
            "BTCUSDT",
            "sl-orphan",
            conditional=True,
        )
