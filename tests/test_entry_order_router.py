from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.entry_order_router import EntryOrderRouter


@pytest.fixture
def router():
    client = MagicMock()
    client.fetch_ticker = AsyncMock(return_value={"bid": 99.0, "ask": 101.0})
    client.fetch_order = AsyncMock()
    orders = MagicMock()
    orders.market_order = AsyncMock(
        return_value={"id": "market", "filled": 1.0, "status": "closed"}
    )
    orders.limit_order = AsyncMock()
    orders.cancel_order = AsyncMock()
    return EntryOrderRouter(client, orders), client, orders


@pytest.mark.asyncio
async def test_urgent_scalp_uses_market_order(router):
    entry_router, client, orders = router

    order = await entry_router.submit("BTCUSDT", "buy", 1.0, strategy="scalp")

    assert order["id"] == "market"
    orders.market_order.assert_awaited_once_with("BTCUSDT", "buy", 1.0)
    orders.limit_order.assert_not_awaited()
    client.fetch_ticker.assert_not_awaited()


@pytest.mark.asyncio
async def test_trend_uses_post_only_limit_when_immediately_filled(router):
    entry_router, client, orders = router
    orders.limit_order.return_value = {
        "id": "limit",
        "filled": 1.0,
        "status": "closed",
    }

    order = await entry_router.submit("BTCUSDT", "buy", 1.0, strategy="trend")

    orders.limit_order.assert_awaited_once_with(
        "BTCUSDT",
        "buy",
        1.0,
        99.0,
        post_only=True,
    )
    orders.market_order.assert_not_awaited()
    assert order["_bot_order_policy"] == "passive_limit"


@pytest.mark.asyncio
async def test_rejected_passive_order_falls_back_to_market(router):
    entry_router, client, orders = router
    orders.limit_order.return_value = {
        "id": "limit",
        "filled": 0.0,
        "status": "open",
    }
    client.fetch_order.return_value = {
        "id": "limit",
        "filled": 0.0,
        "status": "rejected",
    }

    order = await entry_router.submit("BTCUSDT", "sell", 1.0, strategy="range")

    orders.cancel_order.assert_awaited_once_with("BTCUSDT", "limit")
    orders.market_order.assert_awaited_once_with("BTCUSDT", "sell", 1.0)
    assert order["_bot_order_policy"] == "market_fallback"


@pytest.mark.asyncio
async def test_partial_passive_fill_cancels_remainder_without_market_fallback(router):
    entry_router, client, orders = router
    orders.limit_order.return_value = {
        "id": "limit",
        "filled": 0.4,
        "status": "open",
    }

    order = await entry_router.submit("BTCUSDT", "buy", 1.0, strategy="trend")

    orders.cancel_order.assert_awaited_once_with("BTCUSDT", "limit")
    orders.market_order.assert_not_awaited()
    assert order["filled"] == 0.4
