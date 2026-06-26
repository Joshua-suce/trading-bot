from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.fill_resolver import FillResolver


@pytest.fixture
def client():
    fetch_order_mock = AsyncMock()
    fetch_order_mock.side_effect = LookupError("not found")
    return MagicMock(
        fetch_my_trades=AsyncMock(return_value=[]),
        fetch_order=fetch_order_mock,
        fetch_orders=AsyncMock(return_value=[]),
        fetch_positions=AsyncMock(return_value=[]),
        fetch_ticker=AsyncMock(return_value={"last": 101.0}),
        fetch_market=AsyncMock(return_value={"precision": {"amount": 3}}),
    )


@pytest.fixture
def resolver(client):
    return FillResolver(client)


class TestConfirmedFullFill:
    @pytest.mark.asyncio
    async def test_returns_order_when_filled_meets_expected(self, resolver, client):
        order = {"id": "o1", "filled": 1.0}
        result = await resolver.confirmed_full_fill(order, "BTCUSDT", 1.0)
        assert result == order

    @pytest.mark.asyncio
    async def test_refreshes_order_when_filled_below_tolerance(self, resolver, client):
        order = {"id": "o1", "filled": 0.5}
        client.fetch_order.side_effect = None
        client.fetch_order.return_value = {"id": "o1", "filled": 1.0}
        result = await resolver.confirmed_full_fill(order, "BTCUSDT", 1.0)
        assert result == {"id": "o1", "filled": 1.0}

    @pytest.mark.asyncio
    async def test_falls_back_to_execution_fill_when_refresh_fails(
        self, resolver, client
    ):
        order = {"id": "o1", "filled": 0.5}
        client.fetch_order.return_value = {"id": "o1", "filled": 0.5}
        client.fetch_my_trades.return_value = [
            {"order": "o1", "amount": 1.0, "price": 100.0},
        ]
        result = await resolver.confirmed_full_fill(order, "BTCUSDT", 1.0)
        assert result is not None
        assert result["filled"] == 1.0
        assert result["average"] == 100.0

    @pytest.mark.asyncio
    async def test_returns_none_when_no_fill_found(self, resolver, client):
        order = {"id": "o1", "filled": 0.0}
        assert await resolver.confirmed_full_fill(order, "BTCUSDT", 1.0) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_order_is_none(self, resolver, client):
        assert await resolver.confirmed_full_fill(None, "BTCUSDT", 1.0) is None


class TestResolveEntryFillPrice:
    @pytest.mark.asyncio
    async def test_returns_order_average_when_present(self, resolver, client):
        order = {"id": "o1", "filled": 1.0, "average": 100.5}
        result = await resolver.resolve_entry_fill_price(order, "BTCUSDT")
        assert result == 100.5

    @pytest.mark.asyncio
    async def test_returns_execution_price_when_order_has_no_average(
        self, resolver, client
    ):
        order = {"id": "o1", "filled": 1.0}
        client.fetch_my_trades.return_value = [
            {"order": "o1", "amount": 1.0, "price": 100.5},
        ]
        result = await resolver.resolve_entry_fill_price(order, "BTCUSDT")
        assert result == 100.5

    @pytest.mark.asyncio
    async def test_returns_resolved_order_price_when_no_executions(
        self, resolver, client
    ):
        order = {"id": "o1", "filled": 1.0, "average": None}
        client.fetch_order.side_effect = None
        client.fetch_order.return_value = {"id": "o1", "average": 100.5}
        result = await resolver.resolve_entry_fill_price(order, "BTCUSDT")
        assert result == 100.5

    @pytest.mark.asyncio
    async def test_recovers_demo_fill_price_from_position_snapshot(
        self, resolver, client, monkeypatch
    ):
        async def no_sleep(_delay):
            return None

        monkeypatch.setattr("src.execution.fill_resolver.asyncio.sleep", no_sleep)
        order = {
            "id": "o1",
            "filled": 0.057,
            "average": None,
            "side": "buy",
            "status": "closed",
            "info": {"executedQty": "0.057", "status": "FILLED"},
        }
        client.fetch_positions.return_value = [
            {
                "symbol": "ETH/USDT:USDT",
                "contracts": 0.057,
                "side": "long",
                "entryPrice": 1724.92,
            }
        ]

        result = await resolver.resolve_entry_fill_price(order, "ETHUSDT")

        assert result == 1724.92
        client.fetch_order.assert_not_awaited()
        client.fetch_my_trades.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_position_snapshot_rejects_wrong_side(
        self, resolver, client, monkeypatch
    ):
        async def no_sleep(_delay):
            return None

        monkeypatch.setattr("src.execution.fill_resolver.asyncio.sleep", no_sleep)
        client.fetch_positions.return_value = [
            {
                "symbol": "ETH/USDT:USDT",
                "contracts": 0.057,
                "side": "short",
                "entryPrice": 1724.92,
            }
        ]

        result = await resolver.position_entry_price("ETHUSDT", 0.057, "long")

        assert result is None

    @pytest.mark.asyncio
    async def test_rechecks_position_after_demo_ledgers_fail(
        self, resolver, client, monkeypatch
    ):
        async def no_sleep(_delay):
            return None

        monkeypatch.setattr("src.execution.fill_resolver.asyncio.sleep", no_sleep)
        client.fetch_order.side_effect = LookupError("order does not exist")
        client.fetch_positions.side_effect = [
            [],
            [],
            [],
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "contracts": 0.0015,
                    "side": "long",
                    "entryPrice": 65906.2,
                }
            ],
        ]
        order = {
            "id": "15223699934",
            "filled": 0.0015,
            "average": None,
            "side": "buy",
            "status": "closed",
        }

        result = await resolver.resolve_entry_fill_price(order, "BTCUSDT")

        assert result == 65906.2
        assert client.fetch_positions.await_count == 4

    @pytest.mark.asyncio
    async def test_matching_position_can_use_slippage_validated_quote_proxy(
        self, resolver, client, monkeypatch
    ):
        async def no_sleep(_delay):
            return None

        monkeypatch.setattr("src.execution.fill_resolver.asyncio.sleep", no_sleep)
        client.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 0.0015,
                "side": "long",
                "entryPrice": None,
            }
        ]
        client.fetch_ticker.return_value = {
            "last": 65906.0,
            "bid": 65905.0,
            "ask": 65907.0,
        }

        result = await resolver.position_entry_price(
            "BTCUSDT",
            0.0015,
            "long",
            attempts=1,
            allow_market_proxy=True,
        )

        assert result == 65907.0

    @pytest.mark.asyncio
    async def test_short_position_proxy_uses_executable_bid(
        self,
        resolver,
        client,
    ):
        client.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 0.0015,
                "side": "short",
                "entryPrice": None,
            }
        ]
        client.fetch_ticker.return_value = {
            "last": 65906.0,
            "bid": 65905.0,
            "ask": 65907.0,
        }

        result = await resolver.position_entry_price(
            "BTCUSDT",
            0.0015,
            "short",
            attempts=1,
            allow_market_proxy=True,
        )

        assert result == 65905.0

    @pytest.mark.asyncio
    async def test_returns_none_when_no_price_source(self, resolver, client):
        order = {"id": "o1", "filled": 1.0, "average": None}
        client.fetch_order.side_effect = LookupError("not found")
        result = await resolver.resolve_entry_fill_price(order, "BTCUSDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_order_has_no_id(self, resolver, client):
        order = {"filled": 1.0, "average": None}
        result = await resolver.resolve_entry_fill_price(order, "BTCUSDT")
        assert result is None


class TestResolveExitPrice:
    @pytest.mark.asyncio
    async def test_uses_order_price_when_available(self, resolver, client):
        order = {"id": "o1", "average": 101.5}
        result = await resolver.resolve_exit_price(order, "BTCUSDT", 100.0)
        assert result == 101.5

    @pytest.mark.asyncio
    async def test_falls_back_to_market_price(self, resolver, client):
        order = {"id": "o1", "average": None}
        result = await resolver.resolve_exit_price(order, "BTCUSDT", 100.0)
        assert result == 101.0

    @pytest.mark.asyncio
    async def test_uses_fallback_when_market_unavailable(self, resolver, client):
        order = {"id": "o1", "average": None}
        client.fetch_ticker.side_effect = Exception("no connection")
        result = await resolver.resolve_exit_price(order, "BTCUSDT", 100.0)
        assert result == 100.0


class TestPositiveOrderPrice:
    def test_average_takes_priority(self, resolver):
        order = {"id": "o1", "average": 100.5, "price": 99.0}
        assert resolver.positive_order_price(order) == 100.5

    def test_returns_none_when_no_positive_price(self, resolver):
        order = {"id": "o1", "average": 0, "price": -1}
        assert resolver.positive_order_price(order) is None

    def test_checks_info_fields(self, resolver):
        order = {"id": "o1", "info": {"avgPrice": 102.0}}
        assert resolver.positive_order_price(order) == 102.0

    def test_calculates_average_from_quote_cost(self, resolver):
        order = {"filled": 2.0, "cost": 201.0, "average": None}
        assert resolver.positive_order_price(order) == 100.5

    def test_calculates_average_from_raw_binance_quote_cost(self, resolver):
        order = {"info": {"executedQty": "2.0", "cumQuote": "201.0"}}
        assert resolver.positive_order_price(order) == 100.5

    def test_checks_stop_price_fallback(self, resolver):
        order = {"id": "o1", "stopPrice": 103.0}
        assert resolver.positive_order_price(order) == 103.0


class TestOrderIsFilled:
    def test_filled_when_status_closed(self, resolver):
        assert resolver.order_is_filled({"status": "closed"})

    def test_filled_when_status_triggered(self, resolver):
        assert resolver.order_is_filled({"status": "triggered"})

    def test_not_filled_when_status_open(self, resolver):
        assert not resolver.order_is_filled({"status": "open"})

    def test_not_filled_when_order_is_none(self, resolver):
        assert not resolver.order_is_filled(None)

    def test_checks_algoStatus_in_info(self, resolver):
        assert resolver.order_is_filled({"info": {"algoStatus": "finished"}})


class TestEntryFillSlippageBps:
    def test_zero_drift_returns_zero(self, resolver):
        assert resolver.entry_fill_slippage_bps(100.0, 100.0) == 0.0

    def test_positive_drift(self, resolver):
        assert resolver.entry_fill_slippage_bps(100.0, 101.0) == pytest.approx(100.0)

    def test_negative_drift_is_absolute(self, resolver):
        assert resolver.entry_fill_slippage_bps(100.0, 99.0) == pytest.approx(100.0)

    def test_inf_when_ref_price_zero(self, resolver):
        assert resolver.entry_fill_slippage_bps(0, 100.0) == float("inf")


class TestLatestExitOrder:
    @pytest.mark.asyncio
    async def test_finds_matching_exit_order(self, resolver, client):
        trade = MagicMock(symbol="BTCUSDT", side="long", quantity=1.0)
        trade.timestamp.timestamp.return_value = 1000.0
        client.fetch_orders.return_value = [
            {
                "id": "o1",
                "status": "closed",
                "side": "sell",
                "filled": 1.0,
                "timestamp": 1000000,
                "lastUpdateTimestamp": 1000001,
            },
        ]
        result = await resolver.latest_exit_order(trade)
        assert result is not None
        assert result["id"] == "o1"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_matching_order(self, resolver, client):
        trade = MagicMock(symbol="BTCUSDT", side="long", quantity=1.0)
        trade.timestamp.timestamp.return_value = 1000.0
        client.fetch_orders.return_value = []
        result = await resolver.latest_exit_order(trade)
        assert result is None


class TestQuantityTolerance:
    @pytest.mark.asyncio
    async def test_from_market_precision_float(self, resolver, client):
        client.fetch_market.return_value = {"precision": {"amount": 0.001}}
        result = await resolver.quantity_tolerance("BTCUSDT")
        assert result == 0.0005

    @pytest.mark.asyncio
    async def test_from_market_precision_int(self, resolver, client):
        client.fetch_market.return_value = {"precision": {"amount": 3}}
        result = await resolver.quantity_tolerance("BTCUSDT")
        assert result == 0.0005

    @pytest.mark.asyncio
    async def test_fallback_when_market_unavailable(self, resolver, client):
        client.fetch_market.side_effect = Exception("no market")
        result = await resolver.quantity_tolerance("BTCUSDT")
        assert result == 1e-12


class TestOrderFee:
    @pytest.mark.asyncio
    async def test_resolves_binance_commission_from_trade_history(
        self,
        resolver,
        client,
    ):
        client.fetch_my_trades.return_value = [
            {
                "order": "o1",
                "amount": 1.0,
                "price": 100.0,
                "fee": {"cost": 0.03, "currency": "USDT"},
                "info": {"orderId": "o1", "commission": "0.03"},
            },
            {
                "order": "other",
                "fee": {"cost": 99.0, "currency": "USDT"},
            },
        ]

        fee = await resolver.order_fee({"id": "o1"}, "BTCUSDT")

        assert fee == pytest.approx(0.03)

    def test_raw_commission_is_used_when_ccxt_fee_is_absent(self, resolver):
        assert resolver.execution_fee(
            {"info": {"commission": "0.0125", "commissionAsset": "USDT"}}
        ) == pytest.approx(0.0125)

    @pytest.mark.asyncio
    async def test_estimates_commission_when_exchange_history_has_no_fee(
        self,
        resolver,
        client,
    ):
        client.fetch_my_trades.return_value = []

        fee = await resolver.order_fee(
            {"id": "o1"},
            "BTCUSDT",
            fallback_notional=100.0,
            fallback_fee_bps=8.0,
        )

        assert fee == pytest.approx(0.08)
