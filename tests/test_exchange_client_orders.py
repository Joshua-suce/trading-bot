import pytest

from src.exchange.client import ExchangeClient


class FakeRest:
    def __init__(self):
        self.calls = []

    async def fetch_order(self, order_id, symbol, params):
        self.calls.append((order_id, symbol, params))
        return {"id": "resolved"}

    async def fetch_my_trades(self, symbol, since=None, limit=None, params=None):
        self.calls.append((symbol, since, limit, params))
        return [{"order": "123", "price": 100.0}]


@pytest.mark.asyncio
async def test_fetch_standard_order_by_client_id():
    client = ExchangeClient()
    client._rest = FakeRest()

    order = await client.fetch_order_by_client_id("tb_ord_123", "BTCUSDT")

    assert order == {"id": "resolved"}
    assert client.rest.calls == [("", "BTCUSDT", {"origClientOrderId": "tb_ord_123"})]


@pytest.mark.asyncio
async def test_fetch_conditional_order_by_client_id():
    client = ExchangeClient()
    client._rest = FakeRest()

    await client.fetch_order_by_client_id("tb_sl_123", "BTCUSDT", conditional=True)

    assert client.rest.calls == [
        (
            "",
            "BTCUSDT",
            {"trigger": True, "clientAlgoId": "tb_sl_123"},
        )
    ]


@pytest.mark.asyncio
async def test_fetch_trade_executions_by_order_id():
    client = ExchangeClient()
    client._rest = FakeRest()

    trades = await client.fetch_my_trades("BTCUSDT", order_id="123")

    assert trades == [{"order": "123", "price": 100.0}]
    assert client.rest.calls == [("BTCUSDT", None, 100, {"orderId": "123"})]
