import pytest

from src.exchange.client import ExchangeClient


class FakeRest:
    def __init__(self):
        self.calls = []

    async def fetch_order(self, order_id, symbol, params):
        self.calls.append((order_id, symbol, params))
        return {"id": "resolved"}


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
