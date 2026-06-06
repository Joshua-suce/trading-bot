import pytest

from src.exchange.client import ExchangeClient


class FakeRest:
    def __init__(self, markets):
        self.markets = markets

    async def load_markets(self):
        return self.markets


def build_client(markets):
    client = ExchangeClient()
    client._rest = FakeRest(markets)
    return client


@pytest.mark.asyncio
async def test_fetch_market_resolves_binance_raw_symbol():
    market = {
        "id": "BTCUSDT",
        "symbol": "BTC/USDT:USDT",
        "precision": {"amount": 3},
    }
    client = build_client({"BTC/USDT:USDT": market})

    assert await client.fetch_market("BTCUSDT") == market


@pytest.mark.asyncio
async def test_fetch_market_accepts_unified_symbol_key():
    market = {
        "id": "ETHUSDT",
        "symbol": "ETH/USDT:USDT",
        "precision": {"amount": 3},
    }
    client = build_client({"ETH/USDT:USDT": market})

    assert await client.fetch_market("ETH/USDT:USDT") == market


@pytest.mark.asyncio
async def test_fetch_market_reports_unknown_market_clearly():
    client = build_client({})

    with pytest.raises(ValueError, match="Market metadata not found for SOLUSDT"):
        await client.fetch_market("SOLUSDT")
