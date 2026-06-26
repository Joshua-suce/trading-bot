import pytest
from ccxt.base.errors import ExchangeError, ExchangeNotAvailable

from src.exchange.client import BinanceDemoAccountInactiveError, ExchangeClient


@pytest.mark.asyncio
async def test_read_with_retries_translates_inactive_demo_account(monkeypatch):
    client = ExchangeClient()

    async def operation():
        raise ExchangeError(
            'binanceusdm {"code":-4109,"msg":"This account is inactive, '
            'please activate the account first."}'
        )

    monkeypatch.setattr(client, "_is_demo_account_inactive", lambda exc: True)
    monkeypatch.setattr("src.exchange.client.settings.exchange_read_attempts", 1)

    with pytest.raises(BinanceDemoAccountInactiveError, match="inactive"):
        await client._read_with_retries(operation, "test read")


class BalanceRest:
    def __init__(self):
        self.calls = []

    async def fetch_balance(self, params):
        self.calls.append(params)
        if not params.get("useV2"):
            raise ExchangeNotAvailable("demo account v3 unavailable")
        return {"total": {"USDT": 5000.0}}


@pytest.mark.asyncio
async def test_demo_balance_falls_back_to_v2(monkeypatch):
    client = ExchangeClient()
    rest = BalanceRest()
    client._rest = rest
    monkeypatch.setattr(
        "src.exchange.client.settings.binance_api_url",
        "https://demo-fapi.binance.com",
    )

    balance = await client._fetch_balance_snapshot()

    assert balance["total"]["USDT"] == 5000.0
    assert rest.calls == [
        {"type": "swap"},
        {"type": "swap", "useV2": True},
    ]


@pytest.mark.asyncio
async def test_fetch_balance_reuses_recent_authenticated_snapshot(monkeypatch):
    client = ExchangeClient()
    rest = BalanceRest()
    client._rest = rest
    monkeypatch.setattr(
        "src.exchange.client.settings.binance_api_url",
        "https://demo-fapi.binance.com",
    )
    monkeypatch.setattr(
        "src.exchange.client.settings.account_balance_cache_seconds",
        10.0,
    )

    first = await client._fetch_balance_snapshot()
    second = await client.fetch_balance()

    assert second is first
    assert len(rest.calls) == 2
