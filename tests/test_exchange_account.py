import pytest

from src.exchange.account import InvalidAccountSnapshotError, get_account_info


class FakeClient:
    def __init__(self, balance):
        self._balance = balance

    async def fetch_balance(self):
        return self._balance


@pytest.mark.asyncio
async def test_get_account_info_uses_unified_futures_balance_shape():
    client = FakeClient(
        {
            "info": {
                "totalWalletBalance": "100.0",
                "totalUnrealizedProfit": "2.0",
                "totalMarginBalance": "102.0",
                "totalMaintMargin": "1.5",
                "availableBalance": "98.0",
                "assets": [
                    {
                        "asset": "USDT",
                        "walletBalance": "100.0",
                        "marginBalance": "102.0",
                        "maintMargin": "1.5",
                        "unrealizedProfit": "2.0",
                        "availableBalance": "98.0",
                        "maxWithdrawAmount": "98.0",
                    }
                ],
            },
            "USDT": {"free": 98.0, "used": 1.5, "total": 102.0},
            "free": {"USDT": 98.0},
            "total": {"USDT": 102.0},
        }
    )

    account = await get_account_info(client)

    assert account.wallet_balance == 100.0
    assert account.total_equity == 102.0
    assert account.available_balance == 98.0
    assert account.unrealized_pnl == 2.0


@pytest.mark.asyncio
async def test_get_account_info_falls_back_to_raw_demo_asset_payload():
    client = FakeClient(
        {
            "info": {
                "totalUnrealizedProfit": "1.25",
                "totalMaintMargin": "0.75",
                "assets": [
                    {
                        "asset": "USDT",
                        "walletBalance": "250.0",
                        "marginBalance": "251.25",
                        "maintMargin": "0.75",
                        "unrealizedProfit": "1.25",
                        "maxWithdrawAmount": "249.0",
                    }
                ],
            },
            "info_only": True,
        }
    )

    account = await get_account_info(client)

    assert account.wallet_balance == 250.0
    assert account.total_equity == 251.25
    assert account.available_balance == 249.0
    assert account.unrealized_pnl == 1.25


@pytest.mark.asyncio
async def test_get_account_info_does_not_double_count_unrealized_pnl():
    client = FakeClient(
        {
            "info": {
                "totalWalletBalance": "100.0",
                "totalUnrealizedProfit": "5.0",
                "totalMarginBalance": "105.0",
                "availableBalance": "95.0",
            },
            "USDT": {"free": 95.0, "used": 10.0, "total": 105.0},
            "free": {"USDT": 95.0},
            "total": {"USDT": 105.0},
        }
    )

    account = await get_account_info(client)

    assert account.wallet_balance == 100.0
    assert account.total_equity == 105.0
    assert account.unrealized_pnl == 5.0


@pytest.mark.asyncio
async def test_get_account_info_rejects_empty_balance_payload():
    with pytest.raises(
        InvalidAccountSnapshotError,
        match="does not contain a USDT futures balance",
    ):
        await get_account_info(FakeClient({"info": {}}))
