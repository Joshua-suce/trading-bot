# Account information helpers — fetch balances, positions, and trade state
from dataclasses import dataclass
from typing import Optional

from src.exchange.client import ExchangeClient


# Snapshot of the current account state
@dataclass
class AccountInfo:
    total_equity: float
    wallet_balance: float
    available_balance: float
    unrealized_pnl: float
    margin_ratio: float

    # Quick health check — margin ratio below 80% is considered safe
    def is_healthy(self, max_margin_ratio: float = 0.8) -> bool:
        return self.margin_ratio < max_margin_ratio


# Fetch and parse account info from the Binance REST API
async def get_account_info(client: ExchangeClient) -> AccountInfo:
    balance = await client.rest.fetch_balance()
    info = balance["info"]
    return AccountInfo(
        total_equity=float(info.get("totalWalletBalance", 0))
        + float(info.get("totalUnrealizedProfit", 0)),
        wallet_balance=float(info.get("totalWalletBalance", 0)),
        available_balance=float(balance["USDT"].get("free", 0)),
        unrealized_pnl=float(info.get("totalUnrealizedProfit", 0)),
        margin_ratio=float(info.get("totalMaintMargin", 0))
        / max(float(info.get("totalMarginBalance", 1)), 1),
    )


# Get the position dict for a symbol (returns None if no position)
async def get_position(client: ExchangeClient, symbol: str) -> Optional[dict]:
    positions = await client.fetch_positions(symbol)
    for pos in positions:
        if pos["symbol"] == symbol and abs(float(pos.get("contracts", 0))) > 0:
            return pos
    return None


# Check whether there is an open position for the given symbol
async def has_open_position(client: ExchangeClient, symbol: str) -> bool:
    return await get_position(client, symbol) is not None


# Return the size (number of contracts) of an open position
async def get_position_size(client: ExchangeClient, symbol: str) -> float:
    pos = await get_position(client, symbol)
    return float(pos["contracts"]) if pos else 0.0


# Return the side ("long" or "short") of an open position
async def get_position_side(client: ExchangeClient, symbol: str) -> Optional[str]:
    pos = await get_position(client, symbol)
    return pos["side"] if pos else None
