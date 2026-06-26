# Account information helpers — fetch balances, positions, and trade state
from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional

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


class InvalidAccountSnapshotError(RuntimeError):
    pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def _balance_account(balance: dict, currency: str = "USDT") -> dict:
    entry = balance.get(currency)
    return entry if isinstance(entry, dict) else {}


def _raw_balance_asset(info: dict, currency: str = "USDT") -> dict:
    for key in ("assets", "balances", "userAssets"):
        rows = info.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and str(row.get("asset")) == currency:
                return row
    return {}


def _first_present(*values: object) -> object:
    return next((value for value in values if value not in (None, "")), None)


def _validate_account_values(
    *,
    equity: float,
    wallet: float,
    available: float,
    upnl: float,
    margin_ratio: float,
    recognized: bool,
) -> None:
    values = (equity, wallet, available, upnl, margin_ratio)
    if not all(isfinite(value) for value in values):
        raise InvalidAccountSnapshotError(
            "Binance account snapshot contains non-finite numeric values"
        )
    if equity < 0 or wallet < 0 or available < 0:
        raise InvalidAccountSnapshotError(
            "Binance account snapshot contains negative balance values"
        )
    if not recognized:
        raise InvalidAccountSnapshotError(
            "Binance account snapshot does not contain a USDT futures balance"
        )


# Fetch and parse account info from the Binance REST API
async def get_account_info(client: ExchangeClient) -> AccountInfo:
    balance = await client.fetch_balance()
    if not isinstance(balance, dict):
        raise InvalidAccountSnapshotError(
            "Binance returned an invalid account balance payload"
        )
    info = balance.get("info", {})
    if not isinstance(info, dict):
        info = {}
    asset = _raw_balance_asset(info)
    account = _balance_account(balance)

    # Binance demo accounts sometimes have balance data scattered across
    # different fields. Raw futures fields distinguish wallet balance from
    # margin balance; CCXT total is margin balance and must not have uPnL added.
    total = balance.get("total", {}) or {}
    total_usdt = total.get("USDT") if isinstance(total, dict) else None
    wallet_src = _first_present(
        info.get("totalWalletBalance"),
        asset.get("walletBalance"),
        asset.get("crossWalletBalance"),
        total_usdt,
    )
    wallet = _safe_float(wallet_src)

    upnl_src = _first_present(
        info.get("totalUnrealizedProfit"),
        asset.get("unrealizedProfit"),
    )
    upnl = _safe_float(upnl_src)

    free = balance.get("free") or {}
    free_usdt = free.get("USDT") if isinstance(free, dict) else None
    avail_src = _first_present(
        account.get("free"),
        free_usdt,
        info.get("availableBalance"),
        asset.get("availableBalance"),
        asset.get("maxWithdrawAmount"),
    )
    available = _safe_float(avail_src)

    equity_src = _first_present(
        info.get("totalMarginBalance"),
        asset.get("marginBalance"),
        total_usdt,
    )
    equity = _safe_float(equity_src, wallet + upnl)

    maint = _safe_float(info.get("totalMaintMargin"))
    if not maint:
        maint = _safe_float(asset.get("maintMargin"))
    margin = _safe_float(equity_src, equity)
    margin_ratio = maint / max(margin, 1.0) if margin > 0 else 0.0

    recognized = any(
        value not in (None, "")
        for value in (
            info.get("totalWalletBalance"),
            info.get("totalMarginBalance"),
            asset.get("walletBalance"),
            asset.get("marginBalance"),
            total_usdt,
            account.get("total"),
        )
    )
    _validate_account_values(
        equity=equity,
        wallet=wallet,
        available=available,
        upnl=upnl,
        margin_ratio=margin_ratio,
        recognized=recognized,
    )

    return AccountInfo(
        total_equity=equity,
        wallet_balance=wallet,
        available_balance=available,
        unrealized_pnl=upnl,
        margin_ratio=margin_ratio,
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
