from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from src.config import settings


@dataclass(frozen=True)
class PreparedOrder:
    allowed: bool
    reason: str
    quantity: float = 0.0
    price: float | None = None
    reference_price: float | None = None
    slippage_bps: float = 0.0
    notional: float = 0.0


class ExecutionGuard:
    def __init__(
        self,
        *,
        max_slippage_bps: float | None = None,
        min_notional: float | None = None,
    ) -> None:
        self.max_slippage_bps = (
            settings.max_entry_slippage_bps
            if max_slippage_bps is None
            else max_slippage_bps
        )
        self.min_notional = (
            settings.min_order_notional if min_notional is None else min_notional
        )

    async def prepare(
        self,
        client: Any,
        *,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None,
        params: dict,
    ) -> PreparedOrder:
        market = await client.fetch_market(symbol)
        prepared_quantity = self._amount_to_precision(market, quantity)
        prepared_price = self._price_to_precision(market, price)
        reference_price = prepared_price or await self._reference_price(client, symbol)
        notional = prepared_quantity * reference_price
        reduce_only = bool(params.get("reduceOnly"))

        minimum_amount = self._market_min_amount(market)
        if prepared_quantity <= 0:
            return PreparedOrder(
                False,
                f"amount below minimum: {prepared_quantity:.8f} < "
                f"{minimum_amount:.8f}",
                quantity=prepared_quantity,
                price=prepared_price,
                reference_price=reference_price,
                notional=notional,
            )
        if not reduce_only and prepared_quantity < minimum_amount:
            return PreparedOrder(
                False,
                f"amount below minimum: {prepared_quantity:.8f} < "
                f"{minimum_amount:.8f}",
                quantity=prepared_quantity,
                price=prepared_price,
                reference_price=reference_price,
                notional=notional,
            )

        min_notional = self._market_min_notional(market)
        required_notional = max(self.min_notional, min_notional)
        if not reduce_only and notional < required_notional:
            return PreparedOrder(
                False,
                f"notional below minimum: {notional:.8f} < {required_notional:.8f}",
                quantity=prepared_quantity,
                price=prepared_price,
                reference_price=reference_price,
                notional=notional,
            )

        if order_type == "market" and not reduce_only:
            slippage = await self._market_slippage_bps(client, symbol, side)
            if slippage > self.max_slippage_bps:
                return PreparedOrder(
                    False,
                    f"slippage above limit: {slippage:.2f}bps",
                    quantity=prepared_quantity,
                    price=prepared_price,
                    reference_price=reference_price,
                    slippage_bps=slippage,
                    notional=notional,
                )

        return PreparedOrder(
            True,
            "ok",
            quantity=prepared_quantity,
            price=prepared_price,
            reference_price=reference_price,
            notional=notional,
        )

    @staticmethod
    def _amount_to_precision(market: dict, quantity: float) -> float:
        precision = market.get("precision", {}).get("amount")
        return ExecutionGuard._round_down(quantity, precision)

    @staticmethod
    def _price_to_precision(market: dict, price: float | None) -> float | None:
        if price is None:
            return None
        precision = market.get("precision", {}).get("price")
        return ExecutionGuard._round_down(price, precision)

    @staticmethod
    def _round_down(value: float, precision: int | float | None) -> float:
        if precision is None:
            return float(value)
        decimal_precision = Decimal(str(precision))
        if decimal_precision <= 0:
            return float(value)
        if isinstance(precision, float):
            value_decimal = Decimal(str(value))
            steps = (value_decimal / decimal_precision).to_integral_value(
                rounding=ROUND_DOWN
            )
            return float(steps * decimal_precision)
        quant = Decimal("1").scaleb(-int(decimal_precision))
        return float(Decimal(str(value)).quantize(quant, rounding=ROUND_DOWN))

    @staticmethod
    def _market_min_notional(market: dict) -> float:
        limits = market.get("limits", {})
        cost = limits.get("cost", {}) if isinstance(limits, dict) else {}
        return float(cost.get("min") or 0)

    @staticmethod
    def _market_min_amount(market: dict) -> float:
        limits = market.get("limits", {})
        amount = limits.get("amount", {}) if isinstance(limits, dict) else {}
        return float(amount.get("min") or 0)

    @staticmethod
    async def _reference_price(client: Any, symbol: str) -> float:
        ticker = await client.fetch_ticker(symbol)
        for key in ("last", "mark", "index", "ask", "bid"):
            value = ticker.get(key)
            if value:
                return float(value)
        raise RuntimeError(f"No reference price available for {symbol}")

    @staticmethod
    async def _market_slippage_bps(client: Any, symbol: str, side: str) -> float:
        ticker = await client.fetch_ticker(symbol)
        bid = float(ticker.get("bid") or 0)
        ask = float(ticker.get("ask") or 0)
        last = float(ticker.get("last") or ticker.get("mark") or 0)
        if bid <= 0 or ask <= 0 or last <= 0:
            return 0.0
        execution_price = ask if side == "buy" else bid
        return abs(execution_price - last) / last * 10_000
