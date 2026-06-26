import asyncio

from loguru import logger

from src.config import settings


class EntryOrderRouter:
    PASSIVE_STRATEGIES = {"trend", "range", "transition"}

    def __init__(self, client, orders) -> None:
        self.client = client
        self.orders = orders

    async def submit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        strategy: str | None,
    ) -> dict | None:
        if (
            not settings.adaptive_limit_entry_enabled
            or strategy not in self.PASSIVE_STRATEGIES
        ):
            return await self.orders.market_order(symbol, side, quantity)
        quote = await self.client.fetch_ticker(symbol)
        price_key = "bid" if side == "buy" else "ask"
        price = float(quote.get(price_key) or 0.0)
        if price <= 0:
            logger.info(f"Passive quote unavailable for {symbol}; using market entry")
            return await self.orders.market_order(symbol, side, quantity)
        order = await self.orders.limit_order(
            symbol,
            side,
            quantity,
            price,
            post_only=True,
        )
        if not order:
            return await self._market_fallback(symbol, side, quantity)
        resolved = await self._wait_for_fill(symbol, order)
        filled = float(resolved.get("filled") or 0.0)
        if filled > 0:
            if filled + 1e-12 < quantity and str(
                resolved.get("status") or ""
            ).lower() not in {"closed", "canceled", "cancelled"}:
                await self.orders.cancel_order(symbol, str(order.get("id") or ""))
            resolved["_bot_order_policy"] = "passive_limit"
            return resolved
        await self.orders.cancel_order(symbol, str(order.get("id") or ""))
        return await self._market_fallback(symbol, side, quantity)

    async def _wait_for_fill(self, symbol: str, order: dict) -> dict:
        if float(order.get("filled") or 0.0) > 0:
            return order
        order_id = str(order.get("id") or "")
        if not order_id:
            return order
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.limit_entry_timeout_seconds
        current = order
        while loop.time() < deadline:
            await asyncio.sleep(settings.limit_entry_poll_seconds)
            current = await self.client.fetch_order(order_id, symbol)
            if float(current.get("filled") or 0.0) > 0:
                return current
            if str(current.get("status") or "").lower() in {
                "closed",
                "canceled",
                "cancelled",
                "rejected",
                "expired",
            }:
                return current
        return current

    async def _market_fallback(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict | None:
        if not settings.limit_entry_market_fallback:
            return None
        order = await self.orders.market_order(symbol, side, quantity)
        if order:
            order["_bot_order_policy"] = "market_fallback"
        return order
