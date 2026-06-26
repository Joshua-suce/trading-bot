import asyncio
from typing import Any

from loguru import logger

from src.config import settings
from src.exchange.client import ExchangeClient
from src.security import redact_text


class FillResolver:
    def __init__(self, client: ExchangeClient):
        self.client = client

    async def find_order_in_trade_history(
        self,
        symbol: str,
        order_id: str,
    ) -> float | None:
        try:
            trades = await self.client.fetch_my_trades(symbol, order_id=order_id)
        except Exception as exc:
            logger.debug(
                f"Could not fetch trade history for {symbol} order {order_id}: {exc}"
            )
            return None
        if not trades:
            return None
        total_qty = 0.0
        total_value = 0.0
        for t in trades:
            qty = float(t.get("amount") or t.get("info", {}).get("qty") or 0)
            price = float(t.get("price") or t.get("info", {}).get("price") or 0)
            if qty > 0 and price > 0:
                total_qty += qty
                total_value += qty * price
        if total_qty <= 0:
            return None
        return total_value / total_qty

    async def resolved_order(
        self,
        symbol: str,
        order_id: str,
        *,
        reason: str,
        conditional: bool,
    ) -> dict | None:
        try:
            return await self.client.fetch_order(
                order_id, symbol, conditional=conditional
            )
        except Exception as exc:
            logger.warning(
                f"Could not fetch {reason} order {order_id} for {symbol}: {exc}"
            )
            return await self.historical_order(
                symbol, order_id, conditional=conditional
            )

    @staticmethod
    def order_is_filled(order: dict | None) -> bool:
        if order is None:
            return False
        info = order.get("info") or {}
        statuses = {
            str(order.get("status", "")).lower(),
            str(order.get("algoStatus", "")).lower(),
            str(info.get("algoStatus", "")).lower(),
            str(info.get("status", "")).lower(),
        }
        return bool(statuses & {"closed", "filled", "finished", "triggered"})

    async def historical_order(
        self, symbol: str, order_id: str, *, conditional: bool
    ) -> dict | None:
        try:
            orders = await self.client.fetch_orders(symbol, conditional=conditional)
        except Exception as exc:
            logger.warning(f"Could not fetch order history for {symbol}: {exc}")
            return None
        return next(
            (order for order in orders if str(order.get("id", "")) == str(order_id)),
            None,
        )

    async def latest_exit_order(
        self, trade, skip_order_ids: set[str] | None = None
    ) -> dict | None:
        try:
            orders = await self.client.fetch_orders(trade.symbol)
        except Exception as exc:
            logger.warning(f"Could not fetch exit history for {trade.symbol}: {exc}")
            return None
        expected_side = "sell" if trade.side == "long" else "buy"
        opened_ms = int(trade.timestamp.timestamp() * 1000)
        tolerance = await self.quantity_tolerance(trade.symbol)
        candidates = []
        for order in orders:
            if str(order.get("status", "")).lower() not in {"closed", "filled"}:
                continue
            order_id = str(order.get("id") or "")
            if order_id in (skip_order_ids or set()):
                continue
            if str(order.get("side", "")).lower() != expected_side:
                continue
            timestamp = int(order.get("timestamp") or 0)
            if timestamp and timestamp < opened_ms:
                continue
            filled = float(order.get("filled") or 0)
            if filled <= 0:
                continue
            if abs(filled - trade.quantity) > tolerance:
                continue
            candidates.append(order)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda order: int(
                order.get("lastUpdateTimestamp") or order.get("timestamp") or 0
            ),
        )

    async def resolve_exit_price(
        self, order: dict, symbol: str, fallback: float
    ) -> float:
        price = self.positive_order_price(order)
        if price is not None:
            return price
        return await self.market_price(symbol, fallback)

    async def market_price(
        self,
        symbol: str,
        fallback: float,
        *,
        side: str | None = None,
    ) -> float:
        try:
            ticker = await self.client.fetch_ticker(symbol)
            if side == "long":
                price_keys = ("ask", "last", "mark", "index", "bid")
            elif side == "short":
                price_keys = ("bid", "last", "mark", "index", "ask")
            else:
                price_keys = ("last", "mark", "index", "bid", "ask")
            for key in price_keys:
                value = ticker.get(key)
                if value is not None and float(value) > 0:
                    return float(value)
        except Exception as exc:
            logger.warning(f"Could not resolve market price for {symbol}: {exc}")
        return fallback

    async def resolve_entry_fill_price(self, order: dict, symbol: str) -> float | None:
        price = self.positive_order_price(order)
        if price is not None:
            return price
        order_id = str(order.get("id") or "")
        if not order_id:
            return None
        expected_quantity = float(order.get("filled") or 0)
        expected_side = self._entry_position_side(order)
        position_price = await self.position_entry_price(
            symbol,
            expected_quantity,
            expected_side,
        )
        if position_price is not None:
            logger.warning(
                "Recovered {} entry fill price from exchange position snapshot",
                symbol,
            )
            return position_price
        execution_price = await self.entry_execution_price(
            symbol,
            order_id,
            expected_quantity,
        )
        if execution_price is not None:
            return execution_price
        resolved = await self.resolved_order(
            symbol,
            order_id,
            reason="entry",
            conditional=False,
        )
        if resolved is not None:
            resolved_price = self.positive_order_price(resolved)
            if resolved_price is not None:
                return resolved_price

        # Binance demo can report a market order as FILLED while temporarily
        # omitting it from both order and execution history. Recheck the exact
        # resulting position after those slower ledgers have had time to settle.
        position_price = await self.position_entry_price(
            symbol,
            expected_quantity,
            expected_side,
            attempts=settings.entry_fill_resolution_attempts,
            allow_market_proxy=True,
        )
        if position_price is not None:
            logger.warning(
                "Recovered delayed {} entry price from exchange position",
                symbol,
            )
        return position_price

    async def position_entry_price(
        self,
        symbol: str,
        expected_quantity: float,
        expected_side: str | None,
        *,
        attempts: int | None = None,
        allow_market_proxy: bool = False,
    ) -> float | None:
        fetch_positions = getattr(self.client, "fetch_positions", None)
        if not callable(fetch_positions) or expected_quantity <= 0:
            return None
        tolerance = await self.quantity_tolerance(symbol)
        resolution_attempts = attempts or min(
            settings.entry_fill_resolution_attempts,
            3,
        )
        for attempt in range(1, resolution_attempts + 1):
            try:
                positions = await fetch_positions(symbol)
            except Exception as exc:
                logger.warning(
                    "Could not fetch position snapshot for {} entry fill: {}",
                    symbol,
                    redact_text(exc),
                )
                return None
            for position in positions:
                matched, price = self._matching_position_price(
                    position,
                    symbol,
                    expected_quantity,
                    expected_side,
                    tolerance,
                )
                if not matched:
                    continue
                if price is not None:
                    return price
                if allow_market_proxy:
                    proxy_price = await self.market_price(
                        symbol,
                        0.0,
                        side=expected_side,
                    )
                    if proxy_price > 0:
                        logger.warning(
                            "Exchange position for {} is confirmed but entryPrice "
                            "is absent; using a live quote subject to slippage "
                            "validation",
                            symbol,
                        )
                        return proxy_price
            if attempt < resolution_attempts:
                await asyncio.sleep(
                    settings.entry_fill_resolution_backoff_seconds * attempt
                )
        return None

    @classmethod
    def _matching_position_price(
        cls,
        position: dict,
        symbol: str,
        expected_quantity: float,
        expected_side: str | None,
        tolerance: float,
    ) -> tuple[bool, float | None]:
        if cls._normalize_symbol(position.get("symbol")) != cls._normalize_symbol(
            symbol
        ):
            return False, None
        quantity = abs(cls._position_quantity(position))
        if abs(quantity - expected_quantity) > tolerance:
            return False, None
        side = cls._position_side(position)
        if expected_side and side and side != expected_side:
            return False, None
        return True, cls._position_entry_price(position)

    async def entry_execution_price(
        self,
        symbol: str,
        order_id: str,
        expected_quantity: float,
    ) -> float | None:
        fill = await self.execution_fill_details(
            symbol,
            order_id,
            expected_quantity,
        )
        return fill[1] if fill is not None else None

    async def execution_fill_details(
        self,
        symbol: str,
        order_id: str,
        expected_quantity: float,
    ) -> tuple[float, float] | None:
        fetch_my_trades = getattr(self.client, "fetch_my_trades", None)
        if not callable(fetch_my_trades):
            return None
        matched: list[tuple[float, float]] = []
        attempts = settings.entry_fill_resolution_attempts
        for attempt in range(1, attempts + 1):
            try:
                trades = await fetch_my_trades(symbol, order_id=order_id)
            except Exception as exc:
                logger.warning(
                    "Could not fetch trade executions for {} order {}: {}",
                    symbol,
                    order_id,
                    redact_text(exc),
                )
                return None

            matched = self.matched_trade_executions(trades, order_id)
            if matched:
                filled_quantity = sum(amount for amount, _ in matched)
                tolerance = await self.quantity_tolerance(symbol)
                if (
                    expected_quantity <= 0
                    or filled_quantity >= expected_quantity - tolerance
                ):
                    return (
                        filled_quantity,
                        sum(amount * price for amount, price in matched)
                        / filled_quantity,
                    )
            if attempt < attempts:
                await asyncio.sleep(
                    settings.entry_fill_resolution_backoff_seconds * attempt
                )

        if matched:
            logger.warning(
                "Incomplete trade executions for {} order {}: {} < {}",
                symbol,
                order_id,
                sum(amount for amount, _ in matched),
                expected_quantity,
            )
        return None

    @staticmethod
    def matched_trade_executions(
        trades: list[dict],
        order_id: str,
    ) -> list[tuple[float, float]]:
        matched = []
        for trade in trades:
            info = trade.get("info") or {}
            trade_order_id = str(trade.get("order") or info.get("orderId") or "")
            if trade_order_id != order_id:
                continue
            amount = float(
                trade.get("amount") or info.get("qty") or info.get("quantity") or 0
            )
            price = float(trade.get("price") or info.get("price") or 0)
            if amount > 0 and price > 0:
                matched.append((amount, price))
        return matched

    @staticmethod
    def execution_fee(trade: dict) -> float:
        structured_costs = []
        fee = trade.get("fee")
        if isinstance(fee, dict):
            structured_costs.append(fee.get("cost"))
        for item in trade.get("fees") or []:
            if isinstance(item, dict):
                structured_costs.append(item.get("cost"))
        resolved = [
            value
            for cost in structured_costs
            if (value := FillResolver._positive_float(cost)) is not None
        ]
        if resolved:
            return sum(resolved)
        info = trade.get("info") or {}
        return FillResolver._positive_float(info.get("commission")) or 0.0

    async def order_fee(
        self,
        order: dict | None,
        symbol: str,
        *,
        fallback_notional: float = 0.0,
        fallback_fee_bps: float = 0.0,
    ) -> float:
        fallback = (
            max(float(fallback_notional), 0.0)
            * max(float(fallback_fee_bps), 0.0)
            / 10_000
        )
        if not order:
            return fallback
        payload_fee = self.execution_fee(order)
        if payload_fee > 0:
            return payload_fee
        info = order.get("info") or {}
        order_id = str(order.get("id") or info.get("orderId") or "")
        fetch_my_trades = getattr(self.client, "fetch_my_trades", None)
        if not order_id or not callable(fetch_my_trades):
            return fallback
        try:
            trades = await fetch_my_trades(symbol, order_id=order_id)
        except Exception as exc:
            logger.warning(
                "Could not resolve commission for {} order {}: {}",
                symbol,
                order_id,
                redact_text(exc),
            )
            return fallback
        resolved = sum(
            self.execution_fee(trade)
            for trade in trades
            if str(trade.get("order") or (trade.get("info") or {}).get("orderId") or "")
            == order_id
        )
        return resolved if resolved > 0 else fallback

    @staticmethod
    def positive_order_price(order: dict) -> float | None:
        info = order.get("info") or {}
        for value in (
            order.get("average"),
            order.get("price"),
            info.get("avgPrice"),
            info.get("price"),
            order.get("stopPrice"),
            info.get("stopPrice"),
            order.get("triggerPrice"),
            info.get("triggerPrice"),
        ):
            if value is not None and float(value) > 0:
                return float(value)
        filled = FillResolver._positive_float(
            order.get("filled") or info.get("executedQty") or info.get("cumQty")
        )
        cost = FillResolver._positive_float(
            order.get("cost")
            or info.get("cumQuote")
            or info.get("cumQuoteQty")
            or info.get("quoteQty")
        )
        if filled is not None and cost is not None:
            return cost / filled
        return None

    @staticmethod
    def _entry_position_side(order: dict) -> str | None:
        side = str(order.get("side") or (order.get("info") or {}).get("side") or "")
        side = side.lower()
        if side == "buy":
            return "long"
        if side == "sell":
            return "short"
        return None

    @staticmethod
    def _position_quantity(position: dict) -> float:
        info = position.get("info") or {}
        for value in (
            position.get("contracts"),
            info.get("positionAmt"),
            info.get("positionAmount"),
        ):
            number = FillResolver._signed_float(value)
            if number is not None:
                return number
        return 0.0

    @staticmethod
    def _position_side(position: dict) -> str | None:
        side = str(position.get("side") or "").lower()
        if side in {"long", "short"}:
            return side
        quantity = FillResolver._position_quantity(position)
        if quantity > 0:
            return "long"
        if quantity < 0:
            return "short"
        return None

    @staticmethod
    def _position_entry_price(position: dict) -> float | None:
        info = position.get("info") or {}
        for value in (
            position.get("entryPrice"),
            position.get("entry_price"),
            info.get("entryPrice"),
            info.get("breakEvenPrice"),
        ):
            number = FillResolver._positive_float(value)
            if number is not None:
                return number
        return None

    @staticmethod
    def _normalize_symbol(symbol: object) -> str:
        return (
            str(symbol or "").split(":", 1)[0].replace("/", "").replace("-", "").upper()
        )

    @staticmethod
    def _positive_float(value: Any) -> float | None:
        number = FillResolver._signed_float(value)
        return number if number is not None and number > 0 else None

    @staticmethod
    def _signed_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def entry_fill_slippage_bps(ref_price: float, fill_price: float) -> float:
        if ref_price <= 0 or fill_price <= 0:
            return float("inf")
        return abs(fill_price - ref_price) / ref_price * 10_000

    async def quantity_tolerance(self, symbol: str) -> float:
        try:
            market = await self.client.fetch_market(symbol)
            precision = market.get("precision", {}).get("amount")
            if isinstance(precision, float) and precision > 0:
                return precision / 2
            if isinstance(precision, int) and precision >= 0:
                return 10 ** (-precision) / 2
        except Exception as exc:
            logger.debug(f"Could not resolve quantity tolerance for {symbol}: {exc}")
        return 1e-12

    async def confirmed_full_fill(
        self, order: dict | None, symbol: str, expected_quantity: float
    ) -> dict | None:
        if not order:
            return None
        tolerance = await self.quantity_tolerance(symbol)
        if float(order.get("filled") or 0) >= expected_quantity - tolerance:
            return order

        order_id = str(order.get("id") or "")
        if not order_id:
            return None
        try:
            refreshed = await self.client.fetch_order(order_id, symbol)
        except Exception as exc:
            logger.warning(f"Could not confirm exit fill for {symbol}: {exc}")
            refreshed = None
        if (
            refreshed
            and float(refreshed.get("filled") or 0) >= expected_quantity - tolerance
        ):
            return refreshed
        execution_fill = await self.execution_fill_details(
            symbol,
            order_id,
            expected_quantity,
        )
        if execution_fill is not None:
            filled_quantity, average_price = execution_fill
            return {
                **order,
                "filled": filled_quantity,
                "average": average_price,
                "status": "closed",
            }
        return None
