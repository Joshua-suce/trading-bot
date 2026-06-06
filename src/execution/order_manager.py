import asyncio
import uuid
from typing import Optional

from ccxt.base.errors import OrderNotFound
from loguru import logger

from src.audit import AuditStore
from src.exchange.client import ExchangeClient
from src.execution.guards import ExecutionGuard


class OrderManager:
    def __init__(
        self,
        client: ExchangeClient,
        audit_store: AuditStore | None = None,
        guard: ExecutionGuard | None = None,
    ):
        self.client = client
        self.audit_store = audit_store
        self.guard = guard or ExecutionGuard()
        self._last_failure_reasons: dict[str, str] = {}

    async def market_order(
        self, symbol: str, side: str, quantity: float, reduce_only: bool = False
    ) -> Optional[dict]:
        if quantity <= 0:
            reason = f"invalid quantity: {quantity}"
            self._last_failure_reasons[symbol] = reason
            logger.warning(f"Invalid quantity for {symbol}: {quantity}")
            return None
        return await self._create_order_with_retries(
            symbol,
            "market",
            side,
            quantity,
            params=self._params(reduce_only=reduce_only),
        )

    async def limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool = True,
    ) -> Optional[dict]:
        if quantity <= 0:
            logger.warning(f"Invalid quantity for {symbol}: {quantity}")
            return None
        return await self._create_order_with_retries(
            symbol,
            "limit",
            side,
            quantity,
            price,
            params=self._params(post_only=post_only),
        )

    async def stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        price: Optional[float] = None,
    ) -> Optional[dict]:
        if quantity <= 0:
            logger.warning(f"Invalid quantity for {symbol}: {quantity}")
            return None
        params = self._params(reduce_only=True, prefix="sl")
        params["stopPrice"] = stop_price
        order_type = "stop_market" if price is None else "stop_limit"
        if price is not None:
            params["price"] = price
        return await self._create_order_with_retries(
            symbol, order_type, side, quantity, price, params=params
        )

    async def take_profit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Optional[dict]:
        if quantity <= 0:
            logger.warning(f"Invalid quantity for {symbol}: {quantity}")
            return None
        return await self._create_order_with_retries(
            symbol,
            "limit",
            side,
            quantity,
            price,
            params=self._params(reduce_only=True, prefix="tp"),
        )

    async def cancel_all_orders(self, symbol: str):
        for conditional in (False, True):
            try:
                await self.client.cancel_all_orders(symbol, conditional=conditional)
                order_class = "conditional" if conditional else "standard"
                logger.info(f"Cancelled all {order_class} orders for {symbol}")
                self._audit(
                    "orders_cancelled",
                    f"Cancelled all {order_class} open orders for {symbol}",
                    symbol=symbol,
                    payload={"conditional": conditional},
                )
            except Exception as e:
                order_class = "conditional" if conditional else "standard"
                logger.error(f"Cancel all {order_class} orders failed: {e}")
                self._audit(
                    "order_cancel_failed",
                    f"Cancel all {order_class} orders failed",
                    severity="error",
                    symbol=symbol,
                    payload={"conditional": conditional, "error": str(e)},
                )

    async def cancel_order(
        self, symbol: str, order_id: str, *, conditional: bool = False
    ):
        try:
            params = {"trigger": True} if conditional else {}
            await self.client.cancel_order(order_id, symbol, params=params)
            logger.info(f"Cancelled order {order_id} for {symbol}")
            self._audit(
                "order_cancelled",
                f"Cancelled order {order_id}",
                symbol=symbol,
                payload={"order_id": order_id, "conditional": conditional},
            )
        except OrderNotFound:
            logger.info(
                f"Order {order_id} for {symbol} was already closed or cancelled"
            )
            self._audit(
                "order_already_closed",
                f"Order {order_id} already absent",
                symbol=symbol,
                payload={"order_id": order_id, "conditional": conditional},
            )
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            self._audit(
                "order_cancel_failed",
                "Cancel order failed",
                severity="error",
                symbol=symbol,
                payload={
                    "order_id": order_id,
                    "conditional": conditional,
                    "error": str(e),
                },
            )

    @staticmethod
    def _client_order_id(prefix: str) -> str:
        return f"tb_{prefix}_{uuid.uuid4().hex[:24]}"

    def _params(
        self,
        *,
        reduce_only: bool = False,
        post_only: bool | None = None,
        prefix: str = "ord",
    ) -> dict[str, object]:
        params: dict[str, object] = {"newClientOrderId": self._client_order_id(prefix)}
        if reduce_only:
            params["reduceOnly"] = True
        if post_only is not None:
            params["postOnly"] = post_only
        return params

    async def _create_order_with_retries(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: float | None = None,
        *,
        params: dict | None = None,
        attempts: int = 3,
    ) -> Optional[dict]:
        self._last_failure_reasons.pop(symbol, None)
        params = params or {}
        prepared = await self.guard.prepare(
            self.client,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            params=params,
        )
        if not prepared.allowed:
            self._last_failure_reasons[symbol] = prepared.reason
            logger.warning(f"Order blocked for {symbol}: {prepared.reason}")
            self._audit(
                "order_blocked",
                prepared.reason,
                severity="warning",
                symbol=symbol,
                payload={
                    "order_type": order_type,
                    "side": side,
                    "quantity": quantity,
                    "prepared_quantity": prepared.quantity,
                    "price": price,
                    "prepared_price": prepared.price,
                    "reference_price": prepared.reference_price,
                    "slippage_bps": prepared.slippage_bps,
                    "notional": prepared.notional,
                },
            )
            return None

        quantity = prepared.quantity
        price = prepared.price
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                order = await self.client.create_order(
                    symbol, order_type, side, quantity, price, params=params
                )
                logger.info(
                    f"{order_type.upper()} {side} {quantity} {symbol}: "
                    f"id={order.get('id')} filled={order.get('filled')}"
                )
                self._audit(
                    "order_placed",
                    f"{order_type} {side} order placed",
                    symbol=symbol,
                    payload={
                        "order_type": order_type,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "client_order_id": params.get("newClientOrderId"),
                        "order": order,
                    },
                )
                return order
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"{order_type} {side} {symbol} failed "
                    f"attempt {attempt}/{attempts}: {last_error}"
                )
                if attempt < attempts:
                    await asyncio.sleep(0.5 * attempt)

        logger.error(f"{order_type} {side} {symbol} failed permanently: {last_error}")
        self._last_failure_reasons[symbol] = (
            f"{order_type} {side} order failed: {last_error}"
        )
        self._audit(
            "order_failed",
            f"{order_type} {side} order failed",
            severity="error",
            symbol=symbol,
            payload={
                "order_type": order_type,
                "side": side,
                "quantity": quantity,
                "price": price,
                "client_order_id": params.get("newClientOrderId"),
                "error": last_error,
            },
        )
        return None

    def failure_reason(self, symbol: str, fallback: str) -> str:
        return self._last_failure_reasons.get(symbol, fallback)

    def _audit(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        symbol: str | None = None,
        payload: dict | None = None,
    ) -> None:
        if self.audit_store:
            self.audit_store.safe_record_event(
                event_type,
                message,
                severity=severity,
                symbol=symbol,
                payload=payload,
            )
