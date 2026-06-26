# Protection order lifecycle — SL/TP placement, verification, cancellation
from collections.abc import Awaitable, Callable

from loguru import logger

from src.audit import AuditStore
from src.exchange.client import ExchangeClient
from src.execution.fill_resolver import FillResolver
from src.execution.order_manager import OrderManager
from src.monitoring.alerter import Alerter
from src.risk.portfolio import TradeRecord
from src.risk.stop_loss import StopLossLevels
from src.security import redact_text


class ProtectionManager:
    def __init__(
        self,
        client: ExchangeClient,
        orders: OrderManager,
        audit_store: AuditStore,
        fill_resolver: FillResolver,
        alerter: Alerter | None = None,
        mode: str = "trade",
    ):
        self.client = client
        self.orders = orders
        self.audit_store = audit_store
        self.fill_resolver = fill_resolver
        self.alerter = alerter
        self.mode = mode
        self.active_stops: dict[str, str] = {}
        self.active_tps: dict[str, str] = {}

    def track_protection(
        self,
        position_key: str,
        stop_order_id: str,
        take_profit_order_id: str | None,
    ) -> None:
        self.active_stops[position_key] = stop_order_id
        if take_profit_order_id:
            self.active_tps[position_key] = take_profit_order_id

    def clear_protection(
        self,
        position_key: str,
    ) -> tuple[str | None, str | None]:
        return (
            self.active_stops.pop(position_key, None),
            self.active_tps.pop(position_key, None),
        )

    def verify_protective_orders(
        self,
        position_key: str,
        order_ids: tuple[set[str], set[str]],
    ) -> bool:
        stop_order_id = self.active_stops.get(position_key)
        take_profit_order_id = self.active_tps.get(position_key)
        if not stop_order_id:
            return False
        open_order_ids, conditional_order_ids = order_ids
        if stop_order_id not in conditional_order_ids:
            return False
        if take_profit_order_id and take_profit_order_id not in open_order_ids:
            return False
        return True

    def missing_protection_candidates(
        self,
        symbol_trades: list[tuple[str, TradeRecord]],
        order_ids: tuple[set[str], set[str]],
    ) -> list[tuple[str, TradeRecord]]:
        standard_open, conditional_open = order_ids
        candidates = []
        for position_key, trade in symbol_trades:
            stop_id = self.active_stops.get(position_key)
            target_id = self.active_tps.get(position_key)
            if stop_id not in conditional_open or (
                target_id and target_id not in standard_open
            ):
                candidates.append((position_key, trade))
        return sorted(
            candidates,
            key=lambda item: item[1].timestamp,
            reverse=True,
        )

    async def verify_all_protective_orders(
        self,
        open_trades: dict[str, TradeRecord],
        order_snapshots: dict[str, tuple[set[str], set[str]]],
        fail_reconciliation: Callable[..., Awaitable[None]] | None = None,
    ) -> bool:
        for position_key, trade in list(open_trades.items()):
            if self.verify_protective_orders(
                position_key,
                order_snapshots.get(getattr(trade, "symbol", ""), (set(), set())),
            ):
                continue
            reason = f"missing protective orders for {getattr(trade, 'symbol', '')}"
            if fail_reconciliation is not None:
                await fail_reconciliation(
                    "missing_protection",
                    reason,
                    symbol=getattr(trade, "symbol", ""),
                    correlation_id=None,
                    payload={
                        "position_key": position_key,
                        "stop_order_id": self.active_stops.get(position_key),
                        "take_profit_order_id": self.active_tps.get(position_key),
                    },
                )
            return False
        return True

    async def protective_order_snapshots(
        self,
        open_trades: dict[str, TradeRecord],
        fail_reconciliation: Callable[..., Awaitable[None]] | None = None,
    ) -> dict[str, tuple[set[str], set[str]]] | None:
        snapshots: dict[str, tuple[set[str], set[str]]] = {}
        for trade in open_trades.values():
            symbol = getattr(trade, "symbol", "")
            if symbol in snapshots:
                continue
            try:
                open_orders = await self.client.fetch_open_orders(symbol)
                conditional_orders = await self.client.fetch_open_orders(
                    symbol, conditional=True
                )
            except Exception as exc:
                reason = redact_text(
                    f"open order reconciliation failed for {symbol}: {exc}"
                )
                if fail_reconciliation is not None:
                    await fail_reconciliation(
                        "open_order_reconciliation_failed",
                        reason,
                        symbol=symbol,
                    )
                return None
            known_standard = {
                order_id
                for position_key, order_id in self.active_tps.items()
                if getattr(open_trades.get(position_key), "symbol", "") == symbol
            }
            known_conditional = {
                order_id
                for position_key, order_id in self.active_stops.items()
                if getattr(open_trades.get(position_key), "symbol", "") == symbol
            }
            open_orders = await self._cancel_orphan_bot_orders(
                symbol,
                open_orders,
                known_standard,
                conditional=False,
            )
            conditional_orders = await self._cancel_orphan_bot_orders(
                symbol,
                conditional_orders,
                known_conditional,
                conditional=True,
            )
            snapshots[symbol] = (
                {str(order.get("id", "")) for order in open_orders},
                {str(order.get("id", "")) for order in conditional_orders},
            )
        return snapshots

    @staticmethod
    def _client_order_id(order: dict) -> str:
        info = order.get("info") or {}
        return str(
            order.get("clientOrderId")
            or info.get("clientOrderId")
            or info.get("clientAlgoId")
            or ""
        )

    async def _cancel_orphan_bot_orders(
        self,
        symbol: str,
        orders: list[dict],
        known_order_ids: set[str],
        *,
        conditional: bool,
    ) -> list[dict]:
        retained = []
        for order in orders:
            order_id = str(order.get("id") or "")
            client_order_id = self._client_order_id(order)
            bot_protection = client_order_id.startswith(("tb_sl_", "tb_tp_"))
            if not bot_protection or order_id in known_order_ids:
                retained.append(order)
                continue
            logger.warning(
                f"Cancelling orphan bot protective order {order_id} for {symbol}"
            )
            try:
                await self.orders.cancel_order(
                    symbol,
                    order_id,
                    conditional=conditional,
                )
            except Exception as exc:
                logger.warning(
                    f"Could not cancel orphan protective order {order_id}: "
                    f"{redact_text(exc)}"
                )
                retained.append(order)
                continue
            self.audit_store.safe_record_event(
                "orphan_protection_cancelled",
                "Cancelled untracked bot protective order",
                severity="warning",
                symbol=symbol,
                mode=self.mode,
                payload={
                    "order_id": order_id,
                    "client_order_id": client_order_id,
                    "conditional": conditional,
                },
            )
        return retained

    async def place_entry_protection(
        self,
        symbol: str,
        trade_side: str,
        quantity: float,
        levels: StopLossLevels,
    ) -> tuple[dict, dict | None] | None:
        exit_side = "sell" if trade_side == "long" else "buy"
        sl_order = await self.orders.stop_loss_order(
            symbol,
            exit_side,
            quantity,
            levels.stop_loss,
        )
        tp_order = None
        if levels.take_profit is not None:
            tp_order = await self.orders.take_profit_order(
                symbol,
                exit_side,
                quantity,
                levels.take_profit,
            )
        if sl_order and (levels.take_profit is None or tp_order):
            return sl_order, tp_order
        if sl_order:
            await self.orders.cancel_order(
                symbol, str(sl_order.get("id", "")), conditional=True
            )
        return None

    async def cancel_protection(
        self,
        symbol: str,
        stop_order_id: str | None,
        take_profit_order_id: str | None,
    ) -> None:
        if stop_order_id:
            await self.orders.cancel_order(symbol, stop_order_id, conditional=True)
        if take_profit_order_id:
            await self.orders.cancel_order(symbol, take_profit_order_id)

    async def replace_stop(
        self,
        position_key: str,
        trade: TradeRecord,
        stop_price: float,
    ) -> str | None:
        exit_side = "sell" if trade.side == "long" else "buy"
        replacement = await self.orders.stop_loss_order(
            trade.symbol,
            exit_side,
            trade.quantity,
            stop_price,
        )
        if not replacement:
            return None
        replacement_id = str(replacement.get("id") or "")
        if not replacement_id:
            return None
        previous_id = self.active_stops.get(position_key)
        self.active_stops[position_key] = replacement_id
        if previous_id and previous_id != replacement_id:
            await self.orders.cancel_order(
                trade.symbol,
                previous_id,
                conditional=True,
            )
        return replacement_id

    def _audit_entry_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        symbol: str | None = None,
        correlation_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        self.audit_store.safe_record_event(
            event_type,
            message,
            severity=severity,
            symbol=symbol,
            mode=self.mode,
            correlation_id=correlation_id,
            payload=payload,
        )
