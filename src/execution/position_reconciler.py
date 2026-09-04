# Exchange state reconciliation, finalization, and protective exit detection.
from typing import Literal

from loguru import logger

from src.config import settings
from src.risk.stop_loss import StopLossLevels
from src.security import redact_text
from src.strategies import StrategyRegistry


class PositionReconciler:
    def __init__(
        self,
        client,
        orders,
        fill_resolver,
        protection,
        trades,
        portfolio,
        alerter,
        audit_store,
        mode,
    ):
        self.client = client
        self.orders = orders
        self._fill_resolver = fill_resolver
        self.protection = protection
        self.trades = trades
        self.portfolio = portfolio
        self.alerter = alerter
        self.audit_store = audit_store
        self.mode = mode
        self.strategy_registry = StrategyRegistry(settings)
        self._consumed_exit_order_ids: set[str] = set()
        self._position_fetch_failures = 0

    async def _order_fee(
        self,
        order: dict,
        trade,
        execution_price: float,
    ) -> float:
        policy = self.strategy_registry.get(trade.strategy)
        return await self._fill_resolver.order_fee(
            order,
            trade.symbol,
            fallback_notional=execution_price * trade.quantity,
            fallback_fee_bps=policy.estimated_round_trip_fee_bps / 2,
        )

    def _audit(
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

    @staticmethod
    def _position_size(position: dict) -> float:
        info = position.get("info", {})
        for key in ("positionAmt", "positionAmt".lower()):
            value = info.get(key)
            if value not in (None, ""):
                assert value is not None
                return float(value)
        for key in ("contracts",):
            value = position.get(key)
            if value not in (None, ""):
                assert value is not None
                return float(value)
        return 0.0

    @staticmethod
    def _position_side(position: dict) -> str | None:
        info = position.get("info", {})
        position_amount = info.get("positionAmt")
        if position_amount not in (None, ""):
            signed_amount = float(position_amount)
            if signed_amount > 0:
                return "long"
            if signed_amount < 0:
                return "short"
        side = str(position.get("side") or "").lower()
        if side in {"long", "short"}:
            return side
        position_side = str(info.get("positionSide") or "").lower()
        if position_side in {"long", "short"}:
            return position_side
        size = PositionReconciler._position_size(position)
        if size > 0:
            return "long"
        if size < 0:
            return "short"
        return None

    @staticmethod
    def _exchange_symbols(
        exchange_positions: dict[str, dict[str, dict]],
    ) -> set[str]:
        return set(exchange_positions)

    @staticmethod
    def _select_position(
        exchange_positions: dict[str, dict[str, dict]],
        symbol: str,
        preferred_side: str | None = None,
    ) -> dict | None:
        pos_map = exchange_positions.get(symbol)
        if not pos_map:
            return None
        if preferred_side and preferred_side in pos_map:
            return pos_map[preferred_side]
        return next(iter(pos_map.values()), None)

    @staticmethod
    def _close_position_side_param(position: dict) -> str | None:
        info = position.get("info", {})
        position_side = str(info.get("positionSide") or "").upper()
        if position_side in {"LONG", "SHORT"}:
            return position_side
        return None

    async def _fetch_positions_for_reconciliation(self) -> list[dict] | None:
        try:
            positions = await self.client.fetch_positions()
        except Exception as exc:
            self._position_fetch_failures += 1
            reason = redact_text(f"position reconciliation unavailable: {exc}")
            logger.warning(reason)
            self._audit(
                "reconciliation_unavailable",
                reason,
                severity="warning",
                payload={"consecutive_failures": self._position_fetch_failures},
            )
            open_trades = getattr(self.trades, "open_trades", {})
            urgent = bool(open_trades)
            threshold_reached = (
                self._position_fetch_failures
                >= settings.reconciliation_alert_failure_threshold
            )
            if self.alerter and (urgent or threshold_reached):
                await self.alerter.error_alert(reason)
            return None
        if self._position_fetch_failures:
            self._audit(
                "reconciliation_connectivity_recovered",
                "Position reconciliation connectivity recovered",
                payload={"previous_failures": self._position_fetch_failures},
            )
            self._position_fetch_failures = 0
        return positions

    def _classify_exchange_positions(
        self, positions: list[dict]
    ) -> tuple[dict[str, dict[str, dict]], list[dict]]:
        unmanaged = []
        exchange_positions: dict[str, dict[str, dict]] = {}
        for position in positions:
            symbol = self.trades.normalize_symbol(
                position.get("symbol") or position.get("info", {}).get("symbol")
            )
            size = self._position_size(position)
            if abs(size) <= 0 or not symbol:
                continue
            side_key = self._close_position_side_param(position) or "BOTH"
            pos_map = exchange_positions.setdefault(symbol, {})
            pos_map[side_key] = position
            if not self.trades.has_open_trade_for_symbol(symbol):
                unmanaged.append(
                    {
                        "symbol": symbol,
                        "size": size,
                        "side": self._position_side(position),
                        "position_side": self._close_position_side_param(position),
                    }
                )
        return exchange_positions, unmanaged

    async def _close_unmanaged_position(self, item: dict) -> None:
        symbol = item["symbol"]
        size = float(item["size"])
        side = "sell" if size > 0 else "buy"
        qty = abs(size)
        position_side = item.get("position_side")
        logger.warning(
            f"Closing unmanaged position: {symbol} "
            f"{'long' if side == 'sell' else 'short'} {qty}"
        )
        await self.orders.cancel_all_orders(symbol)
        order = await self.orders.market_order(
            symbol,
            side,
            qty,
            reduce_only=True,
            position_side=position_side,
        )
        if order is not None:
            return

        failure_reason = self.orders.failure_reason(symbol, "")
        if "reduceonly" not in failure_reason.lower():
            return
        logger.warning(
            f"Retrying unmanaged close for {symbol} without reduceOnly after "
            "Binance reduce-only rejection"
        )
        await self.orders.market_order(
            symbol,
            side,
            qty,
            reduce_only=False,
            position_side=position_side,
        )

    async def _reconciliation_state(
        self,
        *,
        auto_close_unmanaged: bool = False,
    ) -> (
        tuple[
            dict[str, dict[str, dict]],
            dict[str, tuple[set[str], set[str]]],
        ]
        | Literal[False]
        | None
    ):
        positions = await self._fetch_positions_for_reconciliation()
        if positions is None:
            return None
        exchange_positions, unmanaged = self._classify_exchange_positions(positions)
        if unmanaged:
            if auto_close_unmanaged:
                for item in unmanaged:
                    try:
                        await self._close_unmanaged_position(item)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to close unmanaged {item['symbol']}: {exc}"
                        )
                positions = await self._fetch_positions_for_reconciliation()
                if positions is None:
                    return None
                exchange_positions, unmanaged = self._classify_exchange_positions(
                    positions
                )
                if not unmanaged:
                    order_snapshots = await self.protection.protective_order_snapshots(
                        self.trades.open_trades,
                        fail_reconciliation=self._fail_reconciliation,
                    )
                    if order_snapshots is None:
                        return False
                    return exchange_positions, order_snapshots
            reason = f"unmanaged exchange positions detected: {unmanaged}"
            await self._fail_reconciliation(
                "unmanaged_positions",
                reason,
                payload={"positions": unmanaged},
            )
            return False

        order_snapshots = await self.protection.protective_order_snapshots(
            self.trades.open_trades,
            fail_reconciliation=self._fail_reconciliation,
        )
        if order_snapshots is None:
            return False
        return exchange_positions, order_snapshots

    async def _finalize_trade_leg(
        self,
        position_key: str,
        *,
        exit_price: float,
        reason: str,
        correlation_id: str,
        notify_trade_completed,
        exit_fee: float = 0.0,
    ) -> None:
        trade = self.trades.open_trades[position_key]
        logger.info(
            f"{position_key}: exchange trade leg closed via {reason} at {exit_price}"
        )
        self.portfolio.close_trade(
            trade,
            exit_price,
            reason,
            exit_fee=exit_fee,
        )
        if correlation_id:
            self.audit_store.record_closed_trade(
                trade, mode=self.mode, correlation_id=correlation_id
            )
        await notify_trade_completed(trade, exit_price, reason)

        stop_order_id = self.protection.active_stops.pop(position_key, None)
        take_profit_order_id = self.protection.active_tps.pop(position_key, None)
        if stop_order_id and reason != "stop_loss":
            await self.orders.cancel_order(
                trade.symbol, stop_order_id, conditional=True
            )
        if take_profit_order_id and reason != "take_profit":
            await self.orders.cancel_order(trade.symbol, take_profit_order_id)
        self.trades.open_trades.pop(position_key, None)
        self.trades.trade_correlation_ids.pop(position_key, None)
        self.trades.record_exit_time(trade.symbol)

    async def _filled_protective_exit_details(  # noqa: C901
        self,
        position_key: str,
        *,
        open_order_ids: tuple[set[str], set[str]] | None = None,
    ) -> tuple[float, str, float] | None:
        trade = self.trades.open_trades[position_key]
        standard_open, conditional_open = open_order_ids or (set(), set())
        candidates = (
            ("take_profit", self.protection.active_tps.get(position_key), False),
            ("stop_loss", self.protection.active_stops.get(position_key), True),
        )
        for reason, order_id, conditional in candidates:
            if not order_id:
                continue
            currently_open = (
                order_id in conditional_open
                if conditional
                else order_id in standard_open
            )
            if currently_open:
                continue
            order = await self._fill_resolver.resolved_order(
                trade.symbol,
                order_id,
                reason=reason,
                conditional=conditional,
            )
            if order is None:
                execution_order = await self._fill_resolver.latest_exit_order(
                    trade, skip_order_ids=self._consumed_exit_order_ids
                )
                if execution_order is not None:
                    execution_price = self._fill_resolver.positive_order_price(
                        execution_order
                    )
                    if (
                        execution_price is not None
                        and self._execution_matches_protection(
                            position_key,
                            reason,
                            execution_price,
                        )
                    ):
                        oid = str(execution_order.get("id") or "")
                        if oid:
                            self._consumed_exit_order_ids.add(oid)
                        fee = await self._order_fee(
                            execution_order, trade, execution_price
                        )
                        return execution_price, reason, fee
                continue
            if not self._fill_resolver.order_is_filled(order):
                continue
            if reason == "take_profit":
                price = self._fill_resolver.positive_order_price(order)
                if price is not None:
                    fee = await self._order_fee(order, trade, price)
                    return price, reason, fee
            execution_order = await self._fill_resolver.latest_exit_order(
                trade, skip_order_ids=self._consumed_exit_order_ids
            )
            if execution_order is not None:
                execution_price = self._fill_resolver.positive_order_price(
                    execution_order
                )
                if execution_price is not None and self._execution_matches_protection(
                    position_key,
                    reason,
                    execution_price,
                ):
                    oid = str(execution_order.get("id") or "")
                    if oid:
                        self._consumed_exit_order_ids.add(oid)
                    fee = await self._order_fee(execution_order, trade, execution_price)
                    return execution_price, reason, fee
            price = self._fill_resolver.positive_order_price(order)
            if price is None:
                continue
            fee = await self._order_fee(order, trade, price)
            return price, reason, fee
        return None

    def _execution_matches_protection(
        self,
        position_key: str,
        reason: str,
        execution_price: float,
    ) -> bool:
        correlation_id = self.trades.trade_correlation_ids.get(position_key)
        if not correlation_id or execution_price <= 0:
            return False
        stop_loss, take_profit = self.audit_store.get_trade_protection_levels(
            correlation_id
        )
        expected_price = take_profit if reason == "take_profit" else stop_loss
        if expected_price is None or expected_price <= 0:
            return False
        distance_bps = abs(execution_price - expected_price) / expected_price * 10_000
        return distance_bps <= settings.protection_execution_match_bps

    async def _protective_exit_details(
        self,
        position_key: str,
    ) -> tuple[float, str, float]:
        filled_details = await self._filled_protective_exit_details(position_key)
        if filled_details is not None:
            return filled_details

        trade = self.trades.open_trades[position_key]
        execution_order = await self._fill_resolver.latest_exit_order(trade)
        if execution_order is not None:
            price = self._fill_resolver.positive_order_price(execution_order)
            if price is not None:
                fee = await self._order_fee(execution_order, trade, price)
                return price, "exchange_close", fee
        return (
            await self._fill_resolver.market_price(trade.symbol, trade.entry_price),
            "exchange_close",
            0.0,
        )

    async def _finalize_exchange_closed_position(
        self, position_key: str, notify_trade_completed
    ) -> None:
        correlation_id = self.trades.trade_correlation_ids.get(position_key, "")
        exit_price, reason, exit_fee = await self._protective_exit_details(position_key)
        await self._finalize_trade_leg(
            position_key,
            exit_price=exit_price,
            reason=reason,
            correlation_id=correlation_id,
            notify_trade_completed=notify_trade_completed,
            exit_fee=exit_fee,
        )

    async def _clear_missing_exchange_positions(
        self, exchange_symbols: set[str], notify_trade_completed
    ) -> None:
        missing_position_keys = [
            key
            for key, trade in self.trades.open_trades.items()
            if trade.symbol not in exchange_symbols
        ]
        for position_key in missing_position_keys:
            await self._finalize_exchange_closed_position(
                position_key, notify_trade_completed
            )

    async def _verify_exchange_position_details(
        self, exchange_positions: dict[str, dict[str, dict]]
    ) -> bool:
        for symbol in {trade.symbol for trade in self.trades.open_trades.values()}:
            symbol_trades = self.trades.trades_for_symbol(symbol)
            trade_side = next(iter({t.side for _, t in symbol_trades}), None)
            position = self._select_position(
                exchange_positions,
                symbol,
                preferred_side=trade_side.upper() if trade_side else None,
            )
            if position is None:
                continue
            exchange_side = self._position_side(position)
            audit_sides = {trade.side for _, trade in symbol_trades}
            if len(audit_sides) != 1:
                reason = f"mixed audited sides detected for {symbol}: {audit_sides}"
                await self._fail_reconciliation(
                    "mixed_position_sides",
                    reason,
                    symbol=symbol,
                )
                return False
            audit_side = next(iter(audit_sides))
            if exchange_side and exchange_side != audit_side:
                reason = (
                    f"exchange side mismatch for {symbol}: "
                    f"audit={audit_side} exchange={exchange_side}"
                )
                await self._fail_reconciliation(
                    "position_side_mismatch",
                    reason,
                    symbol=symbol,
                    correlation_id=self.trades.trade_correlation_ids.get(
                        symbol_trades[0][0]
                    ),
                )
                return False

            exchange_quantity = abs(self._position_size(position))
            audit_quantity = sum(trade.quantity for _, trade in symbol_trades)
            tolerance = await self._fill_resolver.quantity_tolerance(symbol)
            if abs(exchange_quantity - audit_quantity) > tolerance:
                reason = (
                    f"exchange quantity mismatch for {symbol}: "
                    f"audit={audit_quantity} exchange={exchange_quantity} "
                    f"tolerance={tolerance}"
                )
                await self._fail_reconciliation(
                    "position_quantity_mismatch",
                    reason,
                    symbol=symbol,
                    correlation_id=self.trades.trade_correlation_ids.get(
                        symbol_trades[0][0]
                    ),
                )
                return False
        return True

    async def _exit_execution_groups(self, symbol: str) -> list[dict]:
        fetch_my_trades = getattr(self.client, "fetch_my_trades", None)
        if not callable(fetch_my_trades):
            return []
        try:
            trades = await fetch_my_trades(symbol, limit=100)
        except Exception as exc:
            logger.warning(
                "Could not fetch exit executions for {}: {}",
                symbol,
                redact_text(exc),
            )
            return []

        groups: dict[str, dict] = {}
        for trade in trades:
            info = trade.get("info") or {}
            order_id = str(trade.get("order") or info.get("orderId") or "")
            amount = float(
                trade.get("amount") or info.get("qty") or info.get("quantity") or 0
            )
            price = float(trade.get("price") or info.get("price") or 0)
            if not order_id or amount <= 0 or price <= 0:
                continue
            group = groups.setdefault(
                order_id,
                {
                    "order_id": order_id,
                    "timestamp": int(trade.get("timestamp") or 0),
                    "side": str(trade.get("side") or "").lower(),
                    "quantity": 0.0,
                    "notional": 0.0,
                    "fee": 0.0,
                },
            )
            group["timestamp"] = min(
                group["timestamp"],
                int(trade.get("timestamp") or 0),
            )
            group["quantity"] += amount
            group["notional"] += amount * price
            group["fee"] += self._fill_resolver.execution_fee(trade)
        executions = []
        for group in groups.values():
            quantity = group["quantity"]
            executions.append(
                {
                    **group,
                    "price": group["notional"] / quantity,
                }
            )
        return sorted(executions, key=lambda item: item["timestamp"])

    async def _reconcile_partial_exchange_exits(
        self,
        exchange_positions: dict[str, dict[str, dict]],
        order_snapshots: dict[str, tuple[set[str], set[str]]],
        notify_trade_completed,
    ) -> bool:
        finalized = False
        # Group by (symbol, side) rather than symbol alone: in hedge mode a
        # symbol can carry both a long and a short leg at once, each backed
        # by its own exchange position. Aggregating quantity across sides
        # and comparing it to a single arbitrarily-picked side's exchange
        # position would misdetect a missing/partial exit on one leg based
        # on the other leg's size entirely.
        executions_by_symbol: dict[str, list[dict]] = {}
        symbol_sides = {
            (trade.symbol, trade.side) for trade in self.trades.open_trades.values()
        }
        for symbol, side in symbol_sides:
            symbol_trades = [
                (key, trade)
                for key, trade in self.trades.trades_for_symbol(symbol)
                if trade.side == side
            ]
            audited_quantity = sum(trade.quantity for _, trade in symbol_trades)
            position = self._select_position(
                exchange_positions, symbol, preferred_side=side.upper()
            )
            exchange_quantity = abs(self._position_size(position)) if position else 0.0
            tolerance = await self._fill_resolver.quantity_tolerance(symbol)
            missing_quantity = audited_quantity - exchange_quantity
            if missing_quantity <= tolerance:
                continue

            candidates = self.protection.missing_protection_candidates(
                symbol_trades,
                order_snapshots.get(symbol, (set(), set())),
            )
            if symbol not in executions_by_symbol:
                executions_by_symbol[symbol] = await self._exit_execution_groups(
                    symbol
                )
            executions = executions_by_symbol[symbol]
            consumed_orders: set[str] = set()
            for position_key, trade in candidates:
                if trade.quantity > missing_quantity + tolerance:
                    continue
                details = await self._filled_protective_exit_details(
                    position_key,
                    open_order_ids=order_snapshots.get(symbol, (set(), set())),
                )
                if details is not None:
                    exit_price, reason, exit_fee = details
                    await self._finalize_trade_leg(
                        position_key,
                        exit_price=exit_price,
                        reason=reason,
                        correlation_id=self.trades.trade_correlation_ids.get(
                            position_key, ""
                        ),
                        notify_trade_completed=notify_trade_completed,
                        exit_fee=exit_fee,
                    )
                    missing_quantity -= trade.quantity
                    finalized = True
                    if missing_quantity <= tolerance:
                        break
                    continue

                execution = next(
                    (
                        item
                        for item in executions
                        if item["order_id"] not in consumed_orders
                        and item["order_id"] not in self._consumed_exit_order_ids
                        and item["timestamp"] >= int(trade.timestamp.timestamp() * 1000)
                        and item["side"] == ("sell" if trade.side == "long" else "buy")
                        and abs(item["quantity"] - trade.quantity) <= tolerance
                    ),
                    None,
                )
                if execution is None:
                    continue
                reason = self._inferred_protective_exit_reason(
                    position_key,
                    trade,
                    execution["price"],
                )
                if not self._execution_matches_protection(
                    position_key,
                    reason,
                    execution["price"],
                ):
                    continue
                await self._finalize_trade_leg(
                    position_key,
                    exit_price=execution["price"],
                    reason=reason,
                    correlation_id=self.trades.trade_correlation_ids.get(
                        position_key, ""
                    ),
                    notify_trade_completed=notify_trade_completed,
                    exit_fee=float(execution.get("fee") or 0.0),
                )
                consumed_orders.add(execution["order_id"])
                self._consumed_exit_order_ids.add(execution["order_id"])
                missing_quantity -= trade.quantity
                finalized = True
                if missing_quantity <= tolerance:
                    break
        return finalized

    async def _finalize_filled_protective_legs(
        self,
        order_snapshots: dict[str, tuple[set[str], set[str]]],
        notify_trade_completed,
    ) -> bool:
        finalized = False
        for position_key in list(self.trades.open_trades):
            trade = self.trades.open_trades[position_key]
            details = await self._filled_protective_exit_details(
                position_key,
                open_order_ids=order_snapshots.get(trade.symbol, (set(), set())),
            )
            if details is None:
                continue
            exit_price, reason, exit_fee = details
            await self._finalize_trade_leg(
                position_key,
                exit_price=exit_price,
                reason=reason,
                correlation_id=self.trades.trade_correlation_ids.get(position_key, ""),
                notify_trade_completed=notify_trade_completed,
                exit_fee=exit_fee,
            )
            finalized = True
        return finalized

    def _inferred_protective_exit_reason(
        self,
        position_key: str,
        trade,
        exit_price: float,
    ) -> str:
        profitable = (
            exit_price > trade.entry_price
            if trade.side == "long"
            else exit_price < trade.entry_price
        )
        reason = "take_profit" if profitable else "stop_loss"
        self._audit(
            "protective_exit_inferred",
            f"Inferred {reason} execution for {position_key}",
            severity="warning",
            symbol=trade.symbol,
            correlation_id=self.trades.trade_correlation_ids.get(position_key),
            payload={
                "position_key": position_key,
                "entry_price": trade.entry_price,
                "exit_price": exit_price,
            },
        )
        return reason

    async def _exchange_reports_position_closed(self, trade) -> bool:
        """True only when the exchange was readable AND reports the leg gone.

        An unreadable snapshot returns False (unknown), so callers never treat a
        failed position read as evidence that a live position has been closed.
        """
        exchange_positions = await self._fetch_positions_for_reconciliation()
        if exchange_positions is None:
            return False
        exchange_pos = next(
            (
                pos
                for pos in exchange_positions
                if self.trades.normalize_symbol(
                    pos.get("symbol") or pos.get("info", {}).get("symbol")
                )
                == trade.symbol
            ),
            None,
        )
        if exchange_pos is None:
            return True
        return abs(self._position_size(exchange_pos)) < trade.quantity * 0.5

    async def _recreate_protection_for_trade(
        self,
        position_key: str,
        notify_trade_completed=None,
    ) -> bool:
        trade = self.trades.open_trades.get(position_key)
        if not trade:
            return False
        correlation_id = self.trades.trade_correlation_ids.get(position_key)
        if not correlation_id:
            return False

        if await self._exchange_reports_position_closed(trade):
            logger.info(
                f"{position_key}: position already closed on exchange; "
                "finalizing instead of recreating protection"
            )
            if notify_trade_completed:
                await self._finalize_exchange_closed_position(
                    position_key, notify_trade_completed
                )
            return True

        sl, tp = self.audit_store.get_trade_protection_levels(correlation_id)
        if sl is None:
            return False
        levels = StopLossLevels(stop_loss=sl, take_profit=tp)
        side = trade.side.lower()
        protection = await self.protection.place_entry_protection(
            trade.symbol,
            side,
            trade.quantity,
            levels,
        )
        if protection is None:
            if await self._exchange_reports_position_closed(trade):
                logger.info(
                    f"{position_key}: position closed on exchange after protection "
                    "recreation failure; finalizing trade"
                )
                if notify_trade_completed:
                    await self._finalize_exchange_closed_position(
                        position_key, notify_trade_completed
                    )
                return True
            return False
        sl_order, tp_order = protection
        stop_order_id = sl_order.get("id", "")
        take_profit_order_id = tp_order.get("id", "") if tp_order else None
        self.protection.active_stops[position_key] = stop_order_id
        if take_profit_order_id:
            self.protection.active_tps[position_key] = take_profit_order_id
        self.audit_store.record_open_trade(
            trade,
            mode=self.mode,
            correlation_id=correlation_id,
            stop_loss=sl,
            take_profit=tp,
            stop_order_id=stop_order_id,
            take_profit_order_id=take_profit_order_id,
        )
        logger.warning(
            f"Recreated protective orders for {position_key}: "
            f"SL={stop_order_id} TP={take_profit_order_id}"
        )
        return True

    async def _fail_reconciliation(
        self,
        event_type: str,
        reason: str,
        *,
        symbol: str | None = None,
        correlation_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        safe_reason = redact_text(reason)
        previous_level = (
            self.audit_store.get_trading_level()
            if symbol is None
            else self.audit_store.get_symbol_trading_level(symbol)
        )
        level = self.audit_store.degrade_trading_level(safe_reason, symbol=symbol)
        if level == self.audit_store.TRADING_LEVEL_RED:
            logger.critical(f"Trading level degraded to {level}: {safe_reason}")
            self._audit(
                event_type,
                f"trading_level={level}: {safe_reason}",
                severity="critical",
                symbol=symbol,
                correlation_id=correlation_id,
                payload=payload,
            )
            if self.alerter and previous_level != self.audit_store.TRADING_LEVEL_RED:
                await self.alerter.error_alert(safe_reason)
        else:
            logger.warning(f"Trading level degraded to {level}: {safe_reason}")
            self._audit(
                event_type,
                f"trading_level={level}: {safe_reason}",
                severity="warning",
                symbol=symbol,
                correlation_id=correlation_id,
                payload=payload,
            )

    async def reconcile_exchange_state(  # noqa: C901
        self, notify_trade_completed, *, auto_close_unmanaged: bool = False
    ) -> bool | None:
        self._consumed_exit_order_ids.clear()
        state = await self._reconciliation_state(
            auto_close_unmanaged=auto_close_unmanaged
        )
        if state is None or state is False:
            return state
        exchange_positions, order_snapshots = state

        changed = await self._reconcile_partial_exchange_exits(
            exchange_positions,
            order_snapshots,
            notify_trade_completed,
        )
        if changed:
            state = await self._reconciliation_state()
            if state is None or state is False:
                return state
            exchange_positions, order_snapshots = state

        await self._clear_missing_exchange_positions(
            self._exchange_symbols(exchange_positions), notify_trade_completed
        )

        if not await self._verify_exchange_position_details(exchange_positions):
            return False

        for position_key in list(self.trades.open_trades):
            trade = self.trades.open_trades[position_key]
            symbol = trade.symbol
            exchange_pos = self._select_position(
                exchange_positions, symbol, preferred_side=trade.side.upper()
            )
            if (
                exchange_pos is None
                or abs(self._position_size(exchange_pos)) < trade.quantity * 0.5
            ):
                await self._finalize_exchange_closed_position(
                    position_key, notify_trade_completed
                )
                continue
            if self.protection.verify_protective_orders(
                position_key,
                order_snapshots.get(symbol, (set(), set())),
            ):
                continue
            if self.protection.active_stops.get(
                position_key
            ) or self.protection.active_tps.get(position_key):
                logger.warning(
                    f"Protective orders missing from exchange for {position_key}, "
                    "attempting recreation"
                )
                if await self._recreate_protection_for_trade(
                    position_key, notify_trade_completed=notify_trade_completed
                ):
                    stop_id = self.protection.active_stops.get(position_key)
                    tp_id = self.protection.active_tps.get(position_key)
                    standard, conditional = order_snapshots.get(symbol, (set(), set()))
                    if stop_id:
                        conditional = conditional | {stop_id}
                    if tp_id:
                        standard = standard | {tp_id}
                    order_snapshots[symbol] = (standard, conditional)

        if not await self.protection.verify_all_protective_orders(
            self.trades.open_trades,
            order_snapshots,
            fail_reconciliation=self._fail_reconciliation,
        ):
            return False

        if self.audit_store.get_trading_level() == self.audit_store.TRADING_LEVEL_RED:
            reason = "verified consistent exchange, audit, and protection state"
            self.audit_store.clear_emergency_stop(reason)
            self._audit(
                "emergency_stop_auto_cleared",
                reason,
                severity="warning",
            )
        else:
            self.audit_store.try_recover_trading_level("reconciliation successful")
        current = self.audit_store.get_trading_level()
        if current == self.audit_store.TRADING_LEVEL_GREEN:
            self._audit("reconciliation_ok", "Exchange state reconciled")
        else:
            self._audit(
                "reconciliation_ok",
                f"Exchange state reconciled, trading level={current}",
                severity="warning",
            )
        return True
