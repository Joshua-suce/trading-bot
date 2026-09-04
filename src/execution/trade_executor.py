# Entry/exit lifecycle, fill validation, preflight checks, and notifications.
import time
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from loguru import logger

from src.config import settings
from src.execution.entry_order_router import EntryOrderRouter
from src.execution.guards import adverse_price_movement_bps
from src.risk.portfolio import TradeRecord
from src.risk.stop_loss import StopLossLevels
from src.security import redact_text
from src.strategies import StrategyRegistry


class TradeExecutor:
    def __init__(
        self,
        client,
        orders,
        fill_resolver,
        sl_manager,
        sizer,
        portfolio,
        alerter,
        audit_store,
        mode,
        trades,
        protection,
        check_exposure_limits,
        fail_reconciliation,
        finalize_trade_leg,
    ):
        self.client = client
        self.strategy_registry = StrategyRegistry(settings)
        self.orders = orders
        self._fill_resolver = fill_resolver
        self._entries_in_progress: set[str] = set()
        self.sl_manager = sl_manager
        self.sizer = sizer
        self.portfolio = portfolio
        self.alerter = alerter
        self.audit_store = audit_store
        self.mode = mode
        self.trades = trades
        self.protection = protection
        self.check_exposure_limits = check_exposure_limits
        self.fail_reconciliation = fail_reconciliation
        self.finalize_trade_leg = finalize_trade_leg
        self.entry_router = EntryOrderRouter(client, orders)

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
    def _new_correlation_id() -> str:
        return str(uuid4())

    def _order_failure_reason(self, symbol: str, fallback: str) -> str:
        failure_reason = getattr(self.orders, "failure_reason", None)
        if callable(failure_reason):
            return str(failure_reason(symbol, fallback))
        return fallback

    async def _notify_trade_opened(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float | None,
    ):
        if self.alerter:
            await self.alerter.trade_opened_alert(
                self.mode,
                symbol,
                side,
                price,
                quantity,
                stop_loss,
                take_profit,
            )

    async def _notify_trade_completed(
        self, trade: TradeRecord, exit_price: float, reason: str
    ):
        if self.alerter:
            await self.alerter.trade_completed_alert(
                self.mode,
                trade.symbol,
                trade.side,
                exit_price,
                trade.pnl,
                reason,
            )

    async def _notify_trade_failed(self, symbol: str, reason: str):
        if self.alerter:
            await self.alerter.trade_failed_alert(self.mode, symbol, reason)

    async def _report_entry_limit_block(
        self,
        symbol: str,
        reason: str,
        *,
        payload: dict,
    ) -> None:
        expected_policy_block = reason.startswith(
            (
                "opposite-side entry blocked",
                "trade leg already open",
                "max trade legs",
            )
        )
        if expected_policy_block:
            logger.info(f"Entry skipped for {symbol}: {reason}")
            self._audit("trade_skipped_policy", reason, symbol=symbol, payload=payload)
            return
        logger.warning(f"Exposure limit blocked {symbol}: {reason}")
        await self._notify_trade_failed(symbol, reason)
        self._audit(
            "trade_blocked_exposure",
            reason,
            severity="warning",
            symbol=symbol,
            payload=payload,
        )

    async def _validated_entry_fill_price(
        self,
        order: dict,
        symbol: str,
        signal_price: float,
        exit_side: str,
        filled_quantity: float,
        correlation_id: str,
        *,
        strategy: str | None = None,
    ) -> float | None:
        entry_price = await self._fill_resolver.resolve_entry_fill_price(order, symbol)
        if entry_price is None:
            reason = "entry fill price unavailable from exchange"
        else:
            entry_side = "long" if exit_side == "sell" else "short"
            fill_slippage = self._adverse_entry_drift_bps(
                entry_side,
                signal_price,
                entry_price,
            )
            max_slip = self._max_slippage_for(symbol)
            if strategy == "scalp":
                max_slip = min(
                    max_slip,
                    settings.scalp_max_entry_slippage_bps,
                )
            if fill_slippage <= max_slip:
                return entry_price
            reason = (
                f"entry fill slippage above limit: {fill_slippage:.2f}bps "
                f"> {max_slip:.2f}bps"
            )
        await self._handle_unprotected_entry(
            symbol,
            exit_side,
            filled_quantity,
            reason,
            correlation_id,
        )
        return None

    async def _entry_policy_blocks(
        self,
        symbol: str,
        strategy: str | None = None,
        *,
        ignore_reentry_cooldown: bool = False,
    ) -> bool:
        can_trade, reason = self.portfolio.can_trade(symbol, strategy)
        if not can_trade:
            logger.warning(f"Cannot enter {symbol}: {reason}")
            if self.trades.policy_alert_due(reason):
                await self._notify_trade_failed(symbol, reason)
                self._audit("trade_blocked", reason, severity="warning", symbol=symbol)
            return True

        if ignore_reentry_cooldown:
            return False

        reason = self.trades.reentry_cooldown_reason(symbol, strategy)
        if not reason:
            return False
        logger.warning(f"Cannot enter {symbol}: {reason}")
        self._audit("trade_blocked_cooldown", reason, severity="warning", symbol=symbol)
        return True

    async def _block_stale_entry(self, symbol: str, reason: str) -> None:
        logger.warning(f"Order blocked for {symbol}: {reason}")
        self._audit(
            "trade_blocked_price_drift", reason, severity="warning", symbol=symbol
        )
        await self._notify_trade_failed(symbol, reason)

    @staticmethod
    def _max_slippage_for(symbol: str) -> float:
        return settings.max_entry_slippage_bps_per_symbol.get(
            symbol, settings.max_entry_slippage_bps
        )

    async def _funding_rate_reason(self, symbol: str, side: str) -> str:
        try:
            funding_rate = float(await self.client.fetch_funding_rate(symbol))
        except Exception as exc:
            self._audit(
                "funding_rate_unavailable",
                redact_text(f"funding rate unavailable: {exc}"),
                severity="warning",
                symbol=symbol,
            )
            return ""
        unfavorable_rate = funding_rate if side == "long" else -funding_rate
        if unfavorable_rate <= settings.max_unfavorable_funding_rate:
            return ""
        return (
            f"unfavorable funding rate above limit: {unfavorable_rate:.6f} "
            f"> {settings.max_unfavorable_funding_rate:.6f}"
        )

    async def _market_depth_reason(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: str | None,
    ) -> str:
        fetch_order_book = getattr(self.client, "fetch_order_book", None)
        if not callable(fetch_order_book):
            self._audit(
                "market_depth_unavailable",
                "runtime adapter does not provide order-book depth",
                severity="warning",
                symbol=symbol,
            )
            return ""
        try:
            order_book = await fetch_order_book(
                symbol,
                limit=settings.market_depth_levels,
            )
        except Exception as exc:
            return redact_text(f"order-book depth unavailable: {exc}")
        levels = order_book.get("asks" if side == "long" else "bids") or []
        remaining = quantity
        notional = 0.0
        best_price = 0.0
        for level in levels:
            price = float(level[0])
            available = float(level[1])
            if price <= 0 or available <= 0:
                continue
            if best_price == 0:
                best_price = price
            filled = min(remaining, available)
            notional += filled * price
            remaining -= filled
            if remaining <= 1e-12:
                break
        if quantity <= 0 or best_price <= 0 or remaining > 1e-12:
            return "insufficient visible order-book depth for proposed quantity"
        average_price = notional / quantity
        impact_bps = adverse_price_movement_bps(side, best_price, average_price)
        limit = settings.max_market_depth_slippage_bps
        if strategy == "scalp":
            limit = min(limit, settings.scalp_max_market_depth_slippage_bps)
        if impact_bps <= limit:
            return ""
        return (
            f"estimated depth impact above limit: {impact_bps:.2f}bps > {limit:.2f}bps"
        )

    async def _block_market_entry(self, symbol: str, reason: str) -> None:
        logger.info(f"Entry skipped for {symbol}: {reason}")
        self._audit(
            "trade_blocked_market_risk",
            reason,
            severity="warning",
            symbol=symbol,
        )
        await self._notify_trade_failed(symbol, reason)

    async def _entry_price_drift_reason(
        self,
        symbol: str,
        side: str,
        signal_price: float,
        signal_timestamp=None,
        strategy: str | None = None,
    ) -> str:
        if signal_price <= 0:
            return "entry signal price is invalid"
        try:
            ticker = await self.client.fetch_ticker(symbol)
        except Exception as exc:
            return redact_text(f"entry quote unavailable: {exc}")
        quote_keys = (
            ("ask", "last", "mark") if side == "long" else ("bid", "last", "mark")
        )
        executable_price = next(
            (
                float(ticker[key])
                for key in quote_keys
                if ticker.get(key) is not None and float(ticker[key]) > 0
            ),
            None,
        )
        if executable_price is None:
            return "entry quote unavailable: no positive executable price"
        if strategy == "scalp":
            bid = float(ticker.get("bid") or 0)
            ask = float(ticker.get("ask") or 0)
            if bid > 0 and ask >= bid:
                executable_price = (bid + ask) / 2
        drift_bps = self._adverse_entry_drift_bps(
            side,
            signal_price,
            executable_price,
        )
        absolute_drift_bps = (
            abs(executable_price - signal_price) / signal_price * 10_000
        )
        max_drift = self._max_slippage_for(symbol)
        if strategy == "scalp":
            max_drift = min(
                max_drift,
                settings.scalp_max_entry_slippage_bps,
            )
        if absolute_drift_bps > max_drift:
            return (
                f"signal price drift above limit: {absolute_drift_bps:.2f}bps "
                f"> {max_drift:.2f}bps "
                f"(signal={signal_price:.8f}, quote={executable_price:.8f})"
            )
        if drift_bps <= max_drift:
            return ""
        return (
            f"signal price drift above limit: {drift_bps:.2f}bps "
            f"> {max_drift:.2f}bps "
            f"(signal={signal_price:.8f}, quote={executable_price:.8f})"
        )

    @staticmethod
    def _adverse_entry_drift_bps(
        side: str,
        signal_price: float,
        executable_price: float,
    ) -> float:
        return adverse_price_movement_bps(side, signal_price, executable_price)

    async def _entry_preflight_blocks(
        self,
        symbol: str,
        side: str,
        signal_price: float,
        signal_timestamp=None,
        strategy: str | None = None,
        ignore_reentry_cooldown: bool = False,
        market_context: dict | None = None,
    ) -> bool:
        if await self._entry_policy_blocks(
            symbol,
            strategy,
            ignore_reentry_cooldown=ignore_reentry_cooldown,
        ):
            return True
        if self.portfolio.loss_cooldown_seconds > 0:
            last_loss = self.portfolio.last_symbol_loss_at.get(symbol)
            if last_loss is not None:
                if last_loss.tzinfo is None:
                    last_loss = last_loss.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - last_loss).total_seconds()
                remaining = self.portfolio.loss_cooldown_seconds - elapsed
                if remaining > 0:
                    reason = (
                        f"loss cooldown active for {symbol}: {remaining:.0f}s remaining"
                    )
                    logger.warning(f"Cannot enter {symbol}: {reason}")
                    await self._notify_trade_failed(symbol, reason)
                    self._audit(
                        "trade_blocked", reason, severity="warning", symbol=symbol
                    )
                    return True
        funding_reason = await self._funding_rate_reason(symbol, side)
        if funding_reason:
            await self._block_market_entry(symbol, funding_reason)
            return True
        price_drift_reason = await self._entry_price_drift_reason(
            symbol,
            side,
            signal_price,
            signal_timestamp=signal_timestamp,
            strategy=strategy,
        )
        if not price_drift_reason:
            return False
        await self._block_stale_entry(symbol, price_drift_reason)
        return True

    async def _handle_unprotected_entry(
        self,
        symbol: str,
        exit_side: str,
        quantity: float,
        reason: str,
        correlation_id: str,
        *,
        stop_order_id: str | None = None,
        take_profit_order_id: str | None = None,
    ) -> None:
        self.trades.record_exit_time(symbol)
        logger.critical(f"{reason}; attempting emergency flatten for {symbol}")
        self._audit(
            "unprotected_entry",
            reason,
            severity="critical",
            symbol=symbol,
            correlation_id=correlation_id,
            payload={"exit_side": exit_side, "quantity": quantity},
        )
        await self.protection.cancel_protection(
            symbol,
            stop_order_id=stop_order_id,
            take_profit_order_id=take_profit_order_id,
        )
        flatten_order = await self.orders.market_order(
            symbol, exit_side, quantity, reduce_only=True
        )
        confirmed_order = await self._fill_resolver.confirmed_full_fill(
            flatten_order,
            symbol,
            quantity,
        )
        if confirmed_order:
            self._audit(
                "emergency_flattened",
                f"Emergency flatten submitted for {symbol}",
                severity="critical",
                symbol=symbol,
                correlation_id=correlation_id,
                payload={"order": confirmed_order},
            )
        else:
            self.audit_store.activate_emergency_stop(
                f"Emergency flatten failed for {symbol}",
            )
            self._audit(
                "emergency_flatten_failed",
                f"Emergency flatten failed for {symbol}; emergency stop activated",
                severity="critical",
                symbol=symbol,
                correlation_id=correlation_id,
            )
        await self._notify_trade_failed(symbol, reason)

    async def _entry_start_blocked(
        self,
        symbol: str,
        side: str,
        price: float,
        signal_timestamp,
        strategy: str | None,
        ignore_reentry_cooldown: bool = False,
    ) -> bool:
        allowed, reason = self.audit_store.trading_allowed(symbol=symbol)
        if not allowed:
            logger.warning(f"Trading disabled for {symbol}: {reason}")
            await self._notify_trade_failed(symbol, reason)
            self._audit("trade_blocked", reason, severity="warning", symbol=symbol)
            return True
        return await self._entry_preflight_blocks(
            symbol,
            side,
            price,
            signal_timestamp=signal_timestamp,
            strategy=strategy,
            ignore_reentry_cooldown=ignore_reentry_cooldown,
        )

    async def _enter_position(  # noqa: C901
        self,
        symbol: str,
        price: float,
        atr: float,
        side: str,
        leverage: int = 1,
        timeframe: str | None = None,
        signal_timestamp=None,
        strategy: str | None = None,
        ignore_reentry_cooldown: bool = False,
        market_context: dict | None = None,
    ) -> bool:
        position_key = self.trades.position_key(symbol, timeframe, strategy)
        if symbol in self._entries_in_progress:
            logger.warning(
                f"Entry already in progress for {symbol}, skipping duplicate"
            )
            return False
        self._entries_in_progress.add(symbol)
        if await self._entry_start_blocked(
            symbol,
            side,
            price,
            signal_timestamp,
            strategy,
            ignore_reentry_cooldown,
        ):
            self._entries_in_progress.discard(symbol)
            return False

        levels = await self._entry_risk_levels(
            symbol,
            price,
            side,
            atr,
            strategy,
            market_context=market_context,
        )
        if levels is None:
            self._entries_in_progress.discard(symbol)
            return False
        pos_size = self.sizer.calculate(
            price,
            levels.stop_loss,
            leverage,
            side,
            strategy=strategy,
        )
        if pos_size.quantity <= 0:
            logger.warning(f"Position size zero for {symbol}")
            await self._notify_trade_failed(symbol, "position size is zero")
            self._audit(
                "trade_blocked",
                "position size is zero",
                severity="warning",
                symbol=symbol,
            )
            self._entries_in_progress.discard(symbol)
            return False

        depth_reason = await self._market_depth_reason(
            symbol,
            side,
            pos_size.quantity,
            strategy,
        )
        if depth_reason:
            await self._block_market_entry(symbol, depth_reason)
            self._entries_in_progress.discard(symbol)
            return False

        exposure_ok, exposure_reason = self.check_exposure_limits(
            symbol,
            side,
            price,
            pos_size.quantity,
            position_key,
            timeframe=timeframe,
        )
        if not exposure_ok:
            await self._report_entry_limit_block(
                symbol,
                exposure_reason,
                payload={
                    "price": price,
                    "quantity": pos_size.quantity,
                    "position_key": position_key,
                },
            )
            self._entries_in_progress.discard(symbol)
            return False

        correlation_id = self._new_correlation_id()
        execution_id = self._new_correlation_id()
        execution_started_at = datetime.now(timezone.utc).isoformat()
        buy_side = "buy" if side == "long" else "sell"
        order_started = time.monotonic()
        order = await self.entry_router.submit(
            symbol,
            buy_side,
            pos_size.quantity,
            strategy=strategy,
        )
        order_latency_ms = (
            float(order.get("_bot_order_latency_ms"))
            if order and order.get("_bot_order_latency_ms") is not None
            else (time.monotonic() - order_started) * 1000
        )
        if order and order.get("filled", 0) > 0:
            filled_quantity = float(order["filled"])
            exit_side = "sell" if side == "long" else "buy"
            fill_source = (
                "order_payload"
                if self._fill_resolver.positive_order_price(order) is not None
                else "exchange_recovery"
            )
            resolution_started = time.monotonic()
            entry_price = await self._validated_entry_fill_price(
                order,
                symbol,
                price,
                exit_side,
                filled_quantity,
                correlation_id,
                strategy=strategy,
            )
            resolution_latency_ms = (time.monotonic() - resolution_started) * 1000
            if entry_price is None:
                self._record_execution_attempt(
                    execution_id=execution_id,
                    correlation_id=correlation_id,
                    phase="entry",
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy,
                    side=side,
                    status="fill_rejected",
                    expected_price=price,
                    quantity=filled_quantity,
                    order_latency_ms=order_latency_ms,
                    fill_resolution_latency_ms=resolution_latency_ms,
                    fill_source=fill_source,
                    order=order,
                    reason="entry fill validation failed; position flattened",
                    started_at=execution_started_at,
                    completed=True,
                )
                self._entries_in_progress.discard(symbol)
                return False
            slippage_bps = self._adverse_entry_drift_bps(
                side,
                price,
                entry_price,
            )
            policy = self.strategy_registry.get(strategy)
            entry_fee = await self._fill_resolver.order_fee(
                order,
                symbol,
                fallback_notional=entry_price * filled_quantity,
                fallback_fee_bps=policy.estimated_round_trip_fee_bps / 2,
            )
            levels = self._calculate_risk_levels(
                entry_price,
                side,
                atr,
                strategy=strategy,
                market_context=market_context,
            )
            post_fill_cost_reason = await self._strategy_cost_reason(
                symbol,
                side,
                entry_price,
                levels,
                strategy,
            )
            if post_fill_cost_reason:
                self._record_execution_attempt(
                    execution_id=execution_id,
                    correlation_id=correlation_id,
                    phase="entry",
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy,
                    side=side,
                    status="post_fill_cost_rejected",
                    expected_price=price,
                    actual_price=entry_price,
                    quantity=filled_quantity,
                    slippage_bps=slippage_bps,
                    order_latency_ms=order_latency_ms,
                    fill_resolution_latency_ms=resolution_latency_ms,
                    fill_source=fill_source,
                    order=order,
                    reason=post_fill_cost_reason,
                    started_at=execution_started_at,
                    completed=True,
                )
                await self._handle_unprotected_entry(
                    symbol,
                    exit_side,
                    filled_quantity,
                    f"post-fill cost check failed: {post_fill_cost_reason}",
                    correlation_id,
                )
                self._entries_in_progress.discard(symbol)
                return False
            trade = TradeRecord(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=filled_quantity,
                timestamp=datetime.now(),
                timeframe=timeframe,
                strategy=strategy,
                entry_fee=entry_fee,
            )

            protection_started = time.monotonic()
            protection = await self.protection.place_entry_protection(
                symbol,
                side,
                filled_quantity,
                levels,
            )
            protection_latency_ms = (time.monotonic() - protection_started) * 1000
            if protection is None:
                self._record_execution_attempt(
                    execution_id=execution_id,
                    correlation_id=correlation_id,
                    phase="entry",
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy,
                    side=side,
                    status="protection_failed",
                    expected_price=price,
                    actual_price=entry_price,
                    quantity=filled_quantity,
                    slippage_bps=slippage_bps,
                    order_latency_ms=order_latency_ms,
                    fill_resolution_latency_ms=resolution_latency_ms,
                    protection_latency_ms=protection_latency_ms,
                    fill_source=fill_source,
                    order=order,
                    reason="protective order placement failed; position flattened",
                    started_at=execution_started_at,
                    completed=True,
                )
                await self._handle_unprotected_entry(
                    symbol,
                    exit_side,
                    filled_quantity,
                    f"protective order placement failed after {side} entry",
                    correlation_id,
                )
                self._entries_in_progress.discard(symbol)
                return False
            sl_order, tp_order = protection

            stop_order_id = sl_order.get("id", "")
            take_profit_order_id = tp_order.get("id", "") if tp_order else None
            try:
                self.audit_store.record_open_trade(
                    trade,
                    mode=self.mode,
                    correlation_id=correlation_id,
                    stop_loss=levels.stop_loss,
                    take_profit=levels.take_profit,
                    stop_order_id=stop_order_id,
                    take_profit_order_id=take_profit_order_id,
                )
            except Exception as exc:
                self._record_execution_attempt(
                    execution_id=execution_id,
                    correlation_id=correlation_id,
                    phase="entry",
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy,
                    side=side,
                    status="audit_failed",
                    expected_price=price,
                    actual_price=entry_price,
                    quantity=filled_quantity,
                    slippage_bps=slippage_bps,
                    order_latency_ms=order_latency_ms,
                    fill_resolution_latency_ms=resolution_latency_ms,
                    protection_latency_ms=protection_latency_ms,
                    fill_source=fill_source,
                    order=order,
                    reason=f"trade persistence failed: {redact_text(exc)}",
                    started_at=execution_started_at,
                    completed=True,
                )
                await self._handle_unprotected_entry(
                    symbol,
                    exit_side,
                    filled_quantity,
                    f"audit persistence failed after {side} entry: {exc}",
                    correlation_id,
                    stop_order_id=stop_order_id,
                    take_profit_order_id=take_profit_order_id,
                )
                self._entries_in_progress.discard(symbol)
                return False

            self.trades.open_trades[position_key] = trade
            self.trades.trade_correlation_ids[position_key] = correlation_id
            self.protection.active_stops[position_key] = stop_order_id
            if take_profit_order_id:
                self.protection.active_tps[position_key] = take_profit_order_id
            self.portfolio.add_trade(trade)
            self._record_execution_attempt(
                execution_id=execution_id,
                correlation_id=correlation_id,
                phase="entry",
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                side=side,
                status="completed",
                expected_price=price,
                actual_price=entry_price,
                quantity=filled_quantity,
                slippage_bps=slippage_bps,
                order_latency_ms=order_latency_ms,
                fill_resolution_latency_ms=resolution_latency_ms,
                protection_latency_ms=protection_latency_ms,
                fill_source=fill_source,
                order=order,
                started_at=execution_started_at,
                completed=True,
            )

            logger.info(
                f"Entered {side.upper()} {symbol} qty={filled_quantity} "
                f"price={entry_price} "
                f"sl={levels.stop_loss} tp={levels.take_profit}"
            )
            await self._notify_trade_opened(
                symbol,
                side,
                entry_price,
                filled_quantity,
                levels.stop_loss,
                levels.take_profit,
            )
            self._entries_in_progress.discard(symbol)
            return True
        failure_reason = self._order_failure_reason(
            symbol,
            f"market {buy_side} order was not filled",
        )
        self._record_execution_attempt(
            execution_id=execution_id,
            correlation_id=correlation_id,
            phase="entry",
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            side=side,
            status="order_failed",
            expected_price=price,
            quantity=pos_size.quantity,
            order_latency_ms=order_latency_ms,
            order=order,
            reason=failure_reason,
            started_at=execution_started_at,
            completed=True,
        )
        await self._notify_trade_failed(symbol, failure_reason)
        self._audit("trade_failed", failure_reason, severity="error", symbol=symbol)
        self._entries_in_progress.discard(symbol)
        return False

    async def _scalp_cost_reason(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        levels,
    ) -> str:
        max_spread_bps = settings.scalp_max_spread_bps
        estimated_fee_bps = settings.scalp_effective_round_trip_fee_bps
        try:
            ticker = await self.client.fetch_ticker(symbol)
        except Exception as exc:
            return redact_text(f"scalp quote unavailable: {exc}")
        bid = float(ticker.get("bid") or 0)
        ask = float(ticker.get("ask") or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            try:
                order_book = await self.client.fetch_order_book(symbol, limit=5)
                bids = order_book.get("bids") or []
                asks = order_book.get("asks") or []
                bid = float(bids[0][0]) if bids else 0.0
                ask = float(asks[0][0]) if asks else 0.0
            except Exception as exc:
                return redact_text(f"scalp order-book quote unavailable: {exc}")
        if bid <= 0 or ask <= 0 or ask < bid:
            return "scalp quote has invalid bid/ask after order-book fallback"
        midpoint = (bid + ask) / 2
        spread_bps = (ask - bid) / midpoint * 10_000
        if spread_bps > max_spread_bps:
            return (
                f"scalp spread above limit: {spread_bps:.2f}bps "
                f"> {max_spread_bps:.2f}bps"
            )
        target = float(levels.take_profit or entry_price)
        if side == "long":
            target_bps = (target - entry_price) / entry_price * 10_000
        elif side == "short":
            target_bps = (entry_price - target) / entry_price * 10_000
        else:
            return f"invalid scalp side: {side}"
        if target_bps <= 0:
            return f"scalp take-profit is not profitable for {side} entry"
        required_bps = estimated_fee_bps + settings.scalp_min_net_edge_bps + spread_bps
        if target_bps < required_bps:
            return (
                f"scalp target edge too small: {target_bps:.2f}bps "
                f"< {required_bps:.2f}bps including fees and spread"
            )
        return self._after_cost_reward_risk_reason(
            name="scalp",
            entry_price=entry_price,
            stop_loss=float(levels.stop_loss or entry_price),
            target_bps=target_bps,
            fee_bps=estimated_fee_bps,
            spread_bps=spread_bps,
            minimum_after_cost_rr=max(1.0, settings.scalp_risk_reward_ratio * 0.65),
        )

    async def _strategy_cost_reason(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        levels,
        strategy: str | None,
    ) -> str:
        if strategy == "scalp":
            return await self._scalp_cost_reason(
                symbol,
                side,
                entry_price,
                levels,
            )
        policy = self.strategy_registry.get(strategy)
        try:
            ticker = await self.client.fetch_ticker(symbol)
            bid = float(ticker.get("bid") or 0)
            ask = float(ticker.get("ask") or 0)
        except Exception as exc:
            return redact_text(f"entry quote unavailable for cost check: {exc}")
        if bid <= 0 or ask <= 0 or ask < bid:
            try:
                order_book = await self.client.fetch_order_book(symbol, limit=5)
                bids = order_book.get("bids") or []
                asks = order_book.get("asks") or []
                bid = float(bids[0][0]) if bids else 0.0
                ask = float(asks[0][0]) if asks else 0.0
            except Exception as exc:
                return redact_text(f"entry order book unavailable: {exc}")
        if bid <= 0 or ask <= 0 or ask < bid:
            return "entry quote is invalid after order-book fallback"
        midpoint = (bid + ask) / 2
        spread_bps = (ask - bid) / midpoint * 10_000
        if spread_bps > policy.max_spread_bps:
            return (
                f"{policy.name} spread above limit: {spread_bps:.2f}bps "
                f"> {policy.max_spread_bps:.2f}bps"
            )
        target = float(levels.take_profit or entry_price)
        target_bps = (
            (target - entry_price) / entry_price * 10_000
            if side == "long"
            else (entry_price - target) / entry_price * 10_000
        )
        required_bps = policy.required_target_edge_bps + spread_bps
        if target_bps < required_bps:
            return (
                f"{policy.name} target edge too small: {target_bps:.2f}bps "
                f"< {required_bps:.2f}bps after fees and spread"
            )
        return self._after_cost_reward_risk_reason(
            name=policy.name,
            entry_price=entry_price,
            stop_loss=float(levels.stop_loss or entry_price),
            target_bps=target_bps,
            fee_bps=policy.estimated_round_trip_fee_bps,
            spread_bps=spread_bps,
            minimum_after_cost_rr=policy.minimum_after_cost_reward_risk,
        )

    @staticmethod
    def _after_cost_reward_risk_reason(
        *,
        name: str,
        entry_price: float,
        stop_loss: float,
        target_bps: float,
        fee_bps: float,
        spread_bps: float,
        minimum_after_cost_rr: float,
    ) -> str:
        risk_bps = abs(entry_price - stop_loss) / entry_price * 10_000
        if risk_bps <= 0:
            return f"{name} stop-loss is invalid for after-cost " "reward/risk check"
        after_cost_reward_bps = target_bps - fee_bps - spread_bps
        after_cost_rr = after_cost_reward_bps / (risk_bps + spread_bps)
        if after_cost_rr < minimum_after_cost_rr:
            return (
                f"{name} after-cost reward/risk too weak: "
                f"{after_cost_rr:.2f}R < {minimum_after_cost_rr:.2f}R"
            )
        return ""

    async def _entry_risk_levels(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        atr: float,
        strategy: str | None,
        market_context: dict | None = None,
    ):
        levels = self._calculate_risk_levels(
            entry_price,
            side,
            atr,
            strategy=strategy,
            market_context=market_context,
        )
        reason = await self._strategy_cost_reason(
            symbol,
            side,
            entry_price,
            levels,
            strategy,
        )
        if not reason:
            return levels
        await self._block_strategy_entry(symbol, strategy, reason)
        return None

    def _calculate_risk_levels(
        self,
        entry_price: float,
        side: str,
        atr: float,
        *,
        strategy: str | None,
        market_context: dict | None,
    ):
        try:
            return self.sl_manager.calculate(
                entry_price,
                side,
                atr,
                strategy=strategy,
                market_context=market_context,
            )
        except TypeError as exc:
            if "market_context" not in str(exc):
                raise
            return self.sl_manager.calculate(
                entry_price,
                side,
                atr,
                strategy=strategy,
            )

    async def _block_strategy_entry(
        self,
        symbol: str,
        strategy: str | None,
        reason: str,
    ) -> None:
        if strategy == "scalp":
            await self._block_scalp_entry(symbol, reason)
            return
        policy = self.strategy_registry.get(strategy)
        logger.info("{} entry skipped for {}: {}", policy.name, symbol, reason)
        self._audit(
            "strategy_entry_blocked",
            reason,
            symbol=symbol,
            payload={
                "strategy": policy.name,
                "estimated_round_trip_fee_bps": (policy.estimated_round_trip_fee_bps),
                "minimum_net_edge_bps": policy.minimum_net_edge_bps,
                "max_spread_bps": policy.max_spread_bps,
            },
        )

    async def _block_scalp_entry(self, symbol: str, reason: str) -> None:
        logger.info(f"Scalp entry skipped for {symbol}: {reason}")
        self._audit(
            "scalp_entry_blocked",
            reason,
            symbol=symbol,
            payload={
                "max_spread_bps": settings.scalp_max_spread_bps,
                "estimated_round_trip_fee_bps": (
                    settings.scalp_estimated_round_trip_fee_bps
                ),
                "minimum_net_edge_bps": settings.scalp_min_net_edge_bps,
            },
        )

    async def enter_long(
        self,
        symbol: str,
        price: float,
        atr: float,
        leverage: int = 1,
        timeframe: str | None = None,
        signal_timestamp=None,
        strategy: str | None = None,
        ignore_reentry_cooldown: bool = False,
        market_context: dict | None = None,
    ) -> bool:
        return await self._enter_position(
            symbol,
            price,
            atr,
            "long",
            leverage,
            timeframe,
            signal_timestamp=signal_timestamp,
            strategy=strategy,
            ignore_reentry_cooldown=ignore_reentry_cooldown,
            market_context=market_context,
        )

    async def enter_short(
        self,
        symbol: str,
        price: float,
        atr: float,
        leverage: int = 1,
        timeframe: str | None = None,
        signal_timestamp=None,
        strategy: str | None = None,
        ignore_reentry_cooldown: bool = False,
        market_context: dict | None = None,
    ) -> bool:
        return await self._enter_position(
            symbol,
            price,
            atr,
            "short",
            leverage,
            timeframe,
            signal_timestamp=signal_timestamp,
            strategy=strategy,
            ignore_reentry_cooldown=ignore_reentry_cooldown,
            market_context=market_context,
        )

    async def exit_position(self, position_key: str, reason: str = "manual"):
        trade = self.trades.open_trades.get(position_key)
        if not trade:
            return
        symbol = trade.symbol

        execution_id = self._new_correlation_id()
        correlation_id = self.trades.trade_correlation_ids.get(position_key, "")
        execution_started_at = datetime.now(timezone.utc).isoformat()
        exit_side = "sell" if trade.side == "long" else "buy"
        order_started = time.monotonic()
        order = await self.orders.market_order(
            symbol,
            exit_side,
            trade.quantity,
            reduce_only=True,
        )
        order_latency_ms = (
            float(order.get("_bot_order_latency_ms"))
            if order and order.get("_bot_order_latency_ms") is not None
            else (time.monotonic() - order_started) * 1000
        )
        resolution_started = time.monotonic()
        confirmed_order = await self._fill_resolver.confirmed_full_fill(
            order, symbol, trade.quantity
        )
        resolution_latency_ms = (time.monotonic() - resolution_started) * 1000
        if confirmed_order:
            exit_price = await self._fill_resolver.resolve_exit_price(
                confirmed_order,
                symbol,
                trade.entry_price,
            )
            policy = self.strategy_registry.get(trade.strategy)
            exit_fee = await self._fill_resolver.order_fee(
                confirmed_order,
                symbol,
                fallback_notional=exit_price * trade.quantity,
                fallback_fee_bps=policy.estimated_round_trip_fee_bps / 2,
            )
            self.portfolio.close_trade(
                trade,
                exit_price,
                reason,
                exit_fee=exit_fee,
            )
            if correlation_id:
                self.audit_store.record_closed_trade(
                    trade,
                    mode=self.mode,
                    correlation_id=correlation_id,
                )
            self._record_execution_attempt(
                execution_id=execution_id,
                correlation_id=correlation_id or None,
                phase="exit",
                symbol=symbol,
                timeframe=trade.timeframe,
                strategy=trade.strategy,
                side=exit_side,
                status="completed",
                expected_price=None,
                actual_price=exit_price,
                quantity=trade.quantity,
                order_latency_ms=order_latency_ms,
                fill_resolution_latency_ms=resolution_latency_ms,
                fill_source=(
                    "order_payload"
                    if self._fill_resolver.positive_order_price(confirmed_order)
                    is not None
                    else "market_fallback"
                ),
                order=confirmed_order,
                reason=reason,
                started_at=execution_started_at,
                completed=True,
            )
            self.trades.open_trades.pop(position_key, None)
            self.trades.trade_correlation_ids.pop(position_key, None)
            self.trades.record_exit_time(symbol)
            await self._notify_trade_completed(trade, exit_price, reason)
            if position_key in self.protection.active_stops:
                await self.orders.cancel_order(
                    symbol,
                    self.protection.active_stops[position_key],
                    conditional=True,
                )
                self.protection.active_stops.pop(position_key, None)
            if position_key in self.protection.active_tps:
                await self.orders.cancel_order(
                    symbol, self.protection.active_tps[position_key]
                )
                self.protection.active_tps.pop(position_key, None)
        else:
            exchange_positions = await self._fetch_positions_or_none()
            # Treat the leg as still open unless the exchange positively
            # reports it gone AND the exit order is no longer working. An
            # unreadable snapshot (None) or a live exit order both mean
            # "unknown", and unknown must never close the book.
            position_exists = (
                exchange_positions is None
                or self._exit_order_is_working(order)
                or self._exchange_position_still_open(
                    exchange_positions, symbol, trade.quantity
                )
            )
            if not position_exists:
                exit_price = trade.entry_price
                self.portfolio.close_trade(
                    trade,
                    exit_price,
                    reason,
                )
                if correlation_id:
                    self.audit_store.record_closed_trade(
                        trade,
                        mode=self.mode,
                        correlation_id=correlation_id,
                    )
                self._record_execution_attempt(
                    execution_id=execution_id,
                    correlation_id=correlation_id or None,
                    phase="exit",
                    symbol=symbol,
                    timeframe=trade.timeframe,
                    strategy=trade.strategy,
                    side=exit_side,
                    status="already_closed",
                    expected_price=None,
                    actual_price=exit_price,
                    quantity=trade.quantity,
                    order_latency_ms=order_latency_ms,
                    fill_resolution_latency_ms=resolution_latency_ms,
                    order=order,
                    reason=f"{reason} (position already closed on exchange)",
                    started_at=execution_started_at,
                    completed=True,
                )
                self.trades.open_trades.pop(position_key, None)
                self.trades.trade_correlation_ids.pop(position_key, None)
                self.trades.record_exit_time(symbol)
                await self._notify_trade_completed(trade, exit_price, reason)
                if position_key in self.protection.active_stops:
                    await self.orders.cancel_order(
                        symbol,
                        self.protection.active_stops[position_key],
                        conditional=True,
                    )
                    self.protection.active_stops.pop(position_key, None)
                if position_key in self.protection.active_tps:
                    await self.orders.cancel_order(
                        symbol, self.protection.active_tps[position_key]
                    )
                    self.protection.active_tps.pop(position_key, None)
                return
            msg = f"exit order failed for {symbol}"
            self._record_execution_attempt(
                execution_id=execution_id,
                correlation_id=correlation_id or None,
                phase="exit",
                symbol=symbol,
                timeframe=trade.timeframe,
                strategy=trade.strategy,
                side=exit_side,
                status="order_failed",
                expected_price=None,
                quantity=trade.quantity,
                order_latency_ms=order_latency_ms,
                fill_resolution_latency_ms=resolution_latency_ms,
                order=order,
                reason=msg,
                started_at=execution_started_at,
                completed=True,
            )
            logger.error(msg)
            self._audit("trade_exit_failed", msg, severity="critical", symbol=symbol)
            await self._notify_trade_failed(symbol, msg)

    async def partial_exit_position(
        self,
        position_key: str,
        fraction: float,
        reason: str,
    ) -> bool:
        trade = self.trades.open_trades.get(position_key)
        if trade is None or not 0 < fraction < 1:
            return False
        quantity = trade.quantity * fraction
        exit_side = "sell" if trade.side == "long" else "buy"
        order = await self.orders.market_order(
            trade.symbol,
            exit_side,
            quantity,
            reduce_only=True,
        )
        confirmed = await self._fill_resolver.confirmed_full_fill(
            order,
            trade.symbol,
            quantity,
        )
        if not confirmed:
            return False
        filled = float(confirmed.get("filled") or quantity)
        remaining = max(trade.quantity - filled, 0.0)
        if remaining <= 0:
            return False
        exit_price = await self._fill_resolver.resolve_exit_price(
            confirmed,
            trade.symbol,
            trade.entry_price,
        )
        policy = self.strategy_registry.get(trade.strategy)
        exit_fee = await self._fill_resolver.order_fee(
            confirmed,
            trade.symbol,
            fallback_notional=exit_price * filled,
            fallback_fee_bps=policy.estimated_round_trip_fee_bps / 2,
        )
        entry_fee_share = trade.entry_fee * filled / trade.quantity
        partial_trade = replace(
            trade,
            quantity=filled,
            entry_fee=entry_fee_share,
        )
        self.portfolio.close_trade(
            partial_trade,
            exit_price,
            reason,
            exit_fee=exit_fee,
        )
        trade.entry_fee = max(trade.entry_fee - entry_fee_share, 0.0)
        correlation_id = self.trades.trade_correlation_ids.get(position_key, "")
        if correlation_id:
            self.audit_store.record_closed_trade(
                partial_trade,
                mode=self.mode,
                correlation_id=f"{correlation_id}:partial:{uuid4().hex[:8]}",
            )

        stop_loss, take_profit = self.audit_store.get_trade_protection_levels(
            correlation_id
        )
        old_stop = self.protection.active_stops.get(position_key)
        old_target = self.protection.active_tps.get(position_key)
        trade.quantity = remaining
        levels = StopLossLevels(
            stop_loss=float(stop_loss or trade.entry_price),
            take_profit=float(take_profit) if take_profit is not None else None,
        )
        protection = await self.protection.place_entry_protection(
            trade.symbol,
            trade.side,
            remaining,
            levels,
        )
        if protection is None:
            logger.critical(
                "Partial exit left {} without replacement protection; flattening",
                trade.symbol,
            )
            await self.exit_position(
                position_key,
                "partial protection replacement failed",
            )
            return False
        sl_order, tp_order = protection
        stop_id = str(sl_order.get("id") or "")
        target_id = str(tp_order.get("id") or "") if tp_order else None
        await self.protection.cancel_protection(
            trade.symbol,
            old_stop,
            old_target,
        )
        self.protection.track_protection(position_key, stop_id, target_id)
        if correlation_id:
            self.audit_store.update_open_trade_state(
                correlation_id,
                quantity=trade.quantity,
                entry_fee=trade.entry_fee,
                stop_loss=levels.stop_loss,
                take_profit=levels.take_profit,
                stop_order_id=stop_id,
                take_profit_order_id=target_id,
            )
        self.audit_store.safe_record_event(
            "scalp_partial_profit",
            "Scalp position partially reduced",
            symbol=trade.symbol,
            mode=self.mode,
            correlation_id=correlation_id or None,
            payload={
                "position_key": position_key,
                "filled_quantity": filled,
                "remaining_quantity": remaining,
                "exit_price": exit_price,
                "reason": reason,
            },
        )
        return True

    def _record_execution_attempt(
        self,
        *,
        execution_id: str,
        correlation_id: str | None,
        phase: str,
        symbol: str,
        timeframe: str | None,
        strategy: str | None,
        side: str,
        status: str,
        expected_price: float | None,
        quantity: float,
        order_latency_ms: float | None,
        order: dict | None,
        reason: str = "",
        actual_price: float | None = None,
        slippage_bps: float | None = None,
        fill_resolution_latency_ms: float | None = None,
        protection_latency_ms: float | None = None,
        fill_source: str | None = None,
        started_at: str | None = None,
        completed: bool = False,
    ) -> None:
        try:
            self.audit_store.record_execution_attempt(
                execution_id=execution_id,
                correlation_id=correlation_id,
                phase=phase,
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                side=side,
                status=status,
                expected_price=expected_price,
                actual_price=actual_price,
                quantity=quantity,
                slippage_bps=slippage_bps,
                order_latency_ms=order_latency_ms,
                fill_resolution_latency_ms=fill_resolution_latency_ms,
                protection_latency_ms=protection_latency_ms,
                fill_source=fill_source,
                order_id=str(order.get("id") or "") if order else None,
                recovered_order=bool(order and order.get("_bot_recovered")),
                reason=reason,
                started_at=started_at,
                completed=completed,
            )
        except Exception as exc:
            self._audit(
                "execution_metrics_persistence_failed",
                "Execution metrics could not be persisted",
                severity="warning",
                symbol=symbol,
                correlation_id=correlation_id,
                payload={"phase": phase, "status": status, "error": redact_text(exc)},
            )

    async def _close_symbol_positions(self, symbol: str, reason: str) -> bool:
        symbol_trades = self.trades.trades_for_symbol(symbol)
        if not symbol_trades:
            return True
        sides = {trade.side for _, trade in symbol_trades}
        if len(sides) != 1:
            message = f"cannot aggregate mixed-side exits for {symbol}"
            await self.fail_reconciliation(
                "aggregate_exit_side_mismatch",
                message,
                symbol=symbol,
            )
            return False

        trade_side = next(iter(sides))
        exit_side = "sell" if trade_side == "long" else "buy"
        quantity = sum(trade.quantity for _, trade in symbol_trades)
        order = await self.orders.market_order(
            symbol, exit_side, quantity, reduce_only=True
        )
        confirmed_order = await self._fill_resolver.confirmed_full_fill(
            order, symbol, quantity
        )
        if confirmed_order is None:
            message = f"aggregate exit order failed for {symbol}"
            logger.critical(message)
            self.audit_store.activate_emergency_stop(message)
            self._audit(
                "aggregate_exit_failed",
                message,
                severity="critical",
                symbol=symbol,
                payload={
                    "quantity": quantity,
                    "position_keys": [key for key, _ in symbol_trades],
                },
            )
            await self._notify_trade_failed(symbol, message)
            return False

        fallback_price = (
            sum(trade.entry_price * trade.quantity for _, trade in symbol_trades)
            / quantity
        )
        exit_price = await self._fill_resolver.resolve_exit_price(
            confirmed_order,
            symbol,
            fallback_price,
        )
        for position_key, _ in list(symbol_trades):
            await self.finalize_trade_leg(
                position_key,
                exit_price=exit_price,
                reason=reason,
                correlation_id=self.trades.trade_correlation_ids.get(position_key, ""),
            )
        return True

    async def _fetch_positions_or_none(self) -> list[dict] | None:
        """Exchange positions, or None when the exchange could not be read.

        A failed read must stay distinguishable from a genuinely empty list:
        callers that decide whether a position is still open would otherwise
        read an API error as "flat" and drop a live position from the book.
        """
        try:
            return await self.client.fetch_positions()
        except Exception as exc:
            logger.warning(f"position snapshot unavailable: {exc}")
            return None

    @staticmethod
    def _exit_order_is_working(order: dict | None) -> bool:
        """True when the exit order is still live on the exchange.

        An unfilled order that has not reached a terminal state may still fill,
        so the position it is closing must not be assumed gone.
        """
        if not order:
            return False
        status = str(order.get("status") or "").strip().lower()
        return status in {"open", "new", "pending", "partially_filled", "accepted"}

    @staticmethod
    def _exchange_position_still_open(
        positions: list[dict],
        symbol: str,
        expected_quantity: float,
    ) -> bool:
        for position in positions:
            pos_symbol = str(
                position.get("symbol") or position.get("info", {}).get("symbol") or ""
            )
            if pos_symbol.upper() != symbol.upper():
                continue
            for key in ("contracts",):
                value = position.get(key)
                if value is None or value == "":
                    continue
                return abs(float(value)) >= expected_quantity * 0.5
            info = position.get("info", {})
            for key in ("positionAmt", "positionamt"):
                value = info.get(key)
                if value is None or value == "":
                    continue
                return abs(float(value)) >= expected_quantity * 0.5
            return False
        return False

    async def close_all(self):
        symbols = {trade.symbol for trade in self.trades.open_trades.values()}
        failed: list[str] = []
        for symbol in symbols:
            try:
                closed = await self._close_symbol_positions(symbol, "close_all")
            except Exception as exc:
                message = f"close_all failed for {symbol}: {redact_text(exc)}"
                logger.critical(message)
                self._audit(
                    "close_all_symbol_failed",
                    message,
                    severity="critical",
                    symbol=symbol,
                )
                failed.append(symbol)
                continue
            if not closed:
                failed.append(symbol)
        if failed:
            self.audit_store.activate_emergency_stop(
                "close_all could not close: " + ", ".join(sorted(failed))
            )
            return False
        return True
