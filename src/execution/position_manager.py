# Position manager — manages position lifecycle: entry, SL/TP placement, exit
from datetime import datetime
from uuid import uuid4

from loguru import logger

from src.audit import AuditStore
from src.config import settings
from src.exchange.client import ExchangeClient
from src.execution.order_manager import OrderManager
from src.monitoring.alerter import Alerter
from src.risk.portfolio import PortfolioManager, TradeRecord
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager


class PositionManager:
    # Wire together all dependencies for position management
    def __init__(
        self,
        client: ExchangeClient,
        order_mgr: OrderManager,
        pos_sizer: PositionSizer,
        sl_mgr: StopLossManager,
        portfolio: PortfolioManager,
        alerter: Alerter | None = None,
        mode: str = "trade",
        audit_store: AuditStore | None = None,
    ):
        self.client = client
        self.orders = order_mgr
        self.sizer = pos_sizer
        self.sl_manager = sl_mgr
        self.portfolio = portfolio
        self.alerter = alerter
        self.mode = mode
        self.audit_store = audit_store or AuditStore()
        self.open_trades: dict[str, TradeRecord] = {}
        self.active_stops: dict[str, str] = {}
        self.active_tps: dict[str, str] = {}
        self.trade_correlation_ids: dict[str, str] = {}

    # Enter a long position: check limits, size, place market order + SL/TP
    async def enter_long(
        self,
        symbol: str,
        price: float,
        atr: float,
        leverage: int = 1,
        timeframe: str | None = None,
    ) -> bool:
        position_key = self.position_key(symbol, timeframe)
        allowed, reason = self.audit_store.trading_allowed()
        if not allowed:
            logger.warning(f"Trading disabled for {symbol}: {reason}")
            await self._notify_trade_failed(symbol, reason)
            self._audit("trade_blocked", reason, severity="warning", symbol=symbol)
            return False

        can_trade, reason = self.portfolio.can_trade()
        if not can_trade:
            logger.warning(f"Cannot enter {symbol}: {reason}")
            await self._notify_trade_failed(symbol, reason)
            self._audit("trade_blocked", reason, severity="warning", symbol=symbol)
            return False

        levels = self.sl_manager.calculate(price, "long", atr)
        pos_size = self.sizer.calculate(price, levels.stop_loss, leverage, "long")
        if pos_size.quantity <= 0:
            logger.warning(f"Position size zero for {symbol}")
            await self._notify_trade_failed(symbol, "position size is zero")
            self._audit(
                "trade_blocked",
                "position size is zero",
                severity="warning",
                symbol=symbol,
            )
            return False

        exposure_ok, exposure_reason = self._check_exposure_limits(
            symbol, price, pos_size.quantity, position_key
        )
        if not exposure_ok:
            logger.warning(f"Exposure limit blocked {symbol}: {exposure_reason}")
            await self._notify_trade_failed(symbol, exposure_reason)
            self._audit(
                "trade_blocked_exposure",
                exposure_reason,
                severity="warning",
                symbol=symbol,
                payload={
                    "price": price,
                    "quantity": pos_size.quantity,
                    "position_key": position_key,
                },
            )
            return False

        order = await self.orders.market_order(symbol, "buy", pos_size.quantity)
        if order and order.get("filled", 0) > 0:
            correlation_id = self._new_correlation_id()
            entry_price = self._order_price(order, price)
            trade = TradeRecord(
                symbol=symbol,
                side="long",
                entry_price=entry_price,
                quantity=pos_size.quantity,
                timestamp=datetime.now(),
                timeframe=timeframe,
            )

            sl_order = await self.orders.stop_loss_order(
                symbol, "sell", pos_size.quantity, levels.stop_loss
            )
            tp_order = None
            if levels.take_profit is not None:
                tp_order = await self.orders.take_profit_order(
                    symbol, "sell", pos_size.quantity, levels.take_profit
                )

            if not sl_order or (levels.take_profit is not None and not tp_order):
                await self._handle_unprotected_entry(
                    symbol,
                    "sell",
                    pos_size.quantity,
                    "protective order placement failed after long entry",
                    correlation_id,
                )
                return False

            self.open_trades[position_key] = trade
            self.trade_correlation_ids[position_key] = correlation_id
            self.active_stops[position_key] = sl_order.get("id", "")
            if tp_order:
                self.active_tps[position_key] = tp_order.get("id", "")
            self.portfolio.add_trade(trade)
            self.audit_store.record_open_trade(
                trade,
                mode=self.mode,
                correlation_id=correlation_id,
                stop_loss=levels.stop_loss,
                take_profit=levels.take_profit,
                stop_order_id=self.active_stops[position_key],
                take_profit_order_id=self.active_tps.get(position_key),
            )

            logger.info(
                f"Entered LONG {symbol} qty={pos_size.quantity} price={entry_price} "
                f"sl={levels.stop_loss} tp={levels.take_profit}"
            )
            await self._notify_trade_opened(
                symbol,
                "long",
                entry_price,
                pos_size.quantity,
                levels.stop_loss,
                levels.take_profit,
            )
            return True
        failure_reason = self._order_failure_reason(
            symbol, "market buy order was not filled"
        )
        await self._notify_trade_failed(symbol, failure_reason)
        self._audit(
            "trade_failed",
            failure_reason,
            severity="error",
            symbol=symbol,
        )
        return False

    # Enter a short position: mirror of enter_long with inverted sides
    async def enter_short(
        self,
        symbol: str,
        price: float,
        atr: float,
        leverage: int = 1,
        timeframe: str | None = None,
    ) -> bool:
        position_key = self.position_key(symbol, timeframe)
        allowed, reason = self.audit_store.trading_allowed()
        if not allowed:
            logger.warning(f"Trading disabled for {symbol}: {reason}")
            await self._notify_trade_failed(symbol, reason)
            self._audit("trade_blocked", reason, severity="warning", symbol=symbol)
            return False

        can_trade, reason = self.portfolio.can_trade()
        if not can_trade:
            logger.warning(f"Cannot enter {symbol}: {reason}")
            await self._notify_trade_failed(symbol, reason)
            self._audit("trade_blocked", reason, severity="warning", symbol=symbol)
            return False

        levels = self.sl_manager.calculate(price, "short", atr)
        pos_size = self.sizer.calculate(price, levels.stop_loss, leverage, "short")
        if pos_size.quantity <= 0:
            logger.warning(f"Position size zero for {symbol}")
            await self._notify_trade_failed(symbol, "position size is zero")
            self._audit(
                "trade_blocked",
                "position size is zero",
                severity="warning",
                symbol=symbol,
            )
            return False

        exposure_ok, exposure_reason = self._check_exposure_limits(
            symbol, price, pos_size.quantity, position_key
        )
        if not exposure_ok:
            logger.warning(f"Exposure limit blocked {symbol}: {exposure_reason}")
            await self._notify_trade_failed(symbol, exposure_reason)
            self._audit(
                "trade_blocked_exposure",
                exposure_reason,
                severity="warning",
                symbol=symbol,
                payload={
                    "price": price,
                    "quantity": pos_size.quantity,
                    "position_key": position_key,
                },
            )
            return False

        order = await self.orders.market_order(symbol, "sell", pos_size.quantity)
        if order and order.get("filled", 0) > 0:
            correlation_id = self._new_correlation_id()
            entry_price = self._order_price(order, price)
            trade = TradeRecord(
                symbol=symbol,
                side="short",
                entry_price=entry_price,
                quantity=pos_size.quantity,
                timestamp=datetime.now(),
                timeframe=timeframe,
            )

            sl_order = await self.orders.stop_loss_order(
                symbol, "buy", pos_size.quantity, levels.stop_loss
            )
            tp_order = None
            if levels.take_profit is not None:
                tp_order = await self.orders.take_profit_order(
                    symbol, "buy", pos_size.quantity, levels.take_profit
                )

            if not sl_order or (levels.take_profit is not None and not tp_order):
                await self._handle_unprotected_entry(
                    symbol,
                    "buy",
                    pos_size.quantity,
                    "protective order placement failed after short entry",
                    correlation_id,
                )
                return False

            self.open_trades[position_key] = trade
            self.trade_correlation_ids[position_key] = correlation_id
            self.active_stops[position_key] = sl_order.get("id", "")
            if tp_order:
                self.active_tps[position_key] = tp_order.get("id", "")
            self.portfolio.add_trade(trade)
            self.audit_store.record_open_trade(
                trade,
                mode=self.mode,
                correlation_id=correlation_id,
                stop_loss=levels.stop_loss,
                take_profit=levels.take_profit,
                stop_order_id=self.active_stops[position_key],
                take_profit_order_id=self.active_tps.get(position_key),
            )

            logger.info(
                f"Entered SHORT {symbol} qty={pos_size.quantity} price={entry_price} "
                f"sl={levels.stop_loss} tp={levels.take_profit}"
            )
            await self._notify_trade_opened(
                symbol,
                "short",
                entry_price,
                pos_size.quantity,
                levels.stop_loss,
                levels.take_profit,
            )
            return True
        failure_reason = self._order_failure_reason(
            symbol, "market sell order was not filled"
        )
        await self._notify_trade_failed(symbol, failure_reason)
        self._audit(
            "trade_failed",
            failure_reason,
            severity="error",
            symbol=symbol,
        )
        return False

    # Exit a position: market order, record PnL, cancel related SL/TP orders
    async def exit_position(self, position_key: str, reason: str = "manual"):
        trade = self.open_trades.get(position_key)
        if not trade:
            return
        symbol = trade.symbol

        exit_side = "sell" if trade.side == "long" else "buy"
        order = await self.orders.market_order(
            symbol, exit_side, trade.quantity, reduce_only=True
        )
        if order:
            exit_price = float(order.get("price", 0)) or float(order.get("average", 0))
            self.portfolio.close_trade(trade, exit_price, reason)
            del self.open_trades[position_key]
            correlation_id = self.trade_correlation_ids.pop(position_key, "")
            if correlation_id:
                self.audit_store.record_closed_trade(
                    trade, mode=self.mode, correlation_id=correlation_id
                )
            await self._notify_trade_completed(trade, exit_price, reason)
            if position_key in self.active_stops:
                await self.orders.cancel_order(symbol, self.active_stops[position_key])
                del self.active_stops[position_key]
            if position_key in self.active_tps:
                await self.orders.cancel_order(symbol, self.active_tps[position_key])
                del self.active_tps[position_key]
        else:
            msg = f"exit order failed for {symbol}"
            logger.error(msg)
            self._audit("trade_exit_failed", msg, severity="critical", symbol=symbol)
            await self._notify_trade_failed(symbol, msg)

    # Close every open position (e.g. on shutdown)
    async def close_all(self):
        for symbol in list(self.open_trades.keys()):
            await self.exit_position(symbol, "close_all")

    async def reconcile_exchange_state(self) -> bool:
        positions = await self._fetch_positions_for_reconciliation()
        if positions is None:
            return False

        exchange_symbols, unmanaged = self._classify_exchange_positions(positions)
        if unmanaged:
            reason = f"unmanaged exchange positions detected: {unmanaged}"
            await self._fail_reconciliation(
                "unmanaged_positions",
                reason,
                payload={"positions": unmanaged},
            )
            return False

        self._clear_missing_exchange_positions(exchange_symbols)

        if not await self._verify_all_protective_orders():
            return False

        self._audit("reconciliation_ok", "Exchange state reconciled")
        return True

    async def _fetch_positions_for_reconciliation(self) -> list[dict] | None:
        try:
            return await self.client.fetch_positions()
        except Exception as exc:
            reason = f"position reconciliation failed: {exc}"
            await self._fail_reconciliation("reconciliation_failed", reason)
            return None

    def _classify_exchange_positions(
        self, positions: list[dict]
    ) -> tuple[set[str], list[dict]]:
        unmanaged = []
        exchange_symbols = set()
        for position in positions:
            symbol = self._normalize_symbol(
                position.get("symbol") or position.get("info", {}).get("symbol")
            )
            size = self._position_size(position)
            if abs(size) <= 0 or not symbol:
                continue
            exchange_symbols.add(symbol)
            if not self._has_open_trade_for_symbol(symbol):
                unmanaged.append({"symbol": symbol, "size": size})
        return exchange_symbols, unmanaged

    def _clear_missing_exchange_positions(self, exchange_symbols: set[str]) -> None:
        missing_position_keys = [
            key
            for key, trade in self.open_trades.items()
            if trade.symbol not in exchange_symbols
        ]
        for position_key in missing_position_keys:
            trade = self.open_trades[position_key]
            correlation_id = self.trade_correlation_ids.get(position_key, "")
            reason = "audited open trade is no longer open on exchange"
            logger.warning(f"{trade.symbol}: {reason}")
            if correlation_id:
                self.audit_store.mark_trade_status(
                    correlation_id, "reconciled_missing", reason=reason
                )
            self.open_trades.pop(position_key, None)
            self.active_stops.pop(position_key, None)
            self.active_tps.pop(position_key, None)
            self.trade_correlation_ids.pop(position_key, None)

    async def _verify_all_protective_orders(self) -> bool:
        for position_key, trade in list(self.open_trades.items()):
            if await self._verify_protective_orders(position_key):
                continue
            reason = f"missing protective orders for {trade.symbol}"
            await self._fail_reconciliation(
                "missing_protection",
                reason,
                symbol=trade.symbol,
                correlation_id=self.trade_correlation_ids.get(position_key),
                payload={
                    "position_key": position_key,
                    "stop_order_id": self.active_stops.get(position_key),
                    "take_profit_order_id": self.active_tps.get(position_key),
                },
            )
            return False
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
        logger.critical(reason)
        self.audit_store.activate_emergency_stop(reason)
        self._audit(
            event_type,
            reason,
            severity="critical",
            symbol=symbol,
            correlation_id=correlation_id,
            payload=payload,
        )
        if self.alerter:
            await self.alerter.error_alert(reason)

    def restore_open_trades_from_audit(self) -> int:
        restored = 0
        for row in self.audit_store.load_open_trades(self.mode):
            symbol = self._normalize_symbol(row["symbol"])
            timeframe = row.get("timeframe")
            trade = TradeRecord(
                symbol=symbol,
                side=str(row["side"]),
                entry_price=float(row["entry_price"]),
                quantity=float(row["quantity"]),
                timestamp=datetime.fromisoformat(str(row["opened_at"])),
                timeframe=str(timeframe) if timeframe else None,
            )
            position_key = self.position_key(symbol, trade.timeframe)
            correlation_id = str(row["correlation_id"])
            self.open_trades[position_key] = trade
            self.trade_correlation_ids[position_key] = correlation_id
            stop_order_id = row.get("stop_order_id")
            take_profit_order_id = row.get("take_profit_order_id")
            if stop_order_id:
                self.active_stops[position_key] = str(stop_order_id)
            if take_profit_order_id:
                self.active_tps[position_key] = str(take_profit_order_id)
            self.portfolio.add_trade(trade)
            restored += 1

        if restored:
            logger.warning(f"Restored {restored} open trade(s) from audit store")
            self._audit(
                "open_trades_restored",
                f"Restored {restored} open trade(s) from audit store",
                severity="warning",
                payload={"count": restored},
            )
        return restored

    def _check_exposure_limits(
        self, symbol: str, entry_price: float, quantity: float, position_key: str
    ) -> tuple[bool, str]:
        if position_key in self.open_trades:
            return False, f"position already open: {position_key}"

        open_count = len(self.open_trades)
        if open_count >= settings.max_open_positions:
            return False, f"max open positions reached: {open_count}"

        account = self.portfolio.account
        if account is None or account.total_equity <= 0:
            return False, "account equity unavailable for exposure check"

        proposed_notional = entry_price * quantity
        total_notional = self._open_notional() + proposed_notional
        max_total = account.total_equity * settings.max_total_open_notional_pct
        if total_notional > max_total:
            return (
                False,
                "total exposure limit exceeded: "
                f"{total_notional:.2f} > {max_total:.2f}",
            )

        symbol_notional = self._open_notional(symbol=symbol) + proposed_notional
        max_symbol = account.total_equity * settings.max_symbol_open_notional_pct
        if symbol_notional > max_symbol:
            return (
                False,
                "symbol exposure limit exceeded: "
                f"{symbol_notional:.2f} > {max_symbol:.2f}",
            )
        return True, "ok"

    def _open_notional(self, symbol: str | None = None) -> float:
        total = 0.0
        for trade in self.open_trades.values():
            if symbol and trade.symbol != symbol:
                continue
            total += trade.entry_price * trade.quantity
        return total

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

    def _order_failure_reason(self, symbol: str, fallback: str) -> str:
        failure_reason = getattr(self.orders, "failure_reason", None)
        if callable(failure_reason):
            return str(failure_reason(symbol, fallback))
        return fallback

    @staticmethod
    def _new_correlation_id() -> str:
        return str(uuid4())

    @staticmethod
    def _order_price(order: dict, fallback: float) -> float:
        return float(order.get("average") or order.get("price") or fallback)

    @staticmethod
    def _position_size(position: dict) -> float:
        for key in ("contracts",):
            value = position.get(key)
            if value not in (None, ""):
                assert value is not None
                return float(value)
        info = position.get("info", {})
        for key in ("positionAmt", "positionAmt".lower()):
            value = info.get(key)
            if value not in (None, ""):
                assert value is not None
                return float(value)
        return 0.0

    @staticmethod
    def _normalize_symbol(symbol: object) -> str:
        raw = str(symbol or "")
        if ":" in raw:
            raw = raw.split(":", 1)[0]
        return raw.replace("/", "").upper()

    async def _verify_protective_orders(self, position_key: str) -> bool:
        trade = self.open_trades[position_key]
        stop_order_id = self.active_stops.get(position_key)
        take_profit_order_id = self.active_tps.get(position_key)
        if not stop_order_id:
            return False

        try:
            open_orders = await self.client.fetch_open_orders(trade.symbol)
        except Exception as exc:
            reason = f"open order reconciliation failed for {trade.symbol}: {exc}"
            logger.critical(reason)
            self._audit(
                "open_order_reconciliation_failed",
                reason,
                severity="critical",
                symbol=trade.symbol,
            )
            return False

        open_order_ids = {str(order.get("id", "")) for order in open_orders}
        if stop_order_id not in open_order_ids:
            return False
        if take_profit_order_id and take_profit_order_id not in open_order_ids:
            return False
        return True

    @staticmethod
    def position_key(symbol: str, timeframe: str | None = None) -> str:
        normalized_symbol = PositionManager._normalize_symbol(symbol)
        if settings.position_scope == "symbol_timeframe" and timeframe:
            return f"{normalized_symbol}:{timeframe}"
        return normalized_symbol

    def _has_open_trade_for_symbol(self, symbol: str) -> bool:
        return any(trade.symbol == symbol for trade in self.open_trades.values())

    async def _handle_unprotected_entry(
        self,
        symbol: str,
        exit_side: str,
        quantity: float,
        reason: str,
        correlation_id: str,
    ) -> None:
        logger.critical(f"{reason}; attempting emergency flatten for {symbol}")
        self._audit(
            "unprotected_entry",
            reason,
            severity="critical",
            symbol=symbol,
            correlation_id=correlation_id,
            payload={"exit_side": exit_side, "quantity": quantity},
        )
        await self.orders.cancel_all_orders(symbol)
        flatten_order = await self.orders.market_order(
            symbol, exit_side, quantity, reduce_only=True
        )
        if flatten_order:
            self._audit(
                "emergency_flattened",
                f"Emergency flatten submitted for {symbol}",
                severity="critical",
                symbol=symbol,
                correlation_id=correlation_id,
                payload={"order": flatten_order},
            )
        else:
            self.audit_store.activate_emergency_stop(
                f"Emergency flatten failed for {symbol}"
            )
            self._audit(
                "emergency_flatten_failed",
                f"Emergency flatten failed for {symbol}; emergency stop activated",
                severity="critical",
                symbol=symbol,
                correlation_id=correlation_id,
            )
        await self._notify_trade_failed(symbol, reason)

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
