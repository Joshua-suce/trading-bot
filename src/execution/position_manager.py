# Position manager — manages position lifecycle: entry, SL/TP placement, exit
import time
from datetime import datetime, timezone
from uuid import uuid4

from loguru import logger

from src.audit import AuditStore
from src.config import settings
from src.exchange.client import ExchangeClient
from src.execution.order_manager import OrderManager
from src.monitoring.alerter import Alerter
from src.risk.portfolio import PortfolioManager, TradeRecord
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossLevels, StopLossManager
from src.security import redact_text


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
        self.last_symbol_exit_at: dict[str, datetime] = {}
        self._last_policy_alert_at: dict[str, float] = {}

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

        if await self._entry_preflight_blocks(symbol, "long", price):
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
            filled_quantity = float(order["filled"])
            entry_price = await self._validated_entry_fill_price(
                order,
                symbol,
                price,
                "sell",
                filled_quantity,
                correlation_id,
            )
            if entry_price is None:
                return False
            levels = self.sl_manager.calculate(entry_price, "long", atr)
            trade = TradeRecord(
                symbol=symbol,
                side="long",
                entry_price=entry_price,
                quantity=filled_quantity,
                timestamp=datetime.now(),
                timeframe=timeframe,
            )

            protection = await self._place_entry_protection(
                symbol,
                "long",
                filled_quantity,
                levels,
                correlation_id,
            )
            if protection is None:
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
                await self._handle_unprotected_entry(
                    symbol,
                    "sell",
                    filled_quantity,
                    f"audit persistence failed after long entry: {exc}",
                    correlation_id,
                )
                return False

            self.open_trades[position_key] = trade
            self.trade_correlation_ids[position_key] = correlation_id
            self.active_stops[position_key] = stop_order_id
            if take_profit_order_id:
                self.active_tps[position_key] = take_profit_order_id
            self.portfolio.add_trade(trade)

            logger.info(
                f"Entered LONG {symbol} qty={filled_quantity} price={entry_price} "
                f"sl={levels.stop_loss} tp={levels.take_profit}"
            )
            await self._notify_trade_opened(
                symbol,
                "long",
                entry_price,
                filled_quantity,
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

        if await self._entry_preflight_blocks(symbol, "short", price):
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
            filled_quantity = float(order["filled"])
            entry_price = await self._validated_entry_fill_price(
                order,
                symbol,
                price,
                "buy",
                filled_quantity,
                correlation_id,
            )
            if entry_price is None:
                return False
            levels = self.sl_manager.calculate(entry_price, "short", atr)
            trade = TradeRecord(
                symbol=symbol,
                side="short",
                entry_price=entry_price,
                quantity=filled_quantity,
                timestamp=datetime.now(),
                timeframe=timeframe,
            )

            protection = await self._place_entry_protection(
                symbol,
                "short",
                filled_quantity,
                levels,
                correlation_id,
            )
            if protection is None:
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
                await self._handle_unprotected_entry(
                    symbol,
                    "buy",
                    filled_quantity,
                    f"audit persistence failed after short entry: {exc}",
                    correlation_id,
                )
                return False

            self.open_trades[position_key] = trade
            self.trade_correlation_ids[position_key] = correlation_id
            self.active_stops[position_key] = stop_order_id
            if take_profit_order_id:
                self.active_tps[position_key] = take_profit_order_id
            self.portfolio.add_trade(trade)

            logger.info(
                f"Entered SHORT {symbol} qty={filled_quantity} price={entry_price} "
                f"sl={levels.stop_loss} tp={levels.take_profit}"
            )
            await self._notify_trade_opened(
                symbol,
                "short",
                entry_price,
                filled_quantity,
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
        confirmed_order = await self._confirmed_full_fill(order, symbol, trade.quantity)
        if confirmed_order:
            exit_price = await self._resolve_exit_price(
                confirmed_order, symbol, trade.entry_price
            )
            self.portfolio.close_trade(trade, exit_price, reason)
            correlation_id = self.trade_correlation_ids.get(position_key, "")
            if correlation_id:
                self.audit_store.record_closed_trade(
                    trade, mode=self.mode, correlation_id=correlation_id
                )
            del self.open_trades[position_key]
            self.trade_correlation_ids.pop(position_key, None)
            self.last_symbol_exit_at[symbol] = datetime.now(timezone.utc)
            await self._notify_trade_completed(trade, exit_price, reason)
            if position_key in self.active_stops:
                await self.orders.cancel_order(
                    symbol,
                    self.active_stops[position_key],
                    conditional=True,
                )
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

    async def reconcile_exchange_state(self) -> bool | None:
        positions = await self._fetch_positions_for_reconciliation()
        if positions is None:
            return None

        exchange_positions, unmanaged = self._classify_exchange_positions(positions)
        if unmanaged:
            reason = f"unmanaged exchange positions detected: {unmanaged}"
            await self._fail_reconciliation(
                "unmanaged_positions",
                reason,
                payload={"positions": unmanaged},
            )
            return False

        await self._clear_missing_exchange_positions(set(exchange_positions))

        if not await self._verify_exchange_position_details(exchange_positions):
            return False

        if not await self._verify_all_protective_orders():
            return False

        self._audit("reconciliation_ok", "Exchange state reconciled")
        return True

    async def _fetch_positions_for_reconciliation(self) -> list[dict] | None:
        try:
            return await self.client.fetch_positions()
        except Exception as exc:
            reason = redact_text(f"position reconciliation unavailable: {exc}")
            logger.warning(reason)
            self._audit(
                "reconciliation_unavailable",
                reason,
                severity="warning",
            )
            if self.alerter:
                await self.alerter.error_alert(reason)
            return None

    def _classify_exchange_positions(
        self, positions: list[dict]
    ) -> tuple[dict[str, dict], list[dict]]:
        unmanaged = []
        exchange_positions = {}
        for position in positions:
            symbol = self._normalize_symbol(
                position.get("symbol") or position.get("info", {}).get("symbol")
            )
            size = self._position_size(position)
            if abs(size) <= 0 or not symbol:
                continue
            exchange_positions[symbol] = position
            if not self._has_open_trade_for_symbol(symbol):
                unmanaged.append({"symbol": symbol, "size": size})
        return exchange_positions, unmanaged

    async def _clear_missing_exchange_positions(
        self, exchange_symbols: set[str]
    ) -> None:
        missing_position_keys = [
            key
            for key, trade in self.open_trades.items()
            if trade.symbol not in exchange_symbols
        ]
        for position_key in missing_position_keys:
            await self._finalize_exchange_closed_position(position_key)

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

    async def _verify_exchange_position_details(
        self, exchange_positions: dict[str, dict]
    ) -> bool:
        for position_key, trade in self.open_trades.items():
            position = exchange_positions.get(trade.symbol)
            if position is None:
                continue
            exchange_side = self._position_side(position)
            if exchange_side and exchange_side != trade.side:
                reason = (
                    f"exchange side mismatch for {trade.symbol}: "
                    f"audit={trade.side} exchange={exchange_side}"
                )
                await self._fail_reconciliation(
                    "position_side_mismatch",
                    reason,
                    symbol=trade.symbol,
                    correlation_id=self.trade_correlation_ids.get(position_key),
                )
                return False

            exchange_quantity = abs(self._position_size(position))
            tolerance = await self._quantity_tolerance(trade.symbol)
            if abs(exchange_quantity - trade.quantity) > tolerance:
                reason = (
                    f"exchange quantity mismatch for {trade.symbol}: "
                    f"audit={trade.quantity} exchange={exchange_quantity} "
                    f"tolerance={tolerance}"
                )
                await self._fail_reconciliation(
                    "position_quantity_mismatch",
                    reason,
                    symbol=trade.symbol,
                    correlation_id=self.trade_correlation_ids.get(position_key),
                )
                return False
        return True

    async def _quantity_tolerance(self, symbol: str) -> float:
        try:
            market = await self.client.fetch_market(symbol)
            precision = market.get("precision", {}).get("amount")
            if isinstance(precision, float) and precision > 0:
                return precision / 2
            if isinstance(precision, int) and precision >= 0:
                return 10 ** (-precision) / 2
        except Exception as exc:
            logger.warning(
                "Could not resolve quantity tolerance for {}: {}",
                symbol,
                redact_text(exc),
            )
        return 1e-12

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
        logger.critical(safe_reason)
        self.audit_store.activate_emergency_stop(safe_reason)
        self._audit(
            event_type,
            safe_reason,
            severity="critical",
            symbol=symbol,
            correlation_id=correlation_id,
            payload=payload,
        )
        if self.alerter:
            await self.alerter.error_alert(safe_reason)

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

    def restore_recent_exit_cooldowns(self) -> int:
        restored = 0
        for row in self.audit_store.load_closed_trades():
            symbol = self._normalize_symbol(row.get("symbol"))
            closed_at = row.get("closed_at")
            if not symbol or not closed_at or symbol in self.last_symbol_exit_at:
                continue
            parsed = datetime.fromisoformat(str(closed_at))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            self.last_symbol_exit_at[symbol] = parsed
            restored += 1
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
    def _position_side(position: dict) -> str | None:
        side = str(position.get("side") or "").lower()
        if side in {"long", "short"}:
            return side
        info = position.get("info", {})
        position_side = str(info.get("positionSide") or "").lower()
        if position_side in {"long", "short"}:
            return position_side
        position_amount = info.get("positionAmt")
        if position_amount not in (None, ""):
            signed_amount = float(position_amount)
            if signed_amount > 0:
                return "long"
            if signed_amount < 0:
                return "short"
        size = PositionManager._position_size(position)
        if size > 0:
            return "long"
        if size < 0:
            return "short"
        return None

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
            conditional_orders = await self.client.fetch_open_orders(
                trade.symbol, conditional=True
            )
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
        conditional_order_ids = {
            str(order.get("id", "")) for order in conditional_orders
        }
        if stop_order_id not in conditional_order_ids:
            return False
        if take_profit_order_id and take_profit_order_id not in open_order_ids:
            return False
        return True

    async def _finalize_exchange_closed_position(self, position_key: str) -> None:
        trade = self.open_trades[position_key]
        correlation_id = self.trade_correlation_ids.get(position_key, "")
        exit_price, reason = await self._protective_exit_details(position_key)
        logger.info(
            f"{trade.symbol}: exchange position closed via {reason} at {exit_price}"
        )
        self.portfolio.close_trade(trade, exit_price, reason)
        if correlation_id:
            self.audit_store.record_closed_trade(
                trade, mode=self.mode, correlation_id=correlation_id
            )
        await self._notify_trade_completed(trade, exit_price, reason)

        stop_order_id = self.active_stops.pop(position_key, None)
        take_profit_order_id = self.active_tps.pop(position_key, None)
        if stop_order_id and reason != "stop_loss":
            await self.orders.cancel_order(
                trade.symbol, stop_order_id, conditional=True
            )
        if take_profit_order_id and reason != "take_profit":
            await self.orders.cancel_order(trade.symbol, take_profit_order_id)
        self.open_trades.pop(position_key, None)
        self.trade_correlation_ids.pop(position_key, None)
        self.last_symbol_exit_at[trade.symbol] = datetime.now(timezone.utc)

    async def _protective_exit_details(self, position_key: str) -> tuple[float, str]:
        trade = self.open_trades[position_key]
        candidates = (
            ("take_profit", self.active_tps.get(position_key), False),
            ("stop_loss", self.active_stops.get(position_key), True),
        )
        for reason, order_id, conditional in candidates:
            if not order_id:
                continue
            order = await self._resolved_order(
                trade.symbol,
                order_id,
                reason=reason,
                conditional=conditional,
            )
            if not self._order_is_filled(order):
                continue
            assert order is not None
            execution_order = await self._latest_exit_order(trade)
            if execution_order is not None:
                execution_price = self._positive_order_price(execution_order)
                if execution_price is not None:
                    return execution_price, reason
            price = self._positive_order_price(order)
            if price is None:
                price = await self._market_price(trade.symbol, trade.entry_price)
            return price, reason

        execution_order = await self._latest_exit_order(trade)
        if execution_order is not None:
            price = self._positive_order_price(execution_order)
            if price is not None:
                return price, "exchange_close"
        return (
            await self._market_price(trade.symbol, trade.entry_price),
            "exchange_close",
        )

    async def _resolved_order(
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
            return await self._historical_order(
                symbol, order_id, conditional=conditional
            )

    @staticmethod
    def _order_is_filled(order: dict | None) -> bool:
        if order is None:
            return False
        info = order.get("info") or {}
        statuses = {
            str(order.get("status", "")).lower(),
            str(order.get("algoStatus", "")).lower(),
            str(info.get("algoStatus", "")).lower(),
            str(info.get("status", "")).lower(),
        }
        return bool(statuses & {"closed", "filled", "finished"})

    async def _historical_order(
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

    async def _latest_exit_order(self, trade: TradeRecord) -> dict | None:
        try:
            orders = await self.client.fetch_orders(trade.symbol)
        except Exception as exc:
            logger.warning(f"Could not fetch exit history for {trade.symbol}: {exc}")
            return None
        expected_side = "sell" if trade.side == "long" else "buy"
        opened_ms = int(trade.timestamp.timestamp() * 1000)
        tolerance = await self._quantity_tolerance(trade.symbol)
        candidates = []
        for order in orders:
            if str(order.get("status", "")).lower() not in {"closed", "filled"}:
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

    async def _resolve_exit_price(
        self, order: dict, symbol: str, fallback: float
    ) -> float:
        price = self._positive_order_price(order)
        if price is not None:
            return price
        return await self._market_price(symbol, fallback)

    async def _market_price(self, symbol: str, fallback: float) -> float:
        try:
            ticker = await self.client.fetch_ticker(symbol)
            for key in ("last", "mark", "index", "bid", "ask"):
                value = ticker.get(key)
                if value is not None and float(value) > 0:
                    return float(value)
        except Exception as exc:
            logger.warning(f"Could not resolve market price for {symbol}: {exc}")
        return fallback

    async def _resolve_entry_fill_price(self, order: dict, symbol: str) -> float | None:
        price = self._positive_order_price(order)
        if price is not None:
            return price
        order_id = str(order.get("id") or "")
        if not order_id:
            return None
        resolved = await self._resolved_order(
            symbol,
            order_id,
            reason="entry",
            conditional=False,
        )
        if resolved is None:
            return None
        return self._positive_order_price(resolved)

    async def _validated_entry_fill_price(
        self,
        order: dict,
        symbol: str,
        signal_price: float,
        exit_side: str,
        filled_quantity: float,
        correlation_id: str,
    ) -> float | None:
        entry_price = await self._resolve_entry_fill_price(order, symbol)
        if entry_price is None:
            reason = "entry fill price unavailable from exchange"
        else:
            fill_slippage = self._entry_fill_slippage_bps(signal_price, entry_price)
            if fill_slippage <= settings.max_entry_slippage_bps:
                return entry_price
            reason = (
                f"entry fill slippage above limit: {fill_slippage:.2f}bps "
                f"> {settings.max_entry_slippage_bps:.2f}bps"
            )
        await self._handle_unprotected_entry(
            symbol,
            exit_side,
            filled_quantity,
            reason,
            correlation_id,
        )
        return None

    @staticmethod
    def _positive_order_price(order: dict) -> float | None:
        info = order.get("info") or {}
        for value in (
            order.get("average"),
            order.get("price"),
            info.get("avgPrice"),
            info.get("price"),
            order.get("stopPrice"),
            order.get("triggerPrice"),
            info.get("triggerPrice"),
        ):
            if value is not None and float(value) > 0:
                return float(value)
        return None

    @staticmethod
    def position_key(symbol: str, timeframe: str | None = None) -> str:
        return PositionManager._normalize_symbol(symbol)

    def _has_open_trade_for_symbol(self, symbol: str) -> bool:
        return any(trade.symbol == symbol for trade in self.open_trades.values())

    def _reentry_cooldown_reason(self, symbol: str) -> str:
        if settings.reentry_cooldown_seconds <= 0:
            return ""
        exited_at = self.last_symbol_exit_at.get(self._normalize_symbol(symbol))
        if exited_at is None:
            return ""
        if exited_at.tzinfo is None:
            exited_at = exited_at.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - exited_at).total_seconds()
        remaining = settings.reentry_cooldown_seconds - elapsed
        if remaining <= 0:
            return ""
        return f"re-entry cooldown active: {remaining:.0f}s remaining"

    async def _entry_policy_blocks(self, symbol: str) -> bool:
        can_trade, reason = self.portfolio.can_trade()
        if not can_trade:
            logger.warning(f"Cannot enter {symbol}: {reason}")
            if self._policy_alert_due(reason):
                await self._notify_trade_failed(symbol, reason)
                self._audit(
                    "trade_blocked",
                    reason,
                    severity="warning",
                    symbol=symbol,
                )
            return True

        reason = self._reentry_cooldown_reason(symbol)
        if not reason:
            return False
        logger.warning(f"Cannot enter {symbol}: {reason}")
        self._audit(
            "trade_blocked_cooldown",
            reason,
            severity="warning",
            symbol=symbol,
        )
        return True

    async def _entry_preflight_blocks(
        self, symbol: str, side: str, signal_price: float
    ) -> bool:
        if await self._entry_policy_blocks(symbol):
            return True
        price_drift_reason = await self._entry_price_drift_reason(
            symbol, side, signal_price
        )
        if not price_drift_reason:
            return False
        await self._block_stale_entry(symbol, price_drift_reason)
        return True

    def _policy_alert_due(self, reason: str) -> bool:
        category = reason.split(":", 1)[0].strip().lower()
        now = time.monotonic()
        last_alert = self._last_policy_alert_at.get(category)
        if (
            last_alert is not None
            and now - last_alert < settings.risk_block_alert_cooldown_seconds
        ):
            return False
        self._last_policy_alert_at[category] = now
        return True

    async def _entry_price_drift_reason(
        self, symbol: str, side: str, signal_price: float
    ) -> str:
        if signal_price <= 0:
            return "entry signal price is invalid"
        try:
            ticker = await self.client.fetch_ticker(symbol)
        except Exception as exc:
            return redact_text(f"entry quote unavailable: {exc}")
        quote_keys = (
            ("ask", "last", "mark")
            if side == "long"
            else (
                "bid",
                "last",
                "mark",
            )
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
        drift_bps = self._entry_fill_slippage_bps(signal_price, executable_price)
        if drift_bps <= settings.max_entry_slippage_bps:
            return ""
        return (
            f"signal price drift above limit: {drift_bps:.2f}bps "
            f"> {settings.max_entry_slippage_bps:.2f}bps "
            f"(signal={signal_price:.8f}, quote={executable_price:.8f})"
        )

    async def _block_stale_entry(self, symbol: str, reason: str) -> None:
        logger.warning(f"Order blocked for {symbol}: {reason}")
        self._audit(
            "trade_blocked_price_drift",
            reason,
            severity="warning",
            symbol=symbol,
        )
        await self._notify_trade_failed(symbol, reason)

    @staticmethod
    def _entry_fill_slippage_bps(reference_price: float, fill_price: float) -> float:
        if reference_price <= 0 or fill_price <= 0:
            return float("inf")
        return abs(fill_price - reference_price) / reference_price * 10_000

    async def _handle_unprotected_entry(
        self,
        symbol: str,
        exit_side: str,
        quantity: float,
        reason: str,
        correlation_id: str,
    ) -> None:
        self.last_symbol_exit_at[self._normalize_symbol(symbol)] = datetime.now(
            timezone.utc
        )
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
        confirmed_order = await self._confirmed_full_fill(
            flatten_order, symbol, quantity
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

    async def _place_entry_protection(
        self,
        symbol: str,
        trade_side: str,
        quantity: float,
        levels: StopLossLevels,
        correlation_id: str,
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

        await self._handle_unprotected_entry(
            symbol,
            exit_side,
            quantity,
            f"protective order placement failed after {trade_side} entry",
            correlation_id,
        )
        return None

    async def _confirmed_full_fill(
        self, order: dict | None, symbol: str, expected_quantity: float
    ) -> dict | None:
        if not order:
            return None
        tolerance = await self._quantity_tolerance(symbol)
        if float(order.get("filled") or 0) >= expected_quantity - tolerance:
            return order

        order_id = str(order.get("id") or "")
        if not order_id:
            return None
        try:
            refreshed = await self.client.fetch_order(order_id, symbol)
        except Exception as exc:
            logger.warning(f"Could not confirm exit fill for {symbol}: {exc}")
            return None
        if float(refreshed.get("filled") or 0) >= expected_quantity - tolerance:
            return refreshed
        return None

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
