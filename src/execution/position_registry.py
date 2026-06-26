# Position registry — in-memory state for open trades, correlation IDs, exit timestamps
import time
from datetime import datetime, timezone

from loguru import logger

from src.config import settings
from src.risk.portfolio import PortfolioManager, TradeRecord


class PositionRegistry:
    def __init__(
        self,
        portfolio: PortfolioManager,
        audit_store,
        protection=None,
    ):
        self.portfolio = portfolio
        self.audit_store = audit_store
        self.protection = protection

        self.open_trades: dict[str, TradeRecord] = {}
        self.trade_correlation_ids: dict[str, str] = {}
        self.last_symbol_exit_at: dict[str, datetime] = {}
        self._last_policy_alert_at: dict[str, float] = {}

    TIMEFRAME_GROUPS: dict[str, set[str]] = {
        "scalp": {"1m", "3m"},
        "short": {"5m", "15m"},
        "medium": {"30m", "1h", "4h"},
        "long": {"1d"},
    }

    @staticmethod
    def timeframe_group(timeframe: str | None) -> str | None:
        if not timeframe:
            return None
        for group, tfs in PositionRegistry.TIMEFRAME_GROUPS.items():
            if timeframe in tfs:
                return group
        return None

    @staticmethod
    def normalize_symbol(symbol: object) -> str:
        raw = str(symbol or "")
        if ":" in raw:
            raw = raw.split(":", 1)[0]
        return raw.replace("/", "").upper()

    @staticmethod
    def position_key(symbol: str, timeframe: str | None = None, strategy: str | None = None) -> str:
        normalized_symbol = PositionRegistry.normalize_symbol(symbol)
        normalized_timeframe = str(timeframe or "").strip()
        if not normalized_timeframe:
            return normalized_symbol
        key = f"{normalized_symbol}:{normalized_timeframe}"
        if strategy:
            key = f"{key}:{strategy}"
        return key

    def has_open_trade_for_symbol(self, symbol: str) -> bool:
        return bool(self.trades_for_symbol(symbol))

    def trades_for_symbol(self, symbol: str) -> list[tuple[str, TradeRecord]]:
        normalized_symbol = self.normalize_symbol(symbol)
        return [
            (key, trade)
            for key, trade in self.open_trades.items()
            if self.normalize_symbol(trade.symbol) == normalized_symbol
        ]

    def open_notional(self, symbol: str | None = None) -> float:
        total = 0.0
        for trade in self.open_trades.values():
            if symbol and trade.symbol != symbol:
                continue
            total += trade.entry_price * trade.quantity
        return total

    def reentry_cooldown_reason(
        self,
        symbol: str,
        strategy: str | None = None,
    ) -> str:
        cooldown_seconds = (
            settings.scalp_reentry_cooldown_seconds
            if strategy == "scalp"
            else settings.reentry_cooldown_seconds
        )
        if cooldown_seconds <= 0:
            return ""
        exited_at = self.last_symbol_exit_at.get(self.normalize_symbol(symbol))
        if exited_at is None:
            return ""
        if exited_at.tzinfo is None:
            exited_at = exited_at.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - exited_at).total_seconds()
        remaining = cooldown_seconds - elapsed
        if remaining <= 0:
            return ""
        return f"re-entry cooldown active: {remaining:.0f}s remaining"

    def policy_alert_due(self, reason: str) -> bool:
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

    def record_exit_time(self, symbol: str) -> None:
        self.last_symbol_exit_at[self.normalize_symbol(symbol)] = datetime.now(
            timezone.utc
        )

    def restore_open_trades_from_audit(self) -> int:
        restored = 0
        for row in self.audit_store.load_open_trades(self.protection.mode):
            symbol = self.normalize_symbol(row["symbol"])
            timeframe = row.get("timeframe")
            correlation_id = str(row["correlation_id"])
            trade = TradeRecord(
                symbol=symbol,
                side=str(row["side"]),
                entry_price=float(row["entry_price"]),
                quantity=float(row["quantity"]),
                timestamp=datetime.fromisoformat(str(row["opened_at"])),
                timeframe=str(timeframe) if timeframe else None,
                strategy=str(row["strategy"]) if row.get("strategy") else None,
                entry_fee=float(row.get("entry_fee") or 0.0),
            )
            position_key = self.position_key(symbol, trade.timeframe, trade.strategy)
            if position_key in self.open_trades:
                reason = f"duplicate audited trade leg detected: {position_key}"
                self.audit_store.activate_emergency_stop(reason)
                self.audit_store.safe_record_event(
                    "duplicate_trade_leg",
                    reason,
                    severity="critical",
                    symbol=symbol,
                    correlation_id=correlation_id,
                    mode=self.protection.mode,
                )
                continue
            self.open_trades[position_key] = trade
            self.trade_correlation_ids[position_key] = correlation_id
            stop_order_id = row.get("stop_order_id")
            take_profit_order_id = row.get("take_profit_order_id")
            if stop_order_id:
                self.protection.active_stops[position_key] = str(stop_order_id)
            if take_profit_order_id:
                self.protection.active_tps[position_key] = str(take_profit_order_id)
            self.portfolio.add_trade(trade)
            restored += 1

        if restored:
            logger.info(f"Restored {restored} open trade(s) from audit store")
            self.audit_store.safe_record_event(
                "open_trades_restored",
                f"Restored {restored} open trade(s) from audit store",
                payload={"count": restored},
                mode=self.protection.mode,
            )
        return restored

    def restore_recent_exit_cooldowns(self) -> int:
        restored = 0
        for row in self.audit_store.load_closed_trades():
            symbol = self.normalize_symbol(row.get("symbol"))
            closed_at = row.get("closed_at")
            if not symbol or not closed_at or symbol in self.last_symbol_exit_at:
                continue
            parsed = datetime.fromisoformat(str(closed_at))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            self.last_symbol_exit_at[symbol] = parsed
            restored += 1
        return restored
