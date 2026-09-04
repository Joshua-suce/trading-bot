# Exposure limiter — enforces per-symbol and total position size limits
from collections.abc import Callable

from src.config import settings
from src.execution.position_registry import PositionRegistry
from src.risk.portfolio import PortfolioManager, TradeRecord


class ExposureLimiter:
    def __init__(
        self, portfolio: PortfolioManager, open_trades: dict[str, TradeRecord]
    ):
        self.portfolio = portfolio
        self.open_trades = open_trades

    def check_exposure_limits(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        position_key: str,
        trades_for_symbol: Callable[[str], list[tuple[str, TradeRecord]]],
        open_notional: Callable[..., float],
        *,
        timeframe: str | None = None,
    ) -> tuple[bool, str]:
        if position_key in self.open_trades:
            return False, f"trade leg already open: {position_key}"

        open_count = len(self.open_trades)
        if open_count >= settings.max_open_positions:
            return False, f"max open positions reached: {open_count}"

        symbol_trades = trades_for_symbol(symbol)
        existing_sides = {trade.side for _, trade in symbol_trades}
        if existing_sides and existing_sides != {side}:
            return (
                False,
                f"opposite-side entry blocked for {symbol}: "
                f"open side={next(iter(existing_sides))}, requested={side}",
            )

        account = self.portfolio.account
        if account is None or account.total_equity <= 0:
            return False, "account equity unavailable for exposure check"

        allowed, reason = self._check_notional_limits(
            symbol,
            side,
            entry_price * quantity,
            open_notional,
            account.total_equity,
        )
        if not allowed:
            return False, reason

        return self._check_leg_count_limits(symbol, symbol_trades, timeframe)

    def _check_notional_limits(
        self,
        symbol: str,
        side: str,
        proposed_notional: float,
        open_notional: Callable[..., float],
        total_equity: float,
    ) -> tuple[bool, str]:
        total_notional = open_notional() + proposed_notional
        max_total = total_equity * settings.max_total_open_notional_pct
        if total_notional > max_total:
            return (
                False,
                "total exposure limit exceeded: "
                f"{total_notional:.2f} > {max_total:.2f}",
            )

        symbol_notional = open_notional(symbol=symbol) + proposed_notional
        max_symbol = total_equity * settings.max_symbol_open_notional_pct
        if symbol_notional > max_symbol:
            return (
                False,
                "symbol exposure limit exceeded: "
                f"{symbol_notional:.2f} > {max_symbol:.2f}",
            )

        correlated_symbols = settings.correlated_symbols_set
        if symbol in correlated_symbols:
            correlated_notional = proposed_notional + sum(
                trade.entry_price * trade.quantity
                for trade in self.open_trades.values()
                if trade.symbol in correlated_symbols and trade.side == side
            )
            max_correlated = total_equity * settings.max_correlated_open_notional_pct
            if correlated_notional > max_correlated:
                return (
                    False,
                    "correlated exposure limit exceeded: "
                    f"{correlated_notional:.2f} > {max_correlated:.2f}",
                )
        return True, "ok"

    @staticmethod
    def _check_leg_count_limits(
        symbol: str,
        symbol_trades: list[tuple[str, TradeRecord]],
        timeframe: str | None,
    ) -> tuple[bool, str]:
        # Per-timeframe-group cap. Narrower than the per-symbol cap below, and
        # only applicable when the caller supplies a timeframe.
        group = PositionRegistry.timeframe_group(timeframe) if timeframe else None
        if group:
            group_trades = [
                (k, t)
                for k, t in symbol_trades
                if t.timeframe
                and PositionRegistry.timeframe_group(t.timeframe) == group
            ]
            if len(group_trades) >= settings.max_positions_per_symbol:
                return (
                    False,
                    f"max {group} legs for {symbol}: {len(group_trades)}",
                )
        # Per-symbol cap across all timeframe groups. This must be evaluated
        # even when a group cap was checked above, otherwise legs spread across
        # groups can exceed max_positions_per_symbol for the symbol as a whole.
        if len(symbol_trades) >= settings.max_positions_per_symbol:
            return (
                False,
                f"max trade legs for {symbol} reached: {len(symbol_trades)}",
            )
        return True, "ok"
