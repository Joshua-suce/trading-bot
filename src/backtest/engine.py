# Backtesting engine — iterates historical data, executes signals, tracks trades
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd

from src.backtest.baseline import ScopeBaseline, build_scope_baselines
from src.backtest.metrics import BacktestMetrics
from src.backtest.types import BacktestTrade
from src.indicators.compute import compute_all_indicators
from src.live.data_quality import timeframe_seconds
from src.risk.stop_loss import StopLossManager
from src.strategies import StrategyRegistry


def _to_trade_datetime(value) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.floor("us").to_pydatetime()


# The complete result of a backtest run
@dataclass
class BacktestResult:
    trades: List[BacktestTrade]
    equity_curve: List[float]
    metrics: BacktestMetrics
    scope_baselines: List[ScopeBaseline]


class BacktestEngine:
    # Configurable commission (0.04%) and slippage (0.05%)
    def __init__(
        self,
        initial_capital: float = 10_000,
        commission: float = 0.0004,
        slippage: float = 0.0005,
        leverage: int = 1,
        max_position_size: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.max_position_size = max_position_size

    # Main backtest loop: compute indicators, iterate candles, simulate positions
    def run(  # noqa: C901
        self,
        df: pd.DataFrame,
        signal_fn: Callable,
        risk_per_trade: float = 0.02,
        atr_mult_sl: float = 1.0,
        atr_mult_tp: float = 2.0,
        *,
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        default_strategy: str = "unknown",
        entry_confidence: float = 0.3,
        use_strategy_policy: bool = False,
    ) -> BacktestResult:
        df = compute_all_indicators(df).copy()
        trades: List[BacktestTrade] = []
        equity_curve: List[float] = [self.initial_capital]
        equity = self.initial_capital  # realised PnL only (used for position sizing)

        in_position = False
        position_side = ""
        entry_price = 0.0
        entry_time: Optional[datetime] = None
        quantity = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        entry_fee = 0.0
        entry_slippage_cost = 0.0
        position_strategy = default_strategy
        entry_bar_index = 0
        maximum_hold_bars: int | None = None
        strategy_registry = StrategyRegistry()
        strategy_names = {policy.name for policy in strategy_registry.all()}
        stop_manager = StopLossManager()

        # Start at index 100 to allow enough data for indicators and features
        for i in range(100, len(df) - 1):
            current = df.iloc[: i + 1]
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # No position — check for entry signal
            if not in_position:
                signal = signal_fn(current)
                if (
                    signal["direction"] != 0
                    and signal["confidence"] >= entry_confidence
                ):
                    side = "long" if signal["direction"] == 1 else "short"
                    position_strategy = str(
                        signal.get("strategy") or default_strategy or "unknown"
                    )
                    atr_val = max(
                        row.get("atr", row["close"] * 0.01), row["close"] * 0.002
                    )
                    # Apply slippage to entry price (worse for aggressive direction)
                    entry_reference_price = float(row["close"])
                    entry_price = entry_reference_price * (
                        1 + self.slippage * signal["direction"]
                    )
                    policy = strategy_registry.get(position_strategy)
                    policy_active = (
                        use_strategy_policy and position_strategy in strategy_names
                    )
                    applied_risk = (
                        policy.risk_fraction if policy_active else risk_per_trade
                    )
                    risk_amount = equity * applied_risk
                    if policy_active:
                        levels = stop_manager.calculate(
                            entry_price,
                            side,
                            atr_val,
                            strategy=position_strategy,
                        )
                        stop_loss = levels.stop_loss
                        take_profit = float(levels.take_profit or entry_price)
                        sl_distance = abs(entry_price - stop_loss)
                        maximum_hold_bars = max(
                            policy.max_hold_seconds(timeframe)
                            // timeframe_seconds(timeframe),
                            1,
                        )
                    else:
                        sl_distance = atr_val * atr_mult_sl
                        maximum_hold_bars = None
                    quantity = risk_amount / sl_distance
                    max_by_margin = equity * self.leverage / entry_price
                    position_cap = (
                        policy.max_position_fraction
                        if policy_active
                        else self.max_position_size
                    )
                    max_by_risk = position_cap * equity / entry_price
                    quantity = min(quantity, max_by_margin, max_by_risk)
                    quantity = max(quantity, 0)

                    if quantity > 0:
                        in_position = True
                        entry_bar_index = i
                        position_side = side
                        entry_time = _to_trade_datetime(row.name)
                        if not policy_active:
                            if side == "long":
                                stop_loss = entry_price - sl_distance
                                take_profit = (
                                    entry_price
                                    + sl_distance * atr_mult_tp / atr_mult_sl
                                )
                            else:
                                stop_loss = entry_price + sl_distance
                                take_profit = (
                                    entry_price
                                    - sl_distance * atr_mult_tp / atr_mult_sl
                                )
                        entry_fee = entry_price * quantity * self.commission
                        entry_slippage_cost = (
                            abs(entry_price - entry_reference_price) * quantity
                        )
                        equity -= entry_fee

            # In position — check for exit (SL/TP/reversal)
            else:
                exit_reason = None
                exit_price = None
                next_close = next_row["close"]
                next_high = next_row["high"]
                next_low = next_row["low"]

                # Check stop-loss and take-profit on the next candle's high/low
                if position_side == "long":
                    if next_low <= stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = stop_loss
                    elif next_high >= take_profit:
                        exit_reason = "take_profit"
                        exit_price = take_profit
                else:
                    if next_high >= stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = stop_loss
                    elif next_low <= take_profit:
                        exit_reason = "take_profit"
                        exit_price = take_profit

                # Exit on opposite signal when no SL/TP triggered
                if (
                    exit_reason is None
                    and maximum_hold_bars is not None
                    and i + 1 - entry_bar_index >= maximum_hold_bars
                ):
                    exit_reason = "maximum_hold"
                    exit_price = next_close

                if exit_reason is None:
                    signal = signal_fn(df.iloc[: i + 2])
                    if signal["confidence"] > 0.4:
                        if (position_side == "long" and signal["direction"] == -1) or (
                            position_side == "short" and signal["direction"] == 1
                        ):
                            exit_reason = "signal_reversal"
                            exit_price = next_close

                if exit_reason:
                    if exit_price is None:
                        exit_price = next_close
                    if entry_time is None:
                        raise RuntimeError("Backtest position is missing entry time")
                    exit_reference_price = float(exit_price)
                    exit_price = exit_reference_price * (
                        1 - self.slippage
                        if position_side == "long"
                        else 1 + self.slippage
                    )
                    if position_side == "long":
                        gross_pnl = (exit_price - entry_price) * quantity
                    else:
                        gross_pnl = (entry_price - exit_price) * quantity

                    exit_fee = exit_price * quantity * self.commission
                    fees = entry_fee + exit_fee
                    pnl = gross_pnl - fees
                    notional = entry_price * quantity
                    pnl_pct = pnl / notional if notional > 0 else 0.0
                    exit_slippage_cost = (
                        abs(exit_price - exit_reference_price) * quantity
                    )
                    equity += gross_pnl - exit_fee

                    exit_time = _to_trade_datetime(next_row.name)
                    trade = BacktestTrade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        side=position_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        symbol=symbol.upper(),
                        timeframe=timeframe,
                        strategy=position_strategy,
                        gross_pnl=gross_pnl,
                        fees=fees,
                        slippage_cost=(entry_slippage_cost + exit_slippage_cost),
                    )
                    trades.append(trade)
                    in_position = False

                # Update trailing stop
                if in_position and not use_strategy_policy:
                    if position_side == "long":
                        new_sl = (
                            next_close - df.iloc[: i + 2]["atr"].iloc[-1] * atr_mult_sl
                        )
                        stop_loss = max(stop_loss, new_sl)
                    else:
                        new_sl = (
                            next_close + df.iloc[: i + 2]["atr"].iloc[-1] * atr_mult_sl
                        )
                        stop_loss = min(stop_loss, new_sl)

            # Mark-to-market NAV for the equity curve
            if in_position:
                if position_side == "long":
                    unrealised = (row["close"] - entry_price) * quantity
                else:
                    unrealised = (entry_price - row["close"]) * quantity
                nav = equity + unrealised
            else:
                nav = equity
            equity_curve.append(max(nav, 0.0))

        # Close any open position at end
        if in_position:
            last_row = df.iloc[-1]
            exit_reference_price = float(last_row["close"])
            exit_price = exit_reference_price * (
                1 - self.slippage if position_side == "long" else 1 + self.slippage
            )
            if entry_time is None:
                raise RuntimeError("Backtest position is missing entry time")
            if position_side == "long":
                gross_pnl = (exit_price - entry_price) * quantity
            else:
                gross_pnl = (entry_price - exit_price) * quantity
            exit_fee = exit_price * quantity * self.commission
            fees = entry_fee + exit_fee
            pnl = gross_pnl - fees
            notional = entry_price * quantity
            pnl_pct = pnl / notional if notional > 0 else 0.0
            exit_slippage_cost = abs(exit_price - exit_reference_price) * quantity
            equity += gross_pnl - exit_fee
            trade = BacktestTrade(
                entry_time=entry_time,
                exit_time=_to_trade_datetime(df.index[-1]),
                side=position_side,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason="end_of_data",
                symbol=symbol.upper(),
                timeframe=timeframe,
                strategy=position_strategy,
                gross_pnl=gross_pnl,
                fees=fees,
                slippage_cost=(entry_slippage_cost + exit_slippage_cost),
            )
            trades.append(trade)
            equity_curve[-1] = max(equity, 0.0)

        periods_per_year = 365.0 * 24 * 60 * 60 / timeframe_seconds(timeframe)
        metrics = BacktestMetrics.calculate(
            trades,
            equity_curve,
            self.initial_capital,
            periods_per_year=periods_per_year,
        )
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            scope_baselines=build_scope_baselines(trades),
        )
