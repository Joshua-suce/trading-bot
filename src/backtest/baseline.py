from collections import defaultdict
from dataclasses import asdict, dataclass
from math import inf

from src.backtest.types import BacktestTrade


@dataclass(frozen=True)
class BaselineAcceptanceCriteria:
    min_trades: int = 30
    min_profit_factor: float = 1.10
    min_expectancy: float = 0.0


@dataclass(frozen=True)
class ScopeBaseline:
    symbol: str
    timeframe: str
    strategy: str
    trades: int
    wins: int
    win_rate_pct: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    total_slippage_cost: float
    total_cost_bps: float
    profit_factor: float
    expectancy: float
    evidence_status: str
    evidence_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_scope_baselines(
    trades: list[BacktestTrade],
    *,
    criteria: BaselineAcceptanceCriteria | None = None,
) -> list[ScopeBaseline]:
    policy = criteria or BaselineAcceptanceCriteria()
    groups: dict[tuple[str, str, str], list[BacktestTrade]] = defaultdict(list)
    for trade in trades:
        groups[(trade.symbol.upper(), trade.timeframe, trade.strategy)].append(trade)

    baselines = [
        _scope_baseline(key, scoped_trades, policy)
        for key, scoped_trades in groups.items()
    ]
    return sorted(
        baselines,
        key=lambda row: (row.symbol, row.timeframe, row.strategy),
    )


def _scope_baseline(
    key: tuple[str, str, str],
    trades: list[BacktestTrade],
    criteria: BaselineAcceptanceCriteria,
) -> ScopeBaseline:
    symbol, timeframe, strategy = key
    net_pnls = [trade.pnl for trade in trades]
    gross_profit = sum(pnl for pnl in net_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in net_pnls if pnl < 0))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (inf if gross_profit > 0 else 0.0)
    )
    count = len(trades)
    wins = sum(pnl > 0 for pnl in net_pnls)
    net_pnl = sum(net_pnls)
    expectancy = net_pnl / count if count else 0.0
    total_fees = sum(trade.fees for trade in trades)
    total_slippage = sum(trade.slippage_cost for trade in trades)
    entry_notional = sum(trade.entry_price * trade.quantity for trade in trades)
    total_cost_bps = (
        (total_fees + total_slippage) / entry_notional * 10_000
        if entry_notional > 0
        else 0.0
    )
    status, reason = _evidence_status(
        count=count,
        profit_factor=profit_factor,
        expectancy=expectancy,
        criteria=criteria,
    )
    return ScopeBaseline(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        trades=count,
        wins=wins,
        win_rate_pct=round(wins / count * 100, 2) if count else 0.0,
        gross_pnl=round(sum(trade.gross_pnl for trade in trades), 8),
        net_pnl=round(net_pnl, 8),
        total_fees=round(total_fees, 8),
        total_slippage_cost=round(total_slippage, 8),
        total_cost_bps=round(total_cost_bps, 4),
        profit_factor=round(profit_factor, 4),
        expectancy=round(expectancy, 8),
        evidence_status=status,
        evidence_reason=reason,
    )


def _evidence_status(
    *,
    count: int,
    profit_factor: float,
    expectancy: float,
    criteria: BaselineAcceptanceCriteria,
) -> tuple[str, str]:
    if count < criteria.min_trades:
        return "insufficient", f"needs at least {criteria.min_trades} trades"
    if (
        profit_factor >= criteria.min_profit_factor
        and expectancy > criteria.min_expectancy
    ):
        return "promotion_candidate", "after-cost baseline clears acceptance criteria"
    return "review", "after-cost baseline does not clear acceptance criteria"
