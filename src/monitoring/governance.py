import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any


def build_scope_governance(
    observations: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    *,
    window_days: int,
    min_signals: int,
    min_trades: int,
    promote_min_accuracy: float,
    disable_max_accuracy: float,
    promote_min_return_bps: float,
    disabled_scopes: set[str] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=window_days)
    disabled = disabled_scopes or set()
    signal_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    trade_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for observation in observations:
        if observation.get("decision") != "accepted":
            continue
        if observation.get("outcome_status") != "resolved":
            continue
        if not observation.get("direction"):
            continue
        if _timestamp(observation.get("candle_timestamp")) < cutoff:
            continue
        key = _scope_key(observation)
        signal_groups[key].append(observation)

    for trade in trades:
        if trade.get("status") != "closed":
            continue
        if _timestamp(trade.get("closed_at") or trade.get("updated_at")) < cutoff:
            continue
        key = _scope_key(trade)
        trade_groups[key].append(trade)

    rows = []
    for key in sorted(set(signal_groups) | set(trade_groups)):
        symbol, timeframe, strategy = key
        scope = f"{symbol}:{timeframe}"
        signals = signal_groups.get(key, [])
        closed_trades = trade_groups.get(key, [])
        scored_values = [
            value
            for signal in signals
            if (value := _finite_number(signal.get("direction_correct"))) is not None
        ]
        correct = sum(1 for value in scored_values if value >= 0.5)
        accuracy = correct / len(scored_values) if scored_values else None
        wilson_low, wilson_high = _wilson_interval(correct, len(scored_values))
        directional_returns = [
            value
            for signal in signals
            if (value := _finite_number(signal.get("directional_return_bps")))
            is not None
        ]
        avg_return = (
            sum(directional_returns) / len(directional_returns)
            if directional_returns
            else None
        )
        trade_metrics = _trade_metrics(closed_trades)
        recommendation, reason = _recommendation(
            signal_count=len(signals),
            scored_count=len(scored_values),
            accuracy=accuracy,
            wilson_low=wilson_low,
            wilson_high=wilson_high,
            avg_return_bps=avg_return,
            trade_count=trade_metrics["trade_count"],
            net_pnl=trade_metrics["net_pnl"],
            profit_factor=trade_metrics["profit_factor"],
            min_signals=min_signals,
            min_trades=min_trades,
            promote_min_accuracy=promote_min_accuracy,
            disable_max_accuracy=disable_max_accuracy,
            promote_min_return_bps=promote_min_return_bps,
        )
        rows.append(
            {
                "scope": scope,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "current_state": "disabled" if scope in disabled else "enabled",
                "recommendation": recommendation,
                "reason": reason,
                "resolved_signals": len(signals),
                "scored_signals": len(scored_values),
                "direction_accuracy_pct": (
                    round(accuracy * 100, 2) if accuracy is not None else None
                ),
                "accuracy_low_pct": (
                    round(wilson_low * 100, 2) if wilson_low is not None else None
                ),
                "accuracy_high_pct": (
                    round(wilson_high * 100, 2) if wilson_high is not None else None
                ),
                "avg_directional_return_bps": (
                    round(avg_return, 3) if avg_return is not None else None
                ),
                **trade_metrics,
                "window_days": window_days,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _recommendation_rank(str(row["recommendation"])),
            -int(row["resolved_signals"]),
            str(row["scope"]),
            str(row["strategy"]),
        ),
    )


def _recommendation(
    *,
    signal_count: int,
    scored_count: int,
    accuracy: float | None,
    wilson_low: float | None,
    wilson_high: float | None,
    avg_return_bps: float | None,
    trade_count: int,
    net_pnl: float,
    profit_factor: float | None,
    min_signals: int,
    min_trades: int,
    promote_min_accuracy: float,
    disable_max_accuracy: float,
    promote_min_return_bps: float,
) -> tuple[str, str]:
    if signal_count < min_signals or scored_count < min_signals:
        return (
            "insufficient",
            f"needs {min_signals} scored accepted signals",
        )

    poor_signals = (
        accuracy is not None
        and accuracy <= disable_max_accuracy
        and wilson_high is not None
        and wilson_high < 0.55
        and avg_return_bps is not None
        and avg_return_bps < 0
    )
    poor_trades = (
        trade_count >= min_trades
        and net_pnl < 0
        and profit_factor is not None
        and profit_factor < 0.8
    )
    if poor_signals or poor_trades:
        evidence = "signal outcomes" if poor_signals else "realized trade results"
        return "disable_candidate", f"negative {evidence} require operator review"

    positive_signals = (
        accuracy is not None
        and accuracy >= promote_min_accuracy
        and wilson_low is not None
        and wilson_low >= 0.45
        and avg_return_bps is not None
        and avg_return_bps >= promote_min_return_bps
    )
    trades_support = trade_count < min_trades or (
        net_pnl >= 0 and (profit_factor is None or profit_factor >= 1.0)
    )
    if positive_signals and trades_support:
        return "promote_candidate", "positive signal evidence; retain risk limits"

    return "watch", "evidence is mixed or not statistically decisive"


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_finite_number(trade.get("pnl")) or 0.0 for trade in trades]
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss
        else (None if not gross_profit else math.inf)
    )
    return {
        "trade_count": len(trades),
        "trade_win_rate_pct": (
            round(sum(pnl > 0 for pnl in pnls) / len(pnls) * 100, 2) if pnls else None
        ),
        "net_pnl": round(sum(pnls), 8),
        "profit_factor": (
            round(profit_factor, 4) if profit_factor is not None else None
        ),
    }


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    )
    return (center - spread) / denominator, (center + spread) / denominator


def _scope_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("symbol") or "UNKNOWN").upper(),
        str(record.get("timeframe") or "unknown"),
        str(record.get("strategy") or "unknown"),
    )


def _timestamp(value: Any) -> datetime:
    try:
        if isinstance(value, datetime):
            parsed = value
        else:
            text = str(value or "")
            if not text:
                return datetime.min.replace(tzinfo=timezone.utc)
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _recommendation_rank(recommendation: str) -> int:
    return {
        "disable_candidate": 0,
        "watch": 1,
        "promote_candidate": 2,
        "insufficient": 3,
    }.get(recommendation, 4)
