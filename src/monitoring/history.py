import csv
import io
import json
from pathlib import Path
from typing import Any

from src.audit import AuditStore
from src.config import settings
from src.monitoring.governance import build_scope_governance

HISTORY_COLUMNS = (
    "closed_at",
    "symbol",
    "timeframe",
    "strategy",
    "side",
    "status",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl",
    "pnl_pct",
    "exit_reason",
    "correlation_id",
)


def history_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [trade for trade in trades if trade.get("status") == "closed"]
    pnls = [float(trade.get("pnl") or 0.0) for trade in closed]
    wins = sum(pnl > 0 for pnl in pnls)
    losses = sum(pnl <= 0 for pnl in pnls)
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    by_strategy: dict[str, dict[str, Any]] = {}
    for trade in closed:
        strategy = str(trade.get("strategy") or "unknown")
        bucket = by_strategy.setdefault(
            strategy,
            {"trades": 0, "wins": 0, "net_pnl": 0.0},
        )
        pnl = float(trade.get("pnl") or 0.0)
        bucket["trades"] += 1
        bucket["wins"] += int(pnl > 0)
        bucket["net_pnl"] += pnl
    for bucket in by_strategy.values():
        bucket["win_rate_pct"] = round(
            bucket["wins"] / bucket["trades"] * 100,
            2,
        )
        bucket["net_pnl"] = round(bucket["net_pnl"], 8)

    return {
        "records": len(trades),
        "closed_trades": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins / len(closed) * 100), 2) if closed else 0.0,
        "net_pnl": round(sum(pnls), 8),
        "profit_factor": (
            round(gross_profit / gross_loss, 4)
            if gross_loss
            else (None if not gross_profit else "infinite")
        ),
        "by_strategy": by_strategy,
    }


def history_payload(
    audit_store: AuditStore,
    *,
    limit: int,
    status: str,
    symbol: str | None,
) -> dict[str, Any]:
    trades = audit_store.load_trade_history(
        limit=limit,
        status=status,
        symbol=symbol,
    )
    governance = build_scope_governance(
        audit_store.load_signal_observations(5000),
        audit_store.load_closed_trades(5000),
        window_days=settings.performance_governance_window_days,
        min_signals=settings.performance_governance_min_signals,
        min_trades=settings.performance_governance_min_trades,
        promote_min_accuracy=settings.performance_promote_min_accuracy,
        disable_max_accuracy=settings.performance_disable_max_accuracy,
        promote_min_return_bps=settings.performance_promote_min_return_bps,
        disabled_scopes=settings.disabled_strategy_scopes_set,
    )
    return {
        "audit_db_path": str(audit_store.db_path),
        "filters": {"limit": limit, "status": status, "symbol": symbol},
        "summary": history_summary(trades),
        "performance_governance": governance,
        "trades": trades,
    }


def render_history(payload: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(payload, indent=2, sort_keys=True, default=str)
    if output_format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for trade in payload["trades"]:
            writer.writerow({column: trade.get(column) for column in HISTORY_COLUMNS})
        return buffer.getvalue()
    if output_format != "table":
        raise ValueError(f"unsupported history format: {output_format}")

    summary = payload["summary"]
    lines = [
        f"Audit DB: {payload['audit_db_path']}",
        (
            f"Records: {summary['records']} | Closed: {summary['closed_trades']} | "
            f"Wins: {summary['wins']} | Losses: {summary['losses']} | "
            f"Win rate: {summary['win_rate_pct']:.2f}% | "
            f"Net PnL: {summary['net_pnl']:.8f}"
        ),
        "",
        (
            f"{'Closed (UTC)':25} {'Symbol':10} {'TF':5} {'Strategy':12} {'Side':6} "
            f"{'Status':10} {'Entry':12} {'Exit':12} {'PnL':12} {'Reason'}"
        ),
    ]
    for trade in payload["trades"]:
        closed_at = str(trade.get("closed_at") or trade.get("updated_at") or "")[:25]
        lines.append(
            f"{closed_at:25} "
            f"{str(trade.get('symbol') or ''):10} "
            f"{str(trade.get('timeframe') or '-'):5} "
            f"{str(trade.get('strategy') or 'unknown'):12} "
            f"{str(trade.get('side') or ''):6} "
            f"{str(trade.get('status') or ''):10} "
            f"{_number(trade.get('entry_price')):12} "
            f"{_number(trade.get('exit_price')):12} "
            f"{_number(trade.get('pnl')):12} "
            f"{str(trade.get('exit_reason') or '')}"
        )
    governance = payload.get("performance_governance") or []
    if governance:
        lines.extend(
            [
                "",
                "Performance governance:",
                (
                    f"{'Scope':18} {'Strategy':12} {'Recommendation':20} "
                    f"{'Signals':8} {'Accuracy':9} {'Avg bps':9} {'Trades':6}"
                ),
            ]
        )
        for row in governance:
            accuracy = row.get("direction_accuracy_pct")
            avg_return = row.get("avg_directional_return_bps")
            lines.append(
                f"{str(row.get('scope') or ''):18} "
                f"{str(row.get('strategy') or ''):12} "
                f"{str(row.get('recommendation') or ''):20} "
                f"{int(row.get('resolved_signals') or 0):8} "
                f"{_optional_number(accuracy):9} "
                f"{_optional_number(avg_return):9} "
                f"{int(row.get('trade_count') or 0):6}"
            )
    return "\n".join(lines)


def write_history_output(content: str, output_path: str) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def _number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def _optional_number(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"
