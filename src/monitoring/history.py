import csv
import io
import json
from pathlib import Path
from typing import Any

from src.audit import AuditStore

HISTORY_COLUMNS = (
    "closed_at",
    "symbol",
    "timeframe",
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
    return {
        "audit_db_path": str(audit_store.db_path),
        "filters": {"limit": limit, "status": status, "symbol": symbol},
        "summary": history_summary(trades),
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
            f"{'Closed (UTC)':25} {'Symbol':10} {'TF':5} {'Side':6} "
            f"{'Status':10} {'Entry':12} {'Exit':12} {'PnL':12} {'Reason'}"
        ),
    ]
    for trade in payload["trades"]:
        closed_at = str(trade.get("closed_at") or trade.get("updated_at") or "")[:25]
        lines.append(
            f"{closed_at:25} "
            f"{str(trade.get('symbol') or ''):10} "
            f"{str(trade.get('timeframe') or '-'):5} "
            f"{str(trade.get('side') or ''):6} "
            f"{str(trade.get('status') or ''):10} "
            f"{_number(trade.get('entry_price')):12} "
            f"{_number(trade.get('exit_price')):12} "
            f"{_number(trade.get('pnl')):12} "
            f"{str(trade.get('exit_reason') or '')}"
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
