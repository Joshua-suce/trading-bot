import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings
from src.risk.portfolio import TradeRecord
from src.security import redact_mapping, redact_text


class AuditStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = Path(db_path or settings.audit_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    symbol TEXT,
                    mode TEXT,
                    correlation_id TEXT,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    correlation_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    opened_at TEXT NOT NULL,
                    exit_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    closed_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS controls (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    reason TEXT,
                    updated_at TEXT NOT NULL
                )
                """)
            self._ensure_column(conn, "trades", "stop_order_id", "TEXT")
            self._ensure_column(conn, "trades", "take_profit_order_id", "TEXT")
            self._ensure_column(conn, "trades", "timeframe", "TEXT")

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table: str, column: str, column_type: str
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        symbol: str | None = None,
        mode: str | None = None,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    id, ts_utc, event_type, severity, symbol, mode,
                    correlation_id, message, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    self._now(),
                    event_type,
                    severity,
                    symbol,
                    mode,
                    correlation_id,
                    redact_text(message),
                    json.dumps(
                        redact_mapping(payload or {}), sort_keys=True, default=str
                    ),
                ),
            )
        return event_id

    def record_open_trade(
        self,
        trade: TradeRecord,
        *,
        mode: str,
        correlation_id: str,
        stop_loss: float | None,
        take_profit: float | None,
        stop_order_id: str | None = None,
        take_profit_order_id: str | None = None,
    ) -> None:
        now = self._now()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    correlation_id, symbol, side, mode, status, entry_price,
                    quantity, stop_loss, take_profit, opened_at, updated_at,
                    stop_order_id, take_profit_order_id, timeframe
                )
                VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(correlation_id) DO UPDATE SET
                    status='open',
                    entry_price=excluded.entry_price,
                    quantity=excluded.quantity,
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    stop_order_id=excluded.stop_order_id,
                    take_profit_order_id=excluded.take_profit_order_id,
                    timeframe=excluded.timeframe,
                    updated_at=excluded.updated_at
                """,
                (
                    correlation_id,
                    trade.symbol,
                    trade.side,
                    mode,
                    trade.entry_price,
                    trade.quantity,
                    stop_loss,
                    take_profit,
                    trade.timestamp.isoformat(),
                    now,
                    stop_order_id,
                    take_profit_order_id,
                    trade.timeframe,
                ),
            )
        self.record_event(
            "trade_opened",
            f"Trade opened: {trade.symbol} {trade.side}",
            symbol=trade.symbol,
            mode=mode,
            correlation_id=correlation_id,
            payload={
                "entry_price": trade.entry_price,
                "quantity": trade.quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "stop_order_id": stop_order_id,
                "take_profit_order_id": take_profit_order_id,
                "timeframe": trade.timeframe,
            },
        )

    def record_closed_trade(
        self, trade: TradeRecord, *, mode: str, correlation_id: str
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE trades
                SET status='closed',
                    exit_price=?,
                    pnl=?,
                    pnl_pct=?,
                    exit_reason=?,
                    closed_at=?,
                    updated_at=?
                WHERE correlation_id=?
                """,
                (
                    trade.exit_price,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.exit_reason,
                    self._now(),
                    self._now(),
                    correlation_id,
                ),
            )
        self.record_event(
            "trade_closed",
            f"Trade closed: {trade.symbol} {trade.side}",
            symbol=trade.symbol,
            mode=mode,
            correlation_id=correlation_id,
            payload={
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "exit_reason": trade.exit_reason,
            },
        )

    def set_control(self, key: str, value: str, reason: str = "") -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO controls (key, value, reason, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    reason=excluded.reason,
                    updated_at=excluded.updated_at
                """,
                (key, value, reason, self._now()),
            )

    def get_control(self, key: str, default: str = "") -> str:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM controls WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else default

    def get_controls(self) -> dict[str, dict[str, str]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT key, value, reason, updated_at FROM controls ORDER BY key"
            ).fetchall()
        return {
            str(row["key"]): {
                "value": str(row["value"]),
                "reason": str(row["reason"] or ""),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        }

    def load_open_trades(self, mode: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM trades WHERE status = 'open'"
        params: tuple[str, ...] = ()
        if mode:
            query += " AND mode = ?"
            params = (mode,)
        query += " ORDER BY opened_at"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def load_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM trades
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, ts_utc, event_type, severity, symbol, mode,
                       correlation_id, message, payload_json
                FROM audit_events
                ORDER BY ts_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            try:
                event["payload"] = json.loads(str(event.pop("payload_json") or "{}"))
            except json.JSONDecodeError:
                event["payload"] = {}
            events.append(event)
        return events

    def mark_trade_status(
        self,
        correlation_id: str,
        status: str,
        *,
        reason: str = "",
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE trades
                SET status=?, exit_reason=?, updated_at=?
                WHERE correlation_id=?
                """,
                (status, reason, self._now(), correlation_id),
            )
        self.record_event(
            "trade_status_changed",
            f"Trade {correlation_id} marked {status}",
            severity="warning" if status != "closed" else "info",
            correlation_id=correlation_id,
            payload={"status": status, "reason": reason},
        )

    def pause_trading(self, reason: str = "") -> None:
        self.set_control("manual_pause", "true", reason)

    def resume_trading(self, reason: str = "") -> None:
        self.set_control("manual_pause", "false", reason)

    def activate_emergency_stop(self, reason: str = "") -> None:
        self.set_control("emergency_stop", "true", reason)

    def clear_emergency_stop(self, reason: str = "") -> None:
        self.set_control("emergency_stop", "false", reason)

    def trading_allowed(self) -> tuple[bool, str]:
        if not settings.trading_enabled:
            return False, "TRADING_ENABLED=false"
        if self.get_control("manual_pause", "false").lower() == "true":
            return False, "manual trading pause is active"
        if self.get_control("emergency_stop", "false").lower() == "true":
            return False, "emergency stop is active"
        return True, "ok"

    def safe_record_event(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.record_event(*args, **kwargs)
        except Exception as exc:
            logger.error(f"Audit write failed: {exc}")
