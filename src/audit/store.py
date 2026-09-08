import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings
from src.risk.portfolio import TradeRecord
from src.security import redact_mapping, redact_text


class AuditStore:
    TRADING_LEVEL_GREEN = "0"
    TRADING_LEVEL_YELLOW = "1"
    TRADING_LEVEL_ORANGE = "2"
    TRADING_LEVEL_RED = "3"
    RECOVERY_THRESHOLD = 3

    def __init__(self, db_path: str | None = None) -> None:
        configured_path = Path(db_path or settings.audit_db_path).expanduser()
        if not configured_path.is_absolute():
            configured_path = Path(__file__).resolve().parents[2] / configured_path
        self.db_path = configured_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        self._retire_legacy_open_trades()
        self._redact_legacy_signed_urls()
        self._migrate_legacy_emergency_stop()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _thread_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @contextmanager
    def _connection(self):
        conn = self._thread_connection()
        try:
            with conn:
                yield conn
        except sqlite3.Error:
            with suppress(Exception):
                conn.close()
            self._local.conn = None
            raise

    def _init_schema(self) -> None:
        attempts = 5
        for attempt in range(1, attempts + 1):
            try:
                self._init_schema_once()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == attempts:
                    raise
                time.sleep(0.1 * attempt)

    def _init_schema_once(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_trade_requests (
                    request_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    timeframe TEXT,
                    correlation_id TEXT,
                    status TEXT NOT NULL,
                    reason TEXT,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    claimed_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_observations (
                    id TEXT PRIMARY KEY,
                    observed_at TEXT NOT NULL,
                    candle_timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    minimum_confidence REAL NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    signal_price REAL NOT NULL,
                    ta_source TEXT NOT NULL,
                    ml_strength REAL NOT NULL,
                    ml_confidence REAL NOT NULL,
                    quality_score REAL,
                    quality_reason TEXT,
                    metrics_json TEXT NOT NULL,
                    execution_status TEXT NOT NULL DEFAULT 'not_attempted',
                    outcome_status TEXT NOT NULL DEFAULT 'pending',
                    outcome_price REAL,
                    raw_return_bps REAL,
                    directional_return_bps REAL,
                    direction_correct INTEGER,
                    outcome_horizon_seconds REAL,
                    evaluated_at TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, timeframe, candle_timestamp, strategy)
                )
                """)
            self._migrate_signal_observation_strategy_uniqueness(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_attempts (
                    id TEXT PRIMARY KEY,
                    correlation_id TEXT,
                    phase TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    strategy TEXT,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expected_price REAL,
                    actual_price REAL,
                    quantity REAL NOT NULL,
                    slippage_bps REAL,
                    order_latency_ms REAL,
                    fill_resolution_latency_ms REAL,
                    protection_latency_ms REAL,
                    fill_source TEXT,
                    order_id TEXT,
                    recovered_order INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """)
            self._ensure_column(
                conn,
                "manual_trade_requests",
                "options_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_column(conn, "trades", "stop_order_id", "TEXT")
            self._ensure_column(conn, "trades", "take_profit_order_id", "TEXT")
            self._ensure_column(conn, "trades", "timeframe", "TEXT")
            self._ensure_column(conn, "trades", "strategy", "TEXT")
            self._ensure_column(conn, "trades", "entry_fee", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "trades", "exit_fee", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "trades", "gross_pnl", "REAL")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status_updated
                ON trades(status, updated_at DESC)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_closed
                ON trades(symbol, closed_at DESC)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
                ON audit_events(ts_utc DESC)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_manual_trade_requests_status_created
                ON manual_trade_requests(status, created_at)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_observations_scope_time
                ON signal_observations(
                    symbol, timeframe, strategy, candle_timestamp DESC
                )
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_observations_outcome
                ON signal_observations(outcome_status, strategy, observed_at DESC)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_attempts_scope_time
                ON execution_attempts(symbol, timeframe, started_at DESC)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_attempts_status_time
                ON execution_attempts(status, started_at DESC)
                """)

    def _migrate_signal_observation_strategy_uniqueness(self, conn: sqlite3.Connection):
        legacy_unique = False
        for index in conn.execute("PRAGMA index_list(signal_observations)").fetchall():
            if not bool(index[2]):
                continue
            columns = [
                row[2]
                for row in conn.execute(
                    f"PRAGMA index_info({self._quote_identifier(index[1])})"
                ).fetchall()
            ]
            if columns == ["symbol", "timeframe", "candle_timestamp"]:
                legacy_unique = True
                break
        if not legacy_unique:
            return

        conn.execute(
            "ALTER TABLE signal_observations RENAME TO signal_observations_old"
        )
        conn.execute("""
            CREATE TABLE signal_observations (
                id TEXT PRIMARY KEY,
                observed_at TEXT NOT NULL,
                candle_timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                minimum_confidence REAL NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                signal_price REAL NOT NULL,
                ta_source TEXT NOT NULL,
                ml_strength REAL NOT NULL,
                ml_confidence REAL NOT NULL,
                quality_score REAL,
                quality_reason TEXT,
                metrics_json TEXT NOT NULL,
                execution_status TEXT NOT NULL DEFAULT 'not_attempted',
                outcome_status TEXT NOT NULL DEFAULT 'pending',
                outcome_price REAL,
                raw_return_bps REAL,
                directional_return_bps REAL,
                direction_correct INTEGER,
                outcome_horizon_seconds REAL,
                evaluated_at TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(symbol, timeframe, candle_timestamp, strategy)
            )
            """)
        conn.execute("""
            INSERT OR IGNORE INTO signal_observations (
                id, observed_at, candle_timestamp, symbol, timeframe, strategy,
                direction, confidence, minimum_confidence, decision, reason,
                signal_price, ta_source, ml_strength, ml_confidence,
                quality_score, quality_reason, metrics_json, execution_status,
                outcome_status, outcome_price, raw_return_bps,
                directional_return_bps, direction_correct,
                outcome_horizon_seconds, evaluated_at, updated_at
            )
            SELECT
                id, observed_at, candle_timestamp, symbol, timeframe, strategy,
                direction, confidence, minimum_confidence, decision, reason,
                signal_price, ta_source, ml_strength, ml_confidence,
                quality_score, quality_reason, metrics_json, execution_status,
                outcome_status, outcome_price, raw_return_bps,
                directional_return_bps, direction_correct,
                outcome_horizon_seconds, evaluated_at, updated_at
            FROM signal_observations_old
            """)
        conn.execute("DROP TABLE signal_observations_old")

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table: str, column: str, column_type: str
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            except sqlite3.OperationalError as exc:
                # A second process may complete the same idempotent migration
                # between table inspection and ALTER TABLE.
                if "duplicate column name" not in str(exc).lower():
                    raise
                refreshed = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if column not in refreshed:
                    raise

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _retire_legacy_open_trades(self) -> None:
        legacy_modes = ("paper", "live", "backtest")
        now = self._now()
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE trades
                SET status='legacy_mode_retired',
                    exit_reason='execution mode removed',
                    updated_at=?
                WHERE status='open' AND mode IN (?, ?, ?)
                """,
                (now, *legacy_modes),
            )
            retired = cursor.rowcount
        if retired:
            self.record_event(
                "legacy_open_trades_retired",
                f"Retired {retired} open trade records from removed modes",
                severity="warning",
                payload={"count": retired, "modes": legacy_modes},
            )

    def _redact_legacy_signed_urls(self) -> None:
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT id, message, payload_json
                FROM audit_events
                WHERE message LIKE '%signature=%'
                   OR payload_json LIKE '%signature=%'
                """).fetchall()
            for row in rows:
                message = redact_text(row["message"])
                raw_payload = str(row["payload_json"] or "{}")
                try:
                    payload = json.loads(raw_payload)
                    payload_json = json.dumps(
                        redact_mapping(payload),
                        sort_keys=True,
                        default=str,
                    )
                except json.JSONDecodeError:
                    payload_json = json.dumps(
                        {"legacy_payload": redact_text(raw_payload)},
                        sort_keys=True,
                    )
                conn.execute(
                    """
                    UPDATE audit_events
                    SET message=?, payload_json=?
                    WHERE id=?
                    """,
                    (message, payload_json, row["id"]),
                )

    def _migrate_legacy_emergency_stop(self) -> None:
        emergency_value = self.get_control("emergency_stop", "false").lower()
        trading_level = self.get_control("trading_level", "")
        # Reconcile on every startup, not just when trading_level has never
        # been set: activate_emergency_stop() writes the emergency_stop flag
        # and trading_level in two separate commits, so a crash between them
        # can leave trading_level at a stale non-RED value (e.g. GREEN) even
        # though emergency_stop='true' survived. trading_allowed() also
        # checks emergency_stop directly as a belt-and-suspenders guard, but
        # this repairs trading_level itself so the state is fully consistent
        # again rather than permanently relying on that fallback.
        if emergency_value == "true" and trading_level != self.TRADING_LEVEL_RED:
            reason = self._get_control_reason("emergency_stop")
            self.set_control(
                "trading_level",
                self.TRADING_LEVEL_RED,
                reason or "migrated from legacy emergency_stop",
            )
            self.set_control(
                "emergency_stop",
                "false",
                "migrated to trading_level",
            )
            self.record_event(
                "trading_level_migrated",
                "Migrated legacy emergency_stop to trading_level=3",
                severity="info",
                payload={"legacy_reason": reason},
            )

    def _get_control_reason(self, key: str) -> str:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT reason FROM controls WHERE key = ?", (key,)
            ).fetchone()
        return str(row["reason"]) if row and row["reason"] else ""

    def set_trading_level(self, level: str, reason: str = "") -> None:
        self.set_control("trading_level", level, reason)
        self.record_event(
            "trading_level_changed",
            f"Trading level set to {level}: {reason}",
            severity="warning" if level != self.TRADING_LEVEL_GREEN else "info",
            payload={"level": level, "reason": reason},
        )

    def get_trading_level(self) -> str:
        return self.get_control("trading_level", self.TRADING_LEVEL_GREEN)

    # Per-symbol circuit-breaker state (level/failures/successes) is
    # persisted through the same controls table as the global trading
    # level, keyed by symbol, rather than kept only in memory. An in-memory
    # dict here would silently clear every symbol-specific trading block on
    # any process restart (deploy, crash, OOM-kill, supervisor restart)
    # even though the underlying reconciliation problem that caused the
    # block was never resolved or reviewed.
    @staticmethod
    def _symbol_control_key(prefix: str, symbol: str) -> str:
        return f"{prefix}:{symbol}"

    def _get_symbol_int(self, prefix: str, symbol: str) -> int:
        raw = self.get_control(self._symbol_control_key(prefix, symbol), "0")
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0

    def _delete_controls_with_prefix(self, prefix: str) -> None:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM controls WHERE key LIKE ? ESCAPE '\\'",
                (escaped + "%",),
            )

    def _clear_all_symbol_trading_state(self) -> None:
        for prefix in (
            "symbol_trading_level",
            "symbol_reconciliation_failures",
            "symbol_reconciliation_successes",
        ):
            self._delete_controls_with_prefix(prefix + ":")

    def get_symbol_trading_level(self, symbol: str) -> str:
        return self.get_control(
            self._symbol_control_key("symbol_trading_level", symbol),
            self.TRADING_LEVEL_GREEN,
        )

    def degrade_trading_level(self, reason: str = "", symbol: str | None = None) -> str:
        if symbol:
            return self._degrade_symbol_level(symbol, reason)
        return self._degrade_global_level(reason)

    def _degrade_global_level(self, reason: str) -> str:
        current = self.get_trading_level()
        levels = [
            self.TRADING_LEVEL_GREEN,
            self.TRADING_LEVEL_YELLOW,
            self.TRADING_LEVEL_ORANGE,
            self.TRADING_LEVEL_RED,
        ]
        if current not in levels:
            current = self.TRADING_LEVEL_GREEN

        if current == self.TRADING_LEVEL_RED:
            return current

        raw = self.get_control("reconciliation_failures", "0")
        try:
            failures = int(raw)
        except (ValueError, TypeError):
            failures = 0
        failures += 1
        self.set_control("reconciliation_failures", str(failures), reason)
        self.set_control("reconciliation_successes", "0", "reset on failure")

        if failures >= self.RECOVERY_THRESHOLD:
            idx = levels.index(current)
            if idx < len(levels) - 1:
                next_level = levels[idx + 1]
                self.set_trading_level(next_level, reason)
                self.set_control(
                    "reconciliation_failures", "0", "reset after escalation"
                )
                return next_level

        if current == self.TRADING_LEVEL_GREEN and failures == 1:
            self.set_trading_level(self.TRADING_LEVEL_YELLOW, reason)
            self.set_control("reconciliation_failures", "1", "set on first degrade")
            return self.TRADING_LEVEL_YELLOW

        return current

    def _degrade_symbol_level(self, symbol: str, reason: str) -> str:
        levels = [
            self.TRADING_LEVEL_GREEN,
            self.TRADING_LEVEL_YELLOW,
            self.TRADING_LEVEL_ORANGE,
            self.TRADING_LEVEL_RED,
        ]
        level_key = self._symbol_control_key("symbol_trading_level", symbol)
        failures_key = self._symbol_control_key(
            "symbol_reconciliation_failures", symbol
        )
        successes_key = self._symbol_control_key(
            "symbol_reconciliation_successes", symbol
        )
        current = self.get_symbol_trading_level(symbol)
        if current == self.TRADING_LEVEL_RED:
            return current

        failures = self._get_symbol_int("symbol_reconciliation_failures", symbol) + 1
        self.set_control(failures_key, str(failures), reason)
        self.set_control(successes_key, "0", "reset on failure")

        if failures >= self.RECOVERY_THRESHOLD:
            idx = levels.index(current)
            if idx < len(levels) - 1:
                next_level = levels[idx + 1]
                self.set_control(level_key, next_level, reason)
                self.set_control(failures_key, "0", "reset after escalation")
                logger.warning(
                    f"Symbol {symbol} degraded to level {next_level}: {reason}"
                )
                return next_level

        if current == self.TRADING_LEVEL_GREEN and failures == 1:
            self.set_control(level_key, self.TRADING_LEVEL_YELLOW, reason)
            self.set_control(failures_key, "1", "set on first degrade")
            logger.warning(
                f"Symbol {symbol} degraded to level "
                f"{self.TRADING_LEVEL_YELLOW}: {reason}"
            )
            return self.TRADING_LEVEL_YELLOW

        return current

    def try_recover_trading_level(
        self, reason: str = "", symbol: str | None = None
    ) -> bool:
        # The blind _clear_all_symbol_trading_state() this used to do here
        # ran on EVERY successful global reconciliation pass (the reconciler
        # calls this with no symbol whenever the global level isn't RED) -
        # instantly wiping every symbol's degrade state with no threshold
        # gate and no exemption for RED, defeating the whole point of the
        # per-symbol scoping added alongside it (a stray issue on one symbol
        # must require its own confirmed recovery, not get cleared as a
        # side effect of an unrelated global check). Per-symbol recovery
        # now goes only through _try_recover_symbol_level, which has its
        # own threshold gate and (like the global path) refuses to
        # auto-clear RED. _clear_all_symbol_trading_state() remains
        # reserved for the explicit manual clear_emergency_stop() path.
        if symbol:
            return self._try_recover_symbol_level(symbol, reason)
        return self._try_recover_global_level(reason)

    def _try_recover_global_level(self, reason: str) -> bool:
        current = self.get_trading_level()
        if current == self.TRADING_LEVEL_GREEN:
            self.set_control("reconciliation_successes", "0", "reset on recovery")
            return True
        if current == self.TRADING_LEVEL_RED:
            return False
        raw = self.get_control("reconciliation_successes", "0")
        try:
            success_count = int(raw)
        except (ValueError, TypeError):
            success_count = 0
        success_count += 1
        self.set_control(
            "reconciliation_successes",
            str(success_count),
            f"consecutive successful reconciliations: {success_count}",
        )
        if success_count >= self.RECOVERY_THRESHOLD:
            self.set_trading_level(self.TRADING_LEVEL_GREEN, reason or "auto-recovered")
            self.set_control("reconciliation_successes", "0", "reset after recovery")
            self.set_control("reconciliation_failures", "0", "reset after recovery")
            self.record_event(
                "trading_level_recovered",
                f"Auto-recovered from level {current} to green",
                severity="info",
                payload={"recovered_from": current},
            )
            return True
        return False

    def _try_recover_symbol_level(self, symbol: str, reason: str) -> bool:
        current = self.get_symbol_trading_level(symbol)
        if current == self.TRADING_LEVEL_GREEN:
            return True
        if current == self.TRADING_LEVEL_RED:
            # Mirror _try_recover_global_level: RED is a manual-clear-only
            # invariant, not something consecutive clean reconciliation
            # passes alone should lift for a symbol either.
            return False
        successes = self._get_symbol_int("symbol_reconciliation_successes", symbol) + 1
        self.set_control(
            self._symbol_control_key("symbol_reconciliation_successes", symbol),
            str(successes),
            reason,
        )
        if successes >= self.RECOVERY_THRESHOLD:
            for prefix in (
                "symbol_trading_level",
                "symbol_reconciliation_failures",
                "symbol_reconciliation_successes",
            ):
                self._delete_control(self._symbol_control_key(prefix, symbol))
            logger.info(f"Symbol {symbol} recovered from level {current}")
            return True
        return False

    def reset_reconciliation_counters(self) -> None:
        self.set_control("reconciliation_failures", "0", "reset by caller")
        self.set_control("reconciliation_successes", "0", "reset by caller")

    def get_trade_protection_levels(
        self,
        correlation_id: str,
    ) -> tuple[float | None, float | None]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT stop_loss, take_profit FROM trades "
                "WHERE correlation_id = ? AND closed_at IS NULL",
                (correlation_id,),
            ).fetchone()
        if row:
            return row["stop_loss"], row["take_profit"]
        return None, None

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
                    stop_order_id, take_profit_order_id, timeframe, strategy,
                    entry_fee
                )
                VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(correlation_id) DO UPDATE SET
                    status='open',
                    entry_price=excluded.entry_price,
                    quantity=excluded.quantity,
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    stop_order_id=excluded.stop_order_id,
                    take_profit_order_id=excluded.take_profit_order_id,
                    timeframe=excluded.timeframe,
                    strategy=excluded.strategy,
                    entry_fee=excluded.entry_fee,
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
                    trade.strategy,
                    trade.entry_fee,
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
                "strategy": trade.strategy,
                "entry_fee": trade.entry_fee,
            },
        )

    def update_open_trade_state(
        self,
        correlation_id: str,
        *,
        quantity: float,
        stop_loss: float | None,
        entry_fee: float | None = None,
        take_profit: float | None,
        stop_order_id: str | None,
        take_profit_order_id: str | None,
    ) -> None:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE trades
                SET quantity = ?, entry_fee = COALESCE(?, entry_fee),
                    stop_loss = ?, take_profit = ?,
                    stop_order_id = ?, take_profit_order_id = ?, updated_at = ?
                WHERE correlation_id = ? AND closed_at IS NULL
                """,
                (
                    quantity,
                    entry_fee,
                    stop_loss,
                    take_profit,
                    stop_order_id,
                    take_profit_order_id,
                    self._now(),
                    correlation_id,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Open trade state not found for correlation {correlation_id}"
            )

    def record_closed_trade(
        self, trade: TradeRecord, *, mode: str, correlation_id: str
    ) -> None:
        if not correlation_id:
            raise ValueError("correlation_id is required to record a closed trade")
        closed_at = self._now()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    correlation_id, symbol, side, mode, status, entry_price,
                    quantity, opened_at, exit_price, pnl, pnl_pct, exit_reason,
                    closed_at, updated_at, timeframe, strategy, gross_pnl,
                    entry_fee, exit_fee
                )
                VALUES (
                    ?, ?, ?, ?, 'closed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                ON CONFLICT(correlation_id) DO UPDATE SET
                    status='closed',
                    exit_price=?,
                    pnl=?,
                    pnl_pct=?,
                    gross_pnl=?,
                    entry_fee=?,
                    exit_fee=?,
                    exit_reason=?,
                    closed_at=?,
                    updated_at=?
                """,
                (
                    correlation_id,
                    trade.symbol,
                    trade.side,
                    mode,
                    trade.entry_price,
                    trade.quantity,
                    trade.timestamp.isoformat(),
                    trade.exit_price,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.exit_reason,
                    closed_at,
                    closed_at,
                    trade.timeframe,
                    trade.strategy,
                    trade.gross_pnl,
                    trade.entry_fee,
                    trade.exit_fee,
                    trade.exit_price,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.gross_pnl,
                    trade.entry_fee,
                    trade.exit_fee,
                    trade.exit_reason,
                    closed_at,
                    closed_at,
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
                "gross_pnl": trade.gross_pnl,
                "entry_fee": trade.entry_fee,
                "exit_fee": trade.exit_fee,
                "exit_reason": trade.exit_reason,
                "strategy": trade.strategy,
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

    def _delete_control(self, key: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM controls WHERE key = ?", (key,))

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
        return self.load_trade_history(limit=limit)

    def load_trade_history(
        self,
        *,
        limit: int = 100,
        status: str | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("history limit must be at least 1")
        query = "SELECT * FROM trades"
        clauses: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def load_closed_trades(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM trades
                WHERE status = 'closed'
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_closed_trade_performance(
        self,
        *,
        symbol: str,
        timeframe: str,
        strategy: str,
        window_days: int,
    ) -> dict[str, float | int]:
        if window_days < 1:
            raise ValueError("performance window must be at least one day")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS trade_count,
                       COALESCE(SUM(pnl), 0.0) AS net_pnl,
                       COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0.0)
                           AS gross_profit,
                       ABS(COALESCE(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END), 0.0))
                           AS gross_loss
                FROM trades
                WHERE status = 'closed'
                  AND symbol = ?
                  AND timeframe = ?
                  AND strategy = ?
                  AND closed_at >= ?
                """,
                (symbol.upper(), timeframe, strategy, cutoff),
            ).fetchone()
        trade_count = int(row["trade_count"] or 0)
        net_pnl = float(row["net_pnl"] or 0.0)
        gross_profit = float(row["gross_profit"] or 0.0)
        gross_loss = float(row["gross_loss"] or 0.0)
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )
        return {
            "trade_count": trade_count,
            "net_pnl": net_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
        }

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

    def record_signal_observation(
        self,
        *,
        candle_timestamp: Any,
        symbol: str,
        timeframe: str,
        strategy: str,
        direction: int,
        confidence: float,
        minimum_confidence: float,
        decision: str,
        reason: str,
        signal_price: float,
        ta_source: str,
        ml_strength: float = 0.0,
        ml_confidence: float = 0.0,
        quality_score: float | None = None,
        quality_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        timestamp_text = self._timestamp_text(candle_timestamp)
        strategy_key = strategy or "unknown"
        observation_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"signal:{symbol.upper()}:{timeframe}:{strategy_key}:{timestamp_text}",
            )
        )
        now = self._now()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO signal_observations (
                    id, observed_at, candle_timestamp, symbol, timeframe,
                    strategy, direction, confidence, minimum_confidence,
                    decision, reason, signal_price, ta_source, ml_strength,
                    ml_confidence, quality_score, quality_reason, metrics_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, candle_timestamp, strategy)
                DO UPDATE SET
                    strategy=excluded.strategy,
                    direction=excluded.direction,
                    confidence=excluded.confidence,
                    minimum_confidence=excluded.minimum_confidence,
                    decision=excluded.decision,
                    reason=excluded.reason,
                    signal_price=excluded.signal_price,
                    ta_source=excluded.ta_source,
                    ml_strength=excluded.ml_strength,
                    ml_confidence=excluded.ml_confidence,
                    quality_score=excluded.quality_score,
                    quality_reason=excluded.quality_reason,
                    metrics_json=excluded.metrics_json,
                    updated_at=excluded.updated_at
                """,
                (
                    observation_id,
                    now,
                    timestamp_text,
                    symbol.upper(),
                    timeframe,
                    strategy_key,
                    int(direction),
                    float(confidence),
                    float(minimum_confidence),
                    decision,
                    redact_text(reason),
                    float(signal_price),
                    ta_source or "unknown",
                    float(ml_strength),
                    float(ml_confidence),
                    quality_score,
                    redact_text(quality_reason or ""),
                    json.dumps(
                        redact_mapping(metrics or {}), sort_keys=True, default=str
                    ),
                    now,
                ),
            )
        return observation_id

    def update_signal_execution(
        self,
        observation_id: str,
        execution_status: str,
    ) -> None:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE signal_observations
                SET execution_status=?, updated_at=?
                WHERE id=?
                """,
                (execution_status, self._now(), observation_id),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"signal observation {observation_id} was not found")

    def resolve_signal_observations(
        self,
        *,
        symbol: str,
        timeframe: str,
        candle_timestamp: Any,
        outcome_price: float,
        minimum_move_bps: float = 5.0,
    ) -> int:
        current_timestamp = self._timestamp_text(candle_timestamp)
        current_time = self._parse_timestamp(current_timestamp)
        now = self._now()
        resolved = 0
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, candle_timestamp, direction, signal_price
                FROM signal_observations
                WHERE symbol=? AND timeframe=? AND outcome_status='pending'
                ORDER BY candle_timestamp
                """,
                (symbol.upper(), timeframe),
            ).fetchall()
            for row in rows:
                observed_time = self._parse_timestamp(str(row["candle_timestamp"]))
                if observed_time >= current_time:
                    continue
                signal_price = float(row["signal_price"])
                if signal_price <= 0:
                    continue
                raw_return_bps = (float(outcome_price) / signal_price - 1.0) * 10_000
                direction = int(row["direction"])
                directional_return_bps = (
                    raw_return_bps * direction if direction else None
                )
                direction_correct = None
                if directional_return_bps is not None:
                    if directional_return_bps >= minimum_move_bps:
                        direction_correct = 1
                    elif directional_return_bps <= -minimum_move_bps:
                        direction_correct = 0
                conn.execute(
                    """
                    UPDATE signal_observations
                    SET outcome_status='resolved',
                        outcome_price=?,
                        raw_return_bps=?,
                        directional_return_bps=?,
                        direction_correct=?,
                        outcome_horizon_seconds=?,
                        evaluated_at=?,
                        updated_at=?
                    WHERE id=? AND outcome_status='pending'
                    """,
                    (
                        float(outcome_price),
                        raw_return_bps,
                        directional_return_bps,
                        direction_correct,
                        (current_time - observed_time).total_seconds(),
                        now,
                        now,
                        row["id"],
                    ),
                )
                resolved += 1
        return resolved

    def load_signal_observations(
        self,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("signal observation limit must be at least 1")
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM signal_observations
                ORDER BY candle_timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        observations = []
        for row in rows:
            observation = dict(row)
            observation["metrics"] = self._decode_json(
                observation.pop("metrics_json", "{}")
            )
            observations.append(observation)
        return observations

    def record_execution_attempt(
        self,
        *,
        execution_id: str,
        phase: str,
        symbol: str,
        side: str,
        status: str,
        quantity: float,
        correlation_id: str | None = None,
        timeframe: str | None = None,
        strategy: str | None = None,
        expected_price: float | None = None,
        actual_price: float | None = None,
        slippage_bps: float | None = None,
        order_latency_ms: float | None = None,
        fill_resolution_latency_ms: float | None = None,
        protection_latency_ms: float | None = None,
        fill_source: str | None = None,
        order_id: str | None = None,
        recovered_order: bool = False,
        reason: str = "",
        started_at: str | None = None,
        completed: bool = False,
    ) -> None:
        now = self._now()
        started = started_at or now
        completed_at = now if completed else None
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO execution_attempts (
                    id, correlation_id, phase, symbol, timeframe, strategy,
                    side, status, expected_price, actual_price, quantity,
                    slippage_bps, order_latency_ms, fill_resolution_latency_ms,
                    protection_latency_ms, fill_source, order_id,
                    recovered_order, reason, started_at, completed_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    correlation_id=excluded.correlation_id,
                    status=excluded.status,
                    actual_price=excluded.actual_price,
                    slippage_bps=excluded.slippage_bps,
                    order_latency_ms=excluded.order_latency_ms,
                    fill_resolution_latency_ms=excluded.fill_resolution_latency_ms,
                    protection_latency_ms=excluded.protection_latency_ms,
                    fill_source=excluded.fill_source,
                    order_id=excluded.order_id,
                    recovered_order=excluded.recovered_order,
                    reason=excluded.reason,
                    completed_at=excluded.completed_at,
                    updated_at=excluded.updated_at
                """,
                (
                    execution_id,
                    correlation_id,
                    phase,
                    symbol.upper(),
                    timeframe,
                    strategy,
                    side,
                    status,
                    expected_price,
                    actual_price,
                    float(quantity),
                    slippage_bps,
                    order_latency_ms,
                    fill_resolution_latency_ms,
                    protection_latency_ms,
                    fill_source,
                    order_id,
                    int(recovered_order),
                    redact_text(reason),
                    started,
                    completed_at,
                    now,
                ),
            )

    def load_execution_attempts(self, limit: int = 5000) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("execution attempt limit must be at least 1")
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM execution_attempts
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _timestamp_text(value: Any) -> str:
        if hasattr(value, "isoformat"):
            return str(value.isoformat())
        text = str(value)
        if not text:
            raise ValueError("signal candle timestamp is required")
        return text

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def create_manual_trade_request(
        self,
        *,
        action: str,
        symbol: str,
        reason: str,
        side: str | None = None,
        timeframe: str | None = None,
        correlation_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        normalized_action = action.strip().lower()
        allowed_actions = {"open", "close", "close-symbol", "close-all"}
        if normalized_action not in allowed_actions:
            raise ValueError(
                "manual trade action must be open, close, close-symbol, or close-all"
            )
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            raise ValueError("manual trade symbol is required")
        normalized_side = str(side or "").strip().lower() or None
        normalized_timeframe = str(timeframe or "").strip() or None
        normalized_correlation_id = str(correlation_id or "").strip() or None
        if normalized_action == "open":
            if normalized_side not in {"long", "short"}:
                raise ValueError("manual open side must be long or short")
            if not normalized_timeframe:
                raise ValueError("manual open timeframe is required")
        elif normalized_action == "close" and not normalized_correlation_id:
            raise ValueError("manual close correlation_id is required")
        clean_options = redact_mapping(options or {})
        if normalized_action == "open":
            clean_options.pop("leverage", None)
            clean_options["require_quality"] = bool(
                clean_options.get("require_quality", True)
            )

        request_id = str(uuid.uuid4())
        now = self._now()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            duplicate = conn.execute(
                """
                SELECT request_id
                FROM manual_trade_requests
                WHERE action=? AND symbol=?
                  AND COALESCE(timeframe, '')=COALESCE(?, '')
                  AND COALESCE(correlation_id, '')=COALESCE(?, '')
                  AND status IN ('pending', 'processing')
                LIMIT 1
                """,
                (
                    normalized_action,
                    normalized_symbol,
                    normalized_timeframe,
                    normalized_correlation_id,
                ),
            ).fetchone()
            if duplicate:
                raise ValueError(
                    f"matching manual request is already active: "
                    f"{duplicate['request_id']}"
                )
            conn.execute(
                """
                INSERT INTO manual_trade_requests (
                    request_id, action, symbol, side, timeframe,
                    correlation_id, status, reason, options_json, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, '{}', ?, ?)
                """,
                (
                    request_id,
                    normalized_action,
                    normalized_symbol,
                    normalized_side,
                    normalized_timeframe,
                    normalized_correlation_id,
                    redact_text(reason),
                    json.dumps(clean_options, sort_keys=True),
                    now,
                    now,
                ),
            )
        self.safe_record_event(
            "manual_trade_requested",
            f"Dashboard requested manual {normalized_action}",
            severity="warning",
            symbol=normalized_symbol,
            mode="dashboard",
            correlation_id=normalized_correlation_id,
            payload={
                "request_id": request_id,
                "action": normalized_action,
                "side": normalized_side,
                "timeframe": normalized_timeframe,
                "reason": redact_text(reason),
                "options": clean_options,
            },
        )
        return request_id

    def claim_next_manual_trade_request(self) -> dict[str, Any] | None:
        now = self._now()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""
                SELECT *
                FROM manual_trade_requests
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT 1
                """).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """
                UPDATE manual_trade_requests
                SET status='processing', claimed_at=?, updated_at=?
                WHERE request_id=? AND status='pending'
                """,
                (now, now, row["request_id"]),
            )
            if cursor.rowcount != 1:
                return None
        request = dict(row)
        request["status"] = "processing"
        request["claimed_at"] = now
        request["updated_at"] = now
        request["options"] = self._decode_json(request.pop("options_json", "{}"))
        return request

    def cancel_manual_trade_request(
        self,
        request_id: str,
        reason: str = "",
    ) -> bool:
        now = self._now()
        clean_reason = redact_text(reason or "cancelled from dashboard")
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE manual_trade_requests
                SET status='cancelled', reason=?, completed_at=?, updated_at=?
                WHERE request_id=? AND status='pending'
                """,
                (clean_reason, now, now, request_id),
            )
        if cursor.rowcount:
            self.safe_record_event(
                "manual_trade_cancelled",
                "Pending dashboard trade request cancelled",
                severity="warning",
                mode="dashboard",
                payload={"request_id": request_id, "reason": clean_reason},
            )
        return cursor.rowcount == 1

    def fail_interrupted_manual_trade_requests(self) -> int:
        now = self._now()
        result = {
            "reason": (
                "trading process restarted while request was processing; "
                "inspect exchange and audited positions before retrying"
            )
        }
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE manual_trade_requests
                SET status='failed', result_json=?, completed_at=?, updated_at=?
                WHERE status='processing'
                """,
                (json.dumps(result, sort_keys=True), now, now),
            )
        if cursor.rowcount:
            self.safe_record_event(
                "manual_trade_interrupted",
                "Interrupted manual trade requests require operator review",
                severity="critical",
                mode="trade",
                payload={"count": cursor.rowcount},
            )
        return cursor.rowcount

    def finish_manual_trade_request(
        self,
        request_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        normalized_status = status.strip().lower()
        if normalized_status not in {"completed", "failed", "cancelled"}:
            raise ValueError("manual trade terminal status is invalid")
        now = self._now()
        clean_result = redact_mapping(result or {})
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE manual_trade_requests
                SET status=?, result_json=?, completed_at=?, updated_at=?
                WHERE request_id=? AND status='processing'
                """,
                (
                    normalized_status,
                    json.dumps(clean_result, sort_keys=True),
                    now,
                    now,
                    request_id,
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"manual trade request {request_id} is not processing")

    def load_manual_trade_requests(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("manual trade request limit must be at least 1")
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM manual_trade_requests
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        requests = []
        for row in rows:
            request = dict(row)
            request["result"] = self._decode_json(request.pop("result_json", "{}"))
            request["options"] = self._decode_json(request.pop("options_json", "{}"))
            requests.append(request)
        return requests

    @staticmethod
    def _decode_json(value: object) -> dict[str, Any]:
        try:
            decoded = json.loads(str(value or "{}"))
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def count_events_since(self, *, severity: str, since_utc: str) -> int:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS event_count
                FROM audit_events
                WHERE severity = ? AND ts_utc >= ?
                """,
                (severity, since_utc),
            ).fetchone()
        return int(row["event_count"]) if row else 0

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
        current = self.get_trading_level()
        if current != self.TRADING_LEVEL_RED:
            self.set_trading_level(
                self.TRADING_LEVEL_RED,
                reason or "emergency stop activated",
            )

    def clear_emergency_stop(self, reason: str = "") -> None:
        self.set_control("emergency_stop", "false", reason)
        self.set_trading_level(
            self.TRADING_LEVEL_GREEN,
            reason or "emergency stop cleared",
        )
        self.reset_reconciliation_counters()
        self._clear_all_symbol_trading_state()

    def trading_allowed(self, symbol: str | None = None) -> tuple[bool, str]:
        if not settings.trading_enabled:
            return False, "trading disabled in settings"
        if self.get_control("manual_pause", "false") == "true":
            return False, "manual pause is active"
        level = self.get_trading_level()
        if (
            level != self.TRADING_LEVEL_RED
            and self.get_control("emergency_stop", "false") == "true"
        ):
            # activate_emergency_stop() writes this flag and trading_level
            # separately (two independent commits), so a crash/kill between
            # them can leave trading_level stale (e.g. still GREEN) while
            # this flag is the only surviving record that a halt was
            # triggered. This is a fallback for exactly that mismatch -
            # when trading_level is already RED the check below already
            # blocks trading with the normal message.
            reason = self._get_control_reason("emergency_stop")
            return False, f"emergency stop active: {reason or 'manual intervention required'}"
        if level == self.TRADING_LEVEL_YELLOW:
            reason = self._get_control_reason("trading_level") or level
            return False, f"trading degraded (level 1): {reason}"
        if level == self.TRADING_LEVEL_ORANGE:
            reason = self._get_control_reason("trading_level")
            detail = reason or "persistent reconciliation failures"
            return False, f"trading partially halted (level 2): {detail}"
        if level == self.TRADING_LEVEL_RED:
            reason = self._get_control_reason("trading_level")
            detail = reason or "manual intervention required"
            return False, f"emergency stop active (level 3): {detail}"
        if level != self.TRADING_LEVEL_GREEN:
            return False, f"trading blocked by unknown level: {level}"
        if symbol:
            sym_level = self.get_symbol_trading_level(symbol)
            if sym_level != self.TRADING_LEVEL_GREEN:
                return (
                    False,
                    f"symbol {symbol} blocked (level {sym_level})",
                )
        return True, "ok"

    def safe_record_event(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.record_event(*args, **kwargs)
        except Exception as exc:
            logger.error(f"Audit write failed: {exc}")
