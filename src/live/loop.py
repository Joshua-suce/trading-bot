# Unified trading loop for Binance Demo Trading and mainnet.
import asyncio
import html
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from src.audit import AuditStore
from src.config import settings
from src.exchange.account import get_account_info
from src.exchange.client import BinanceDemoAccountInactiveError, ExchangeClient
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.live.data_quality import OHLCVQualityValidator, timeframe_seconds
from src.monitoring.alerter import Alerter
from src.monitoring.heartbeat import RuntimeHeartbeat
from src.risk.portfolio import PortfolioManager
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager
from src.security import redact_text
from src.signals.aggregator import SignalAggregator
from src.signals.decision_policy import (
    strategy_minimum_confidence,
    strategy_quality_gate_from_settings,
)
from src.signals.invocation import generate_with_context
from src.signals.signal_gate import SignalGate
from src.strategies import StrategyRegistry


class LiveTradingLoop:
    def __init__(
        self,
        *,
        audit_store: AuditStore | None = None,
    ):
        self.mode = "trade"
        self.audit_store = audit_store or AuditStore()
        self.client = ExchangeClient()
        self.order_mgr = OrderManager(self.client, audit_store=self.audit_store)
        self.portfolio = PortfolioManager()
        self.sizer = PositionSizer(self.portfolio)
        self.sl_mgr = StopLossManager(
            min_stop_loss_pct=settings.min_stop_loss_pct,
            max_stop_loss_pct=settings.max_stop_loss_pct,
        )
        self.alerter = Alerter(
            telegram_token=settings.telegram_bot_token,
            telegram_chat_id=settings.telegram_chat_id,
            discord_webhook=settings.discord_webhook_url,
            telegram_commands_enabled=settings.telegram_commands_enabled,
            telegram_allowed_user_ids=settings.telegram_allowed_user_ids_list,
            telegram_poll_timeout=settings.telegram_poll_timeout_seconds,
            queue_size=settings.telegram_alert_queue_size,
            delivery_timeout=settings.telegram_delivery_timeout_seconds,
        )
        self.pos_mgr = PositionManager(
            self.client,
            self.order_mgr,
            self.sizer,
            self.sl_mgr,
            self.portfolio,
            alerter=self.alerter,
            mode=self.mode,
            audit_store=self.audit_store,
        )
        self.aggregator = SignalAggregator()
        self.data_quality = OHLCVQualityValidator()
        self.strategy_quality = strategy_quality_gate_from_settings()
        self.strategy_registry = StrategyRegistry(settings)
        self.signal_gate = self._load_signal_gate()
        self._market_regimes: dict[str, int] = {}
        self._last_processed_candles: dict[str, object] = {}
        self._disabled_scope_log: set[str] = set()
        self._reversal_locks: dict[str, asyncio.Lock] = {}
        self._entry_reservation_lock = asyncio.Lock()
        self._pending_entry_sides: dict[str, int] = {"long": 0, "short": 0}
        self._next_scan_due: dict[str, float] = {}
        self._account_refresh_failures = 0
        self._account_degradation_alerted = False
        self._connectivity_failures = 0
        self._connectivity_outage_until = 0.0
        self._connectivity_outage_alerted = False
        self._last_connectivity_alert_at = 0.0
        self._last_account_refresh_at = 0.0
        self._account_refresh_interval_seconds = (
            settings.account_refresh_interval_seconds
        )
        self._last_reconciliation_at = 0.0
        self._reconciliation_interval_seconds = settings.reconciliation_interval_seconds
        self._market_data_semaphore = asyncio.Semaphore(
            settings.market_data_concurrency
        )
        self._candle_cache: dict[str, pd.DataFrame] = {}
        self._scalp_stream_tasks: list[asyncio.Task] = []
        self._scalp_stream_heartbeat: dict[str, float] = {}
        self._scalp_stream_degraded: set[str] = set()
        self._scalp_stream_failures: dict[str, int] = {}
        self._last_scalp_management_at = 0.0
        self._scalp_peak_prices: dict[str, float] = {}
        self._scalp_initial_risk: dict[str, float] = {}
        self._scalp_partial_completed: set[str] = set()
        # Break-even/trailing state for non-scalp (swing) positions - see
        # _manage_swing_position. Previously these strategies had no
        # analogous mechanism at all: a fixed stop/target set at entry,
        # never adjusted until stop, target, or max-hold timeout.
        self._swing_peak_prices: dict[str, float] = {}
        self._swing_initial_risk: dict[str, float] = {}
        self._stopped = False
        self._heartbeat = RuntimeHeartbeat(settings.runtime_heartbeat_path)
        self._heartbeat_task: asyncio.Task | None = None
        # Updated every time the main loop completes a unit of real work -
        # one processed (symbol, timeframe) pair, or one finished step of
        # the loop body - NOT once per full sweep. The heartbeat publisher
        # (a separate asyncio task) checks this before writing a "running"
        # heartbeat, so a main loop hung on an await that never resolves
        # stops refreshing the heartbeat instead of looking perpetually
        # healthy to the supervisor.
        #
        # Per-sweep granularity was the wrong measurement: it timed "how
        # long does a full sweep take" rather than "is the loop alive", so
        # when indicator computation was stalling the event loop and REST
        # reads looked like 4-10s hangs, a fully healthy loop needed >90s
        # per sweep and was killed mid-trade (2026-09-08 03:36:26, "has
        # not progressed in 102s", fired while a BTCUSDT entry was being
        # placed). See _mark_loop_progress for the invariant that keeps
        # this honest.
        self._last_loop_progress_at = time.monotonic()
        # Startup ("bootstrapping") heartbeat state. start() can take
        # minutes on a slow link - exchange connect, account snapshot,
        # audit restore, full exchange reconciliation - and until the
        # main-loop publisher starts, NOTHING refreshed the heartbeat. The
        # supervisor grants an extended grace only while the heartbeat
        # state is "launching"/"bootstrapping", but the only such record
        # was the one the supervisor wrote itself before spawning, so it
        # simply aged out and every restarted child was killed mid-startup
        # having logged nothing at all (2026-09-07 18:46, 2026-09-08
        # 05:10 - all died at 180-186s).
        #
        # A dedicated thread now publishes "bootstrapping" for the whole
        # pre-loop sequence, but only while the phase marker below keeps
        # advancing - so a startup genuinely wedged on one await still
        # stops heartbeating and is still restarted.
        self._bootstrap_lock = threading.Lock()
        self._bootstrap_phase = "starting"
        self._bootstrap_phase_at = time.monotonic()
        self._bootstrap_stop = threading.Event()
        self._bootstrap_thread: threading.Thread | None = None

    # Connect, fetch account, then scan every configured timeframe in sequence
    async def start(self):
        try:
            # Publish "bootstrapping" heartbeats across the whole startup
            # sequence below. The one-shot "starting" write this replaces
            # was actively harmful: "starting" is not in the supervisor's
            # extended-grace state set, so it downgraded the child's stale
            # limit from the startup grace back to the much tighter
            # heartbeat-stale limit while it was still legitimately
            # bootstrapping.
            self._set_bootstrap_phase("alerter_start")
            self._start_bootstrap_heartbeat()
            await self.alerter.start(self._handle_telegram_command)
            self._set_bootstrap_phase("initializing_alert")
            await self.alerter.initializing_alert(
                self.mode, settings.binance_environment
            )
            self._set_bootstrap_phase("exchange_connect")
            await self.client.connect()
            self._set_bootstrap_phase("exchange_connected")
            logger.info(
                "Starting trading loop on Binance "
                f"{settings.binance_environment.upper()}"
            )
            logger.info("Binance API URL: {}", settings.binance_api_url)
            self.audit_store.safe_record_event(
                "bot_started",
                f"Trading loop started in {self.mode} mode",
                mode=self.mode,
                payload={
                    "symbols": settings.symbols_list,
                    "timeframes": settings.timeframes_list,
                },
            )

            self._set_bootstrap_phase("restore_risk_state")
            self._restore_persisted_risk_state()
            self._set_bootstrap_phase("account_snapshot")
            await self._refresh_account(required=True)
            if self.portfolio.account:
                logger.info(
                    f"Account equity: {self.portfolio.account.total_equity:.2f}"
                )
            self._set_bootstrap_phase("restore_open_trades")
            self.pos_mgr.trades.restore_open_trades_from_audit()
            self._set_bootstrap_phase("restore_exit_cooldowns")
            self.pos_mgr.trades.restore_recent_exit_cooldowns()
            self._set_bootstrap_phase("exchange_reconciliation")
            reconciled = await self.pos_mgr.reconcile_exchange_state(
                auto_close_unmanaged=True
            )
            self._set_bootstrap_phase("reconciliation_complete")
            if reconciled is None:
                logger.warning(
                    "Startup reconciliation unavailable; exchange may be unreachable. "
                    "Continuing in degraded mode — new entries blocked, "
                    "existing positions retain protective orders."
                )
            elif reconciled is False:
                level = self.audit_store.get_trading_level()
                logger.warning(
                    f"Startup reconciliation failed; trading level={level}. "
                    "New entries are blocked, existing positions retain "
                    "protective orders."
                )
            else:
                self._set_bootstrap_phase("restored_exposure_cleanup")
                await self._cleanup_restored_position_exposure()
            self._set_bootstrap_phase("manual_request_recovery")
            interrupted_requests = (
                self.audit_store.fail_interrupted_manual_trade_requests()
            )
            if interrupted_requests:
                logger.critical(
                    "{} interrupted dashboard trade request(s) require review",
                    interrupted_requests,
                )
            self._last_reconciliation_at = time.monotonic()

            self._set_bootstrap_phase("set_leverage")
            await self._set_configured_leverage()

            self._set_bootstrap_phase("startup_alert")
            await self.alerter.startup_alert(
                self.mode,
                settings.binance_environment,
                settings.symbols_list,
            )
            self._set_bootstrap_phase("scalp_streams")
            self._start_scalp_streams()
            # Reset the clock right as heartbeat publishing begins, not at
            # __init__ time - construction happens before the (potentially
            # long) startup sequence above, so an unreset timestamp here
            # would look falsely stale on the very first stall check below.
            self._last_loop_progress_at = time.monotonic()
            # Stop the bootstrap publisher BEFORE starting the main-loop
            # publisher, never the reverse: a surviving bootstrap thread
            # would keep writing fresh "bootstrapping" records over the
            # heartbeat that _heartbeat_publisher deliberately withholds
            # when the main loop stalls, defeating that detector entirely.
            self._stop_bootstrap_heartbeat()
            self._start_heartbeat_publisher()

            while True:
                await self._refresh_account_if_due()
                self._mark_loop_progress()
                await self._process_manual_trade_requests()
                self._mark_loop_progress()
                if self._network_outage_active():
                    logger.warning(
                        "Network outage cooldown active for {:.0f}s; "
                        "skipping exchange reconciliation, active management, "
                        "market scans, and new entries",
                        max(self._connectivity_outage_until - time.monotonic(), 0.0),
                    )
                else:
                    await self._reconcile_if_due()
                    self._mark_loop_progress()
                    await self._manage_scalp_positions_if_due()
                    self._mark_loop_progress()
                    await self._scan_due_timeframes_once()
                # Retained end-of-iteration mark. Still required: in the
                # outage branch above the loop does no work at all but is
                # very much alive, and this is its only liveness signal.
                self._mark_loop_progress()
                await asyncio.sleep(settings.scan_sleep_seconds)
        except asyncio.CancelledError:
            await self.stop("cancelled")
        except BinanceDemoAccountInactiveError as exc:
            error = self._describe_exception(exc)
            logger.warning(f"Trading loop stopped: {error}")
            self.audit_store.safe_record_event(
                "bot_error",
                "Demo account inactive",
                severity="warning",
                mode=self.mode,
                payload={"error": error},
            )
            await self.alerter.error_alert(error)
            await self.stop("demo account inactive")
        except Exception as e:
            error = self._describe_exception(e)
            logger.error(f"Trading loop error: {error}")
            self.audit_store.safe_record_event(
                "bot_error",
                "Live loop crashed",
                severity="critical",
                mode=self.mode,
                payload={"error": error},
            )
            await self.alerter.error_alert(error)
            await self.stop("fatal error")
        finally:
            # Idempotent safety net. stop() also stops this thread, but
            # tests monkeypatch stop(), and a leaked daemon thread would
            # keep rewriting the heartbeat file after the loop is gone.
            self._stop_bootstrap_heartbeat()

    async def _write_heartbeat(self, state: str, **details: object) -> None:
        try:
            await self._heartbeat.write_async(state, **details)
        except Exception as exc:
            logger.warning(
                "Runtime heartbeat update failed (state={}): {}",
                state,
                self._describe_exception(exc),
            )

    async def _emit_heartbeat(self, state: str, **details: object) -> None:
        result = self._write_heartbeat(state, **details)
        if asyncio.iscoroutine(result):
            await result

    def _set_bootstrap_phase(self, phase: str) -> None:
        """Mark forward progress through the startup sequence.

        The bootstrap publisher refreshes the heartbeat only while this
        marker keeps moving, so a startup wedged on a single await is still
        detected and restarted by the supervisor.
        """
        with self._bootstrap_lock:
            self._bootstrap_phase = phase
            self._bootstrap_phase_at = time.monotonic()
        logger.debug("Bootstrap phase: {}", phase)

    def _start_bootstrap_heartbeat(self) -> None:
        if self._bootstrap_thread is not None:
            return
        self._bootstrap_stop.clear()
        self._bootstrap_thread = threading.Thread(
            target=self._publish_bootstrap_heartbeat,
            name="bootstrap-heartbeat",
            daemon=True,
        )
        self._bootstrap_thread.start()

    def _stop_bootstrap_heartbeat(self) -> None:
        thread = self._bootstrap_thread
        self._bootstrap_thread = None
        if thread is None:
            return
        self._bootstrap_stop.set()
        thread.join(timeout=5.0)
        if thread.is_alive():
            logger.warning(
                "Bootstrap heartbeat thread did not stop within 5s; it is a "
                "daemon thread and will not block shutdown"
            )

    def _publish_bootstrap_heartbeat(self) -> None:
        # Deliberately a plain thread, not an asyncio task: the startup
        # sequence makes blocking sync calls (audit/SQLite restore) that
        # would freeze an event-loop publisher and reproduce the very
        # "child did not publish a valid heartbeat" kill this exists to
        # prevent.
        #
        # Do NOT pass instance_id here - RuntimeHeartbeat.write() reads
        # TRADING_BOT_INSTANCE_ID from the environment the supervisor
        # injected, and the supervisor matches on exactly that.
        interval = max(
            1.0,
            min(15.0, settings.supervisor_heartbeat_stale_seconds / 3),
        )
        stall_after = settings.supervisor_bootstrap_phase_stall_seconds
        while not self._bootstrap_stop.is_set():
            with self._bootstrap_lock:
                phase = self._bootstrap_phase
                stuck_for = time.monotonic() - self._bootstrap_phase_at
            if stuck_for > stall_after:
                logger.error(
                    "Startup phase {!r} has not advanced in {:.0f}s "
                    "(> {:.0f}s); withholding heartbeat so the supervisor "
                    "detects the wedged startup and restarts",
                    phase,
                    stuck_for,
                    stall_after,
                )
            else:
                try:
                    self._heartbeat.write(
                        "bootstrapping",
                        mode=self.mode,
                        environment=settings.binance_environment,
                        phase=phase,
                    )
                except Exception as exc:  # noqa: BLE001 - never kill startup
                    logger.warning(
                        "Bootstrap heartbeat write failed ({}); continuing",
                        self._describe_exception(exc),
                    )
            self._bootstrap_stop.wait(interval)

    def _mark_loop_progress(self) -> None:
        # Call on COMPLETION of a unit of real work in the main loop, never
        # on entry: the signal the watchdog needs is "an await resolved", so
        # a unit that starts and then hangs forever must leave the timestamp
        # frozen.
        #
        # Safe to call from inside an asyncio.gather() child even though a
        # hang in one child is briefly masked by siblings still finishing:
        # every gather in the loop body is awaited to completion, so a
        # permanently hung child stops the main loop from ever starting
        # another sweep. The pool of siblings that can mask it is therefore
        # finite and drains, after which the timestamp freezes and the
        # stall is detected. Two rules keep that property true - do not
        # break them:
        #   1. Never call this from a fire-and-forget task. In particular
        #      _process_market_frame (and _on_candle beneath it) is shared
        #      with _run_scalp_stream, which the main loop launches via
        #      create_task and never awaits - a mark there would let a
        #      healthy WebSocket stream mask a genuinely deadlocked main
        #      loop forever. tests/test_live_loop.py guards this.
        #   2. Never convert the loop-body gathers to create_task().
        self._last_loop_progress_at = time.monotonic()

    def _start_heartbeat_publisher(self) -> None:
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_publisher())

    async def _heartbeat_publisher(self) -> None:
        interval = max(
            1.0,
            min(15.0, settings.supervisor_heartbeat_stale_seconds / 3),
        )
        # This task runs independently of the main scan loop, so it must not
        # treat "the event loop is scheduling me" as "the trading loop is
        # actually working" - a deadlocked await in the main loop would
        # otherwise leave this publisher writing fresh "running" heartbeats
        # forever, and the supervisor (which only checks heartbeat age) would
        # never restart the hung process.
        # Deliberately NOT supervisor_heartbeat_stale_seconds: that governs
        # how stale the heartbeat file may get, this governs how long the
        # loop may go without completing a unit of work. They are different
        # questions and 90s is far too tight for the second - one market
        # read can legitimately take ~93s on its own.
        stall_after = settings.bot_progress_stall_seconds
        while not self._stopped:
            try:
                stalled_for = time.monotonic() - self._last_loop_progress_at
                if stalled_for > stall_after:
                    logger.error(
                        "Main scan loop has not progressed in {:.0f}s "
                        "(> {:.0f}s); withholding heartbeat so the "
                        "supervisor detects the stall and restarts",
                        stalled_for,
                        stall_after,
                    )
                else:
                    await self._emit_heartbeat(
                        "running",
                        mode=self.mode,
                        environment=settings.binance_environment,
                        open_positions=len(self.pos_mgr.open_trades),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Heartbeat publisher crashed ({}); restarting loop",
                    self._describe_exception(exc),
                )
                await asyncio.sleep(interval)
                continue
            await asyncio.sleep(interval)

    async def _stop_heartbeat_publisher(self) -> None:
        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat_task is None or heartbeat_task is asyncio.current_task():
            return
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task

    async def _set_configured_leverage(self) -> None:
        for symbol in settings.symbols_list:
            try:
                await self.client.set_leverage(symbol, settings.max_leverage)
            except BinanceDemoAccountInactiveError:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not set leverage for {}: {} — continuing with "
                    "current leverage",
                    symbol,
                    self._describe_exception(exc),
                )

    async def _scan_timeframes_once(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
    ):
        scan_symbols = symbols or settings.symbols_list
        scan_timeframes = timeframes or settings.timeframes_list
        await asyncio.gather(
            *(
                self._bounded_process_timeframe(symbol, timeframe)
                for symbol in scan_symbols
                for timeframe in scan_timeframes
            )
        )

    async def _process_manual_trade_requests(self, limit: int = 3) -> None:
        for _ in range(limit):
            request = self.audit_store.claim_next_manual_trade_request()
            if request is None:
                return
            request_id = str(request["request_id"])
            try:
                result = await self._execute_manual_trade_request(request)
            except asyncio.CancelledError:
                self.audit_store.finish_manual_trade_request(
                    request_id,
                    status="cancelled",
                    result={"reason": "trading loop stopped"},
                )
                raise
            except Exception as exc:
                error = self._describe_exception(exc)
                logger.error(
                    "Manual trade request {} failed: {}",
                    request_id,
                    error,
                )
                self.audit_store.finish_manual_trade_request(
                    request_id,
                    status="failed",
                    result={"reason": error},
                )
                self.audit_store.safe_record_event(
                    "manual_trade_failed",
                    "Dashboard manual trade request failed",
                    severity="error",
                    symbol=str(request["symbol"]),
                    mode=self.mode,
                    correlation_id=request.get("correlation_id"),
                    payload={"request_id": request_id, "error": error},
                )
                continue

            success = bool(result.get("success"))
            status = "completed" if success else "failed"
            result_payload = {
                key: value for key, value in result.items() if key != "success"
            }
            try:
                self.audit_store.finish_manual_trade_request(
                    request_id,
                    status=status,
                    result=result_payload,
                )
            except Exception as exc:
                error = self._describe_exception(exc)
                logger.critical(
                    "Manual trade request {} executed but terminal status "
                    "could not be persisted: {}",
                    request_id,
                    error,
                )
                self.audit_store.safe_record_event(
                    "manual_trade_result_persistence_failed",
                    "Manual trade result requires operator reconciliation",
                    severity="critical",
                    symbol=str(request["symbol"]),
                    mode=self.mode,
                    correlation_id=result_payload.get("correlation_id")
                    or request.get("correlation_id"),
                    payload={
                        "request_id": request_id,
                        "intended_status": status,
                        "result": result_payload,
                        "error": error,
                    },
                )
                continue
            self.audit_store.safe_record_event(
                f"manual_trade_{status}",
                f"Dashboard manual trade request {status}",
                severity="warning" if status == "failed" else "info",
                symbol=str(request["symbol"]),
                mode=self.mode,
                correlation_id=result_payload.get("correlation_id")
                or request.get("correlation_id"),
                payload={"request_id": request_id, **result_payload},
            )

    async def _execute_manual_trade_request(
        self,
        request: dict,
    ) -> dict[str, object]:
        action = str(request["action"])
        symbol = str(request["symbol"]).upper()
        reason = str(request.get("reason") or "dashboard manual trade")
        if action == "close":
            return await self._execute_manual_close(request, reason)
        if action in {"close-symbol", "close-all"}:
            return await self._execute_manual_bulk_close(action, symbol, reason)

        side = str(request.get("side") or "").lower()
        timeframe = str(request.get("timeframe") or "")
        options = request.get("options") or {}
        require_quality = bool(options.get("require_quality", True))
        scope = self._model_scope_key(symbol, timeframe)
        if symbol not in settings.symbols_list:
            return {"success": False, "reason": "symbol is not configured"}
        if timeframe not in settings.timeframes_list:
            return {"success": False, "reason": "timeframe is not configured"}
        # Manual (dashboard-queued) entries must honor the same
        # connectivity/account-freshness guards as automated entries -
        # without this, a queued "open long/short" request could still open
        # a leveraged position while the bot has itself determined account
        # or exchange connectivity data is unreliable.
        if self._network_outage_active():
            return {
                "success": False,
                "reason": "network outage cooldown active",
            }
        if self._entry_runtime_blocked({"symbol": symbol, "timeframe": timeframe}, scope):
            return {
                "success": False,
                "reason": "entry blocked: stale account data or disabled scope",
            }
        if side not in {"long", "short"}:
            return {"success": False, "reason": "side must be long or short"}

        from src.indicators.compute import compute_all_indicators

        df = await self.client.fetch_ohlcv(
            symbol,
            timeframe,
            limit=settings.min_ohlcv_candles + 1,
        )
        quality = self.data_quality.validate(df, timeframe)
        if not quality.valid:
            return {
                "success": False,
                "reason": f"market data rejected: {quality.reason}",
            }
        closed_df = df.iloc[:-1].copy()
        indicators = await asyncio.to_thread(compute_all_indicators, closed_df)
        price = float(indicators["close"].iloc[-1])
        atr = float(indicators["atr"].iloc[-1])
        if price <= 0 or atr <= 0:
            return {
                "success": False,
                "reason": "validated price or ATR is unavailable",
            }
        quality_result = await self._manual_entry_quality(
            symbol,
            timeframe,
            side,
            indicators,
            require_quality,
        )
        if isinstance(quality_result, dict):
            return quality_result

        enter = self.pos_mgr.enter_long if side == "long" else self.pos_mgr.enter_short
        opened = await enter(
            symbol,
            price,
            atr,
            settings.max_leverage,
            timeframe,
            signal_timestamp=closed_df.index[-1],
        )
        position_key = self.pos_mgr.trades.position_key(symbol, timeframe)
        correlation_id = self.pos_mgr.trades.trade_correlation_ids.get(
            position_key,
            "",
        )
        return {
            "success": bool(opened),
            "reason": reason if opened else "entry was blocked or not filled",
            "position_key": position_key,
            "correlation_id": correlation_id,
            "side": side,
            "timeframe": timeframe,
            "reference_price": price,
            "atr": atr,
            "exchange_leverage": settings.max_leverage,
            "quality_required": require_quality,
            "quality_score": quality_result.score if quality_result else None,
        }

    async def _execute_manual_bulk_close(
        self,
        action: str,
        symbol: str,
        reason: str,
    ) -> dict[str, object]:
        if action == "close-symbol":
            closed = await self.pos_mgr.close_symbol(
                symbol,
                reason=f"dashboard: {reason}",
            )
            return {
                "success": closed,
                "reason": reason if closed else "symbol exit was not confirmed",
                "symbol": symbol,
            }
        await self.pos_mgr.close_all()
        closed = not self.pos_mgr.open_trades
        return {
            "success": closed,
            "reason": reason if closed else "one or more exits were not confirmed",
            "remaining_positions": len(self.pos_mgr.open_trades),
        }

    async def _manual_entry_quality(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        indicators,
        required: bool,
    ):
        if not required:
            return None
        from src.indicators.compute import compute_all_indicators

        direction = 1 if side == "long" else -1
        higher_regime = self._market_regimes.get(self._model_scope_key(symbol, "1h"))
        if timeframe in {"1m", "3m", "5m", "15m", "30m"} and (higher_regime is None):
            context_df = await self.client.fetch_ohlcv(
                symbol,
                "1h",
                limit=settings.min_ohlcv_candles + 1,
            )
            context_quality = self.data_quality.validate(context_df, "1h")
            if not context_quality.valid:
                return {
                    "success": False,
                    "reason": f"1h context rejected: {context_quality.reason}",
                }
            context_indicators = await asyncio.to_thread(
                compute_all_indicators, context_df.iloc[:-1].copy()
            )
            higher_regime = int(context_indicators["trend_regime"].iloc[-1])
        quality_result = self.strategy_quality.evaluate(
            indicators,
            direction,
            timeframe=timeframe,
            higher_timeframe_regime=higher_regime,
        )
        if not quality_result.accepted:
            return {
                "success": False,
                "reason": f"strategy quality rejected: {quality_result.reason}",
                "quality_score": quality_result.score,
                "quality_metrics": quality_result.metrics,
            }
        return quality_result

    async def _execute_manual_close(
        self,
        request: dict,
        reason: str,
    ) -> dict[str, object]:
        correlation_id = str(request.get("correlation_id") or "")
        position_key = next(
            (
                key
                for key, value in self.pos_mgr.trades.trade_correlation_ids.items()
                if value == correlation_id
            ),
            "",
        )
        if not position_key:
            return {
                "success": False,
                "reason": "open trade was not found in the runtime registry",
                "correlation_id": correlation_id,
            }
        trade = self.pos_mgr.open_trades.get(position_key)
        if trade is None or trade.symbol != str(request["symbol"]).upper():
            return {
                "success": False,
                "reason": "trade request does not match the open position",
                "correlation_id": correlation_id,
            }

        await self.pos_mgr.exit_position(
            position_key,
            reason=f"dashboard: {reason}",
        )
        closed = position_key not in self.pos_mgr.open_trades
        return {
            "success": closed,
            "reason": reason if closed else "reduce-only exit was not confirmed",
            "position_key": position_key,
            "correlation_id": correlation_id,
        }

    async def _scan_due_timeframes_once(self):
        now = time.monotonic()
        pairs = self._due_scan_pairs(now)
        processed = await asyncio.gather(
            *(
                self._bounded_process_timeframe(symbol, timeframe)
                for symbol, timeframe in pairs
            )
        )
        # Only advance the schedule for pairs actually processed this cycle.
        # A pair can be skipped mid-gather (_bounded_process_timeframe
        # short-circuits if a network outage is flagged by a sibling task
        # while this one was still queued behind the semaphore); advancing
        # its due-time regardless would permanently skip the candle that was
        # due this cycle instead of retrying it once the outage clears.
        for (symbol, timeframe), was_processed in zip(pairs, processed):
            if not was_processed:
                continue
            self._next_scan_due[self._scan_key(symbol, timeframe)] = (
                self._next_candle_scan_due(
                    timeframe,
                    monotonic_now=time.monotonic(),
                    epoch_now=time.time(),
                )
            )

    def _due_scan_pairs(self, now: float) -> list[tuple[str, str]]:
        due = []
        for symbol in settings.symbols_list:
            for timeframe in settings.timeframes_list:
                key = self._scan_key(symbol, timeframe)
                if timeframe in {"1m", "3m"} and self._scalp_stream_is_healthy(
                    key, now
                ):
                    continue
                if now >= self._next_scan_due.get(key, 0.0):
                    due.append((symbol, timeframe))
        return sorted(
            due,
            key=lambda pair: (
                pair[1] not in {"1m", "3m"},
                self._timeframe_seconds(pair[1]),
                pair[0],
            ),
        )

    async def _bounded_process_timeframe(
        self,
        symbol: str,
        timeframe: str,
    ) -> bool:
        if self._network_outage_active():
            # Deliberately no progress mark: this is a zero-work
            # short-circuit, not a completed unit. The outage branch in the
            # main loop body marks progress for the iteration instead.
            return False
        async with self._market_data_semaphore:
            await self._process_timeframe(symbol, timeframe)
        # One (symbol, timeframe) pair fully processed - the unit of work
        # that corresponds to a "Candle: ..." log line. Marking here is
        # what stops a merely-slow sweep from being mistaken for a hang:
        # the semaphore serializes pairs, so a full sweep legitimately runs
        # for minutes when REST is degraded.
        #
        # This is a correct boundary because this method's call sites are
        # both inside gathers the main loop awaits - unlike
        # _process_market_frame, which is also reached from the
        # fire-and-forget scalp stream tasks.
        self._mark_loop_progress()
        return True

    async def _process_timeframe(self, symbol: str, timeframe: str):
        try:
            df = await self._fetch_rest_market_frame(symbol, timeframe)
            # Bound the gap between marks across the two slowest awaits of
            # a pair. The fetch can legitimately take ~93s (3 ccxt attempts
            # x 30s plus backoff), and _process_market_frame beneath this
            # may place an entry - market order, fill resolve, stop, limit -
            # each its own multi-second exchange round trip. Without these
            # two marks the entire order-placement chain is invisible to
            # the watchdog, which is exactly the window in which the
            # 2026-09-08 03:36:26 false stall fired.
            #
            # Safe here (unlike in _process_market_frame itself): this
            # method is only ever reached from the awaited scan gather,
            # never from the fire-and-forget scalp stream tasks.
            self._mark_loop_progress()
            self._record_connectivity_success()
            self._cache_market_frame(symbol, timeframe, df)
            await self._process_market_frame(
                symbol,
                timeframe,
                df,
                market_data_source="rest",
            )
            self._mark_loop_progress()
        except Exception as e:
            await self._record_connectivity_failure(e)
            await self._report_timeframe_error(symbol, timeframe, e)

    async def _fetch_rest_market_frame(
        self,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        limit = settings.min_ohlcv_candles + 1
        # Freshness verification/retry applies to every timeframe, not just
        # 1m/3m scalp: without it, a candle the exchange publishes late at a
        # scan boundary is silently treated as fresh, deduped as an
        # already-seen candle by _process_market_frame, and never retried -
        # that hour's/day's trade evaluation is then permanently skipped.
        # Reuses the scalp-tuned retry budget (small, bounded cost) rather
        # than adding a parallel set of settings for the same mechanism.
        attempts = settings.scalp_rest_freshness_attempts
        frame = pd.DataFrame()
        for attempt in range(1, attempts + 1):
            frame = await self.client.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit,
            )
            if len(frame) < 2:
                return frame
            candle_timestamp = pd.Timestamp(frame.index[-2])
            expected_timestamp = self._latest_expected_closed_candle_timestamp(
                timeframe
            )
            if candle_timestamp.tzinfo is None:
                candle_timestamp = candle_timestamp.tz_localize("UTC")
            else:
                candle_timestamp = candle_timestamp.tz_convert("UTC")
            if candle_timestamp >= expected_timestamp:
                return frame
            if attempt < attempts:
                latency = self._closed_candle_latency_seconds(
                    {"timestamp": candle_timestamp, "timeframe": timeframe}
                )
                logger.info(
                    "REST candle stale for {} {}: {:.2f}s; refetching ({}/{})",
                    symbol,
                    timeframe,
                    latency,
                    attempt,
                    attempts,
                )
                await asyncio.sleep(settings.scalp_rest_freshness_retry_seconds)
        return frame

    @classmethod
    def _latest_expected_closed_candle_timestamp(
        cls,
        timeframe: str,
        *,
        now: pd.Timestamp | None = None,
    ) -> pd.Timestamp:
        current = now or pd.Timestamp.now(tz="UTC")
        if current.tzinfo is None:
            current = current.tz_localize("UTC")
        else:
            current = current.tz_convert("UTC")
        interval = cls._timeframe_seconds(timeframe)
        grace = settings.scalp_rest_candle_close_grace_seconds
        safe_epoch = current.timestamp() - grace
        latest_close_epoch = int(safe_epoch // interval) * interval
        return pd.Timestamp(latest_close_epoch - interval, unit="s", tz="UTC")

    async def _process_market_frame(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        *,
        market_data_source: str = "rest",
    ) -> None:
        try:
            from src.indicators.compute import compute_all_indicators

            quality = self.data_quality.validate(df, timeframe)
            if not quality.valid:
                logger.warning(
                    f"Skipping {symbol} {timeframe}: data quality failed: "
                    f"{quality.reason}"
                )
                self.audit_store.safe_record_event(
                    "data_quality_rejected",
                    "OHLCV data rejected",
                    severity="warning",
                    symbol=symbol,
                    mode=self.mode,
                    payload={"timeframe": timeframe, "reason": quality.reason},
                )
                return

            candle_key = f"{symbol}_{timeframe}"
            closed_df = df.iloc[:-1].copy()
            candle_timestamp = closed_df.index[-1]
            if self._last_processed_candles.get(candle_key) == candle_timestamp:
                logger.debug(f"Skipping duplicate candle: {symbol} {timeframe}")
                return

            self._last_processed_candles[candle_key] = candle_timestamp
            # Off the event loop: this blocks ~1s per 200-row frame, and
            # with several pairs in flight it stalled the loop for seconds
            # at a time - which is what made awaited REST reads look like
            # 4-10s hangs and tripped the loop-progress watchdog into
            # killing a healthy process mid-trade (2026-09-08 03:36:26).
            df_ind = await asyncio.to_thread(compute_all_indicators, closed_df)
            last = closed_df.iloc[-1]
            candle = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": candle_timestamp,
                "open": last["open"],
                "high": last["high"],
                "low": last["low"],
                "close": last["close"],
                "volume": last["volume"],
                "market_data_source": market_data_source,
            }
            await self._on_candle(candle, df_ind=df_ind)
        except Exception as e:
            await self._report_timeframe_error(symbol, timeframe, e)

    async def _report_timeframe_error(
        self,
        symbol: str,
        timeframe: str,
        exc: Exception,
    ) -> None:
        error = self._describe_exception(exc)
        logger.error(f"Timeframe scan error for {symbol} {timeframe}: {error}")
        self.audit_store.safe_record_event(
            "market_data_scan_failed",
            "OHLCV timeframe scan failed",
            severity="warning",
            symbol=symbol,
            mode=self.mode,
            payload={"timeframe": timeframe, "error": error},
        )
        if self._data_feed_alert_allowed():
            await self.alerter.data_feed_alert(
                self.mode,
                symbol,
                timeframe,
                error,
            )

    def _network_outage_active(self) -> bool:
        return time.monotonic() < self._connectivity_outage_until

    async def _record_connectivity_failure(self, exc: Exception) -> None:
        self._connectivity_failures += 1
        if self._connectivity_failures < settings.network_outage_failure_threshold:
            return
        self._connectivity_outage_until = max(
            self._connectivity_outage_until,
            time.monotonic() + settings.network_outage_cooldown_seconds,
        )
        now = time.monotonic()
        if (
            self._connectivity_outage_alerted
            and now - self._last_connectivity_alert_at
            < settings.network_outage_alert_cooldown_seconds
        ):
            return
        reason = (
            "Exchange connectivity unavailable; pausing new market scans and "
            "entries while protected positions remain managed by exchange orders"
        )
        self._connectivity_outage_alerted = True
        self._last_connectivity_alert_at = now
        self.audit_store.safe_record_event(
            "network_outage_detected",
            reason,
            severity="critical",
            mode=self.mode,
            payload={
                "consecutive_failures": self._connectivity_failures,
                "cooldown_seconds": settings.network_outage_cooldown_seconds,
                "error": self._describe_exception(exc),
            },
        )
        await self.alerter.error_alert(reason)

    def _record_connectivity_success(self) -> None:
        if self._connectivity_failures == 0 and not self._connectivity_outage_alerted:
            return
        previous_failures = self._connectivity_failures
        was_alerted = self._connectivity_outage_alerted
        self._connectivity_failures = 0
        self._connectivity_outage_until = 0.0
        self._connectivity_outage_alerted = False
        if was_alerted:
            self.audit_store.safe_record_event(
                "network_outage_recovered",
                "Exchange connectivity recovered",
                mode=self.mode,
                payload={"previous_consecutive_failures": previous_failures},
            )

    def _data_feed_alert_allowed(self) -> bool:
        if not self._network_outage_active():
            return True
        now = time.monotonic()
        return (
            now - self._last_connectivity_alert_at
            >= settings.network_outage_alert_cooldown_seconds
        )

    def _cache_market_frame(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        key = self._scan_key(symbol, timeframe)
        cached = self._candle_cache.get(key)
        combined = df if cached is None else pd.concat([cached, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined = combined.tail(settings.candle_cache_size)
        self._candle_cache[key] = combined
        return combined

    def _start_scalp_streams(self) -> None:
        if not settings.scalp_streaming_enabled:
            logger.info(
                "Scalp market data: prioritized REST fast lane "
                "(WebSocket disabled for {})",
                settings.binance_environment,
            )
            self.audit_store.safe_record_event(
                "scalp_rest_fast_lane_enabled",
                "Scalp scopes using prioritized REST market data",
                mode=self.mode,
                payload={"environment": settings.binance_environment},
            )
            return
        for symbol in settings.symbols_list:
            for timeframe in ("1m", "3m"):
                if timeframe not in settings.timeframes_list:
                    continue
                self._scalp_stream_tasks.append(
                    asyncio.create_task(
                        self._run_scalp_stream(symbol, timeframe),
                        name=f"scalp-stream-{symbol}-{timeframe}",
                    )
                )

    async def _run_scalp_stream(self, symbol: str, timeframe: str) -> None:
        key = self._scan_key(symbol, timeframe)
        while not self._stopped:
            try:
                frame = await self.client.watch_ohlcv(symbol, timeframe)
                self._scalp_stream_heartbeat[key] = time.monotonic()
                self._scalp_stream_failures.pop(key, None)
                if key in self._scalp_stream_degraded:
                    self._scalp_stream_degraded.remove(key)
                    self.audit_store.safe_record_event(
                        "scalp_stream_recovered",
                        "Scalp WebSocket stream recovered",
                        symbol=symbol,
                        mode=self.mode,
                        payload={"timeframe": timeframe},
                    )
                cached = self._cache_market_frame(symbol, timeframe, frame)
                if len(cached) < settings.min_ohlcv_candles + 1:
                    continue
                async with self._market_data_semaphore:
                    await self._process_market_frame(
                        symbol,
                        timeframe,
                        cached,
                        market_data_source="websocket",
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Scalp stream {} {} failed: {}; REST fallback remains active",
                    symbol,
                    timeframe,
                    self._describe_exception(exc),
                )
                self._scalp_stream_heartbeat.pop(key, None)
                failures = self._scalp_stream_failures.get(key, 0) + 1
                self._scalp_stream_failures[key] = failures
                if key not in self._scalp_stream_degraded:
                    self._scalp_stream_degraded.add(key)
                    self.audit_store.safe_record_event(
                        "scalp_stream_degraded",
                        "Scalp WebSocket unavailable; REST fallback active",
                        severity="warning",
                        symbol=symbol,
                        mode=self.mode,
                        payload={
                            "timeframe": timeframe,
                            "error": self._describe_exception(exc),
                        },
                    )
                if failures >= settings.scalp_stream_failure_threshold:
                    self.audit_store.safe_record_event(
                        "scalp_stream_circuit_open",
                        "Scalp WebSocket circuit opened; REST fallback active",
                        severity="warning",
                        symbol=symbol,
                        mode=self.mode,
                        payload={
                            "timeframe": timeframe,
                            "failures": failures,
                            "retry_seconds": settings.scalp_stream_circuit_seconds,
                        },
                    )
                    await asyncio.sleep(settings.scalp_stream_circuit_seconds)
                    self._scalp_stream_failures[key] = 0
                else:
                    await asyncio.sleep(settings.exchange_read_backoff_seconds)

    def _scalp_stream_is_healthy(self, key: str, now: float) -> bool:
        if not settings.scalp_streaming_enabled:
            return False
        heartbeat = self._scalp_stream_heartbeat.get(key)
        return (
            heartbeat is not None
            and now - heartbeat <= settings.scalp_stream_fallback_seconds
        )

    def _load_signal_gate(self) -> SignalGate:
        gate = SignalGate(
            min_win_rate=settings.signal_source_gate_min_accuracy,
            min_samples=settings.signal_source_gate_min_samples,
            min_avg_directional_bps=settings.signal_source_gate_min_avg_bps,
        )
        if not settings.signal_source_gate_enabled:
            return gate
        try:
            gate.load_from_audit(
                self.audit_store.load_signal_observations(
                    limit=settings.signal_source_gate_lookback
                )
            )
            blocked_count = sum(
                1
                for stats in gate.summary.values()
                if stats["total"] >= settings.signal_source_gate_min_samples
                and (
                    stats["win_rate"] < settings.signal_source_gate_min_accuracy
                    and stats["avg_directional_bps"]
                    < settings.signal_source_gate_min_avg_bps
                )
            )
            if blocked_count:
                logger.info(
                    "Signal source gate loaded {} blocked setup family(s)",
                    blocked_count,
                )
        except Exception as exc:
            logger.warning(
                "Could not load signal source gate from audit history: {}",
                self._describe_exception(exc),
            )
        return gate

    def _cleanup_scalp_state(self) -> None:
        active = set(self.pos_mgr.open_trades)
        for key in list(self._scalp_peak_prices):
            if key not in active:
                del self._scalp_peak_prices[key]
        for key in list(self._scalp_initial_risk):
            if key not in active:
                del self._scalp_initial_risk[key]
        self._scalp_partial_completed.intersection_update(active)
        for key in list(self._swing_peak_prices):
            if key not in active:
                del self._swing_peak_prices[key]
        for key in list(self._swing_initial_risk):
            if key not in active:
                del self._swing_initial_risk[key]

    async def _manage_scalp_positions_if_due(self) -> None:
        now = time.monotonic()
        if (
            now - self._last_scalp_management_at
            < settings.scalp_management_interval_seconds
        ):
            return
        self._last_scalp_management_at = now
        self._cleanup_scalp_state()
        positions = [
            (key, trade)
            for key, trade in list(self.pos_mgr.open_trades.items())
            if trade.strategy == "scalp"
        ]
        non_scalp_positions = [
            (key, trade)
            for key, trade in list(self.pos_mgr.open_trades.items())
            if trade.strategy != "scalp"
        ]
        stale_keys = {
            key
            for key, trade in non_scalp_positions
            if self._strategy_max_hold_reached(trade)
        }
        stale_positions = [
            (key, trade) for key, trade in non_scalp_positions if key in stale_keys
        ]
        swing_positions = [
            (key, trade)
            for key, trade in non_scalp_positions
            if key not in stale_keys
        ]
        await asyncio.gather(
            *(self._manage_scalp_position(key, trade) for key, trade in positions),
            *(
                self._manage_swing_position(key, trade)
                for key, trade in swing_positions
            ),
            *(
                self._exit_position_safely(key, "maximum hold")
                for key, _ in stale_positions
            ),
        )

    async def _exit_position_safely(self, position_key: str, reason: str) -> bool:
        try:
            await self.pos_mgr.exit_position(position_key, reason)
            return True
        except Exception as exc:
            logger.warning(
                "Managed exit failed for {} ({}): {}",
                position_key,
                reason,
                self._describe_exception(exc),
            )
            self.audit_store.safe_record_event(
                "managed_exit_failed",
                f"Managed exit failed for {position_key}",
                severity="warning",
                mode=self.mode,
                payload={
                    "position_key": position_key,
                    "reason": reason,
                    "error": self._describe_exception(exc),
                },
            )
            return False

    def _cleanup_scalp_position(self, position_key: str) -> None:
        # Also clears the swing-side dicts for this key (harmless no-op if
        # it was never a swing position). This matters for the same-key
        # reversal path (_close_opposite_symbol_trades_if_needed): the old
        # side's exit and the new side's entry share one position_key with
        # no gap where the key is absent from open_trades, so the periodic
        # _cleanup_scalp_state() sweep (which only clears a key once it's
        # missing from open_trades entirely) never catches the transition -
        # without this, a fresh reversed trade could inherit the previous
        # trade's stale initial_risk/peak price and misprice favorable_r
        # for break-even/trailing-stop decisions.
        self._scalp_peak_prices.pop(position_key, None)
        self._scalp_initial_risk.pop(position_key, None)
        self._scalp_partial_completed.discard(position_key)
        self._swing_peak_prices.pop(position_key, None)
        self._swing_initial_risk.pop(position_key, None)

    async def _manage_scalp_position(self, position_key: str, trade) -> None:
        try:
            if self._scalp_max_hold_reached(trade):
                await self.pos_mgr.exit_position(position_key, "scalp maximum hold")
                self._cleanup_scalp_position(position_key)
                return

            ticker = await self.client.fetch_ticker(trade.symbol)
            price = float(ticker.get("mark") or ticker.get("last") or 0)
            if price <= 0:
                return
            correlation_id = self.pos_mgr.trades.trade_correlation_ids.get(
                position_key,
                "",
            )
            stop_loss, take_profit = self.audit_store.get_trade_protection_levels(
                correlation_id
            )
            if stop_loss is None:
                return
            initial_risk = self._scalp_initial_risk.setdefault(
                position_key,
                abs(trade.entry_price - float(stop_loss)),
            )
            if initial_risk <= 0:
                return
            favorable_move = (
                price - trade.entry_price
                if trade.side == "long"
                else trade.entry_price - price
            )
            favorable_r = favorable_move / initial_risk
            peak = self._scalp_peak_prices.get(position_key, trade.entry_price)
            peak = max(peak, price) if trade.side == "long" else min(peak, price)
            self._scalp_peak_prices[position_key] = peak

            if await self._take_scalp_partial_if_due(
                position_key,
                trade,
                initial_risk,
                favorable_r,
            ):
                stop_loss, take_profit = self.audit_store.get_trade_protection_levels(
                    correlation_id
                )
                if stop_loss is None:
                    return

            new_stop = self._responsive_scalp_stop(
                trade,
                price,
                peak,
                float(stop_loss),
                initial_risk,
                favorable_r,
            )
            if new_stop is None:
                return
            replacement_id = await self.pos_mgr.protection.replace_stop(
                position_key,
                trade,
                new_stop,
            )
            if not replacement_id:
                return
            target_id = self.pos_mgr.protection.active_tps.get(position_key)
            self.audit_store.update_open_trade_state(
                correlation_id,
                quantity=trade.quantity,
                stop_loss=new_stop,
                take_profit=take_profit,
                stop_order_id=replacement_id,
                take_profit_order_id=target_id,
            )
            self.audit_store.safe_record_event(
                "scalp_stop_advanced",
                "Scalp protective stop advanced",
                symbol=trade.symbol,
                mode=self.mode,
                correlation_id=correlation_id or None,
                payload={
                    "position_key": position_key,
                    "price": price,
                    "stop_loss": new_stop,
                    "favorable_r": favorable_r,
                },
            )
        except Exception as exc:
            logger.warning(
                "Scalp position management failed for {}: {}",
                position_key,
                self._describe_exception(exc),
            )

    async def _manage_swing_position(self, position_key: str, trade) -> None:
        """Break-even/trailing management for non-scalp strategies.

        Mirrors _manage_scalp_position's break-even and trailing logic
        (minus partial-profit-taking, which stays scalp-only) so a trend/
        range/breakout/reversal/countertrend/transition trade that moves
        meaningfully favorable and then reverses captures some of that
        move instead of riding a static, never-adjusted stop all the way
        back to a full loss.
        """
        try:
            ticker = await self.client.fetch_ticker(trade.symbol)
            price = float(ticker.get("mark") or ticker.get("last") or 0)
            if price <= 0:
                return
            correlation_id = self.pos_mgr.trades.trade_correlation_ids.get(
                position_key,
                "",
            )
            stop_loss, take_profit = self.audit_store.get_trade_protection_levels(
                correlation_id
            )
            if stop_loss is None:
                return
            initial_risk = self._swing_initial_risk.setdefault(
                position_key,
                abs(trade.entry_price - float(stop_loss)),
            )
            if initial_risk <= 0:
                return
            favorable_move = (
                price - trade.entry_price
                if trade.side == "long"
                else trade.entry_price - price
            )
            favorable_r = favorable_move / initial_risk
            peak = self._swing_peak_prices.get(position_key, trade.entry_price)
            peak = max(peak, price) if trade.side == "long" else min(peak, price)
            self._swing_peak_prices[position_key] = peak

            new_stop = self._responsive_swing_stop(
                trade,
                price,
                peak,
                float(stop_loss),
                initial_risk,
                favorable_r,
            )
            if new_stop is None:
                return
            replacement_id = await self.pos_mgr.protection.replace_stop(
                position_key,
                trade,
                new_stop,
            )
            if not replacement_id:
                return
            target_id = self.pos_mgr.protection.active_tps.get(position_key)
            self.audit_store.update_open_trade_state(
                correlation_id,
                quantity=trade.quantity,
                stop_loss=new_stop,
                take_profit=take_profit,
                stop_order_id=replacement_id,
                take_profit_order_id=target_id,
            )
            self.audit_store.safe_record_event(
                "swing_stop_advanced",
                "Swing protective stop advanced",
                symbol=trade.symbol,
                mode=self.mode,
                correlation_id=correlation_id or None,
                payload={
                    "position_key": position_key,
                    "strategy": trade.strategy,
                    "price": price,
                    "stop_loss": new_stop,
                    "favorable_r": favorable_r,
                },
            )
        except Exception as exc:
            logger.warning(
                "Swing position management failed for {}: {}",
                position_key,
                self._describe_exception(exc),
            )

    @staticmethod
    def _responsive_swing_stop(
        trade,
        price: float,
        peak: float,
        current_stop: float,
        initial_risk: float,
        favorable_r: float,
    ) -> float | None:
        candidates = [current_stop]
        if favorable_r >= settings.swing_break_even_trigger_r:
            offset = trade.entry_price * settings.swing_break_even_offset_bps / 10_000
            candidates.append(
                trade.entry_price + offset
                if trade.side == "long"
                else trade.entry_price - offset
            )
        if favorable_r >= settings.swing_trailing_trigger_r:
            distance = initial_risk * settings.swing_trailing_distance_r
            candidates.append(
                peak - distance if trade.side == "long" else peak + distance
            )
        proposed = max(candidates) if trade.side == "long" else min(candidates)
        improvement = (
            proposed - current_stop if trade.side == "long" else current_stop - proposed
        )
        minimum = price * settings.swing_stop_update_min_bps / 10_000
        if improvement < minimum:
            return None
        if (trade.side == "long" and proposed >= price) or (
            trade.side == "short" and proposed <= price
        ):
            return None
        return round(proposed, 8)

    def _scalp_max_hold_reached(self, trade) -> bool:
        timestamp = trade.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        policy = self.strategy_registry.get("scalp")
        try:
            max_hold = policy.max_hold_seconds(str(trade.timeframe or ""))
        except ValueError:
            max_hold = settings.scalp_max_hold_seconds_default
        return age >= max_hold

    def _strategy_max_hold_reached(self, trade) -> bool:
        timestamp = trade.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        policy = self.strategy_registry.get(trade.strategy)
        try:
            max_hold = policy.max_hold_seconds(str(trade.timeframe or ""))
        except ValueError:
            logger.warning(
                "Cannot manage unsupported strategy scope {}:{}",
                trade.strategy,
                trade.timeframe,
            )
            return False
        return age >= max_hold

    @staticmethod
    def _scalp_cost_floor_bps() -> float:
        return max(
            settings.scalp_break_even_offset_bps,
            settings.scalp_effective_round_trip_fee_bps
            + settings.scalp_min_net_edge_bps,
        )

    @classmethod
    def _scalp_cost_floor_r(cls, trade, initial_risk: float) -> float:
        if initial_risk <= 0 or trade.entry_price <= 0:
            return float("inf")
        required_move = trade.entry_price * cls._scalp_cost_floor_bps() / 10_000
        return required_move / initial_risk

    async def _take_scalp_partial_if_due(
        self,
        position_key: str,
        trade,
        initial_risk: float,
        favorable_r: float,
    ) -> bool:
        trigger_r = max(
            settings.scalp_partial_profit_trigger_r,
            self._scalp_cost_floor_r(trade, initial_risk),
        )
        due = (
            settings.scalp_partial_profit_enabled
            and position_key not in self._scalp_partial_completed
            and favorable_r >= trigger_r
        )
        if not due:
            return False
        reduced = await self.pos_mgr.partial_exit_position(
            position_key,
            settings.scalp_partial_profit_fraction,
            "scalp partial profit",
        )
        if reduced:
            self._scalp_partial_completed.add(position_key)
        return reduced

    @staticmethod
    def _responsive_scalp_stop(
        trade,
        price: float,
        peak: float,
        current_stop: float,
        initial_risk: float,
        favorable_r: float,
    ) -> float | None:
        candidates = [current_stop]
        cost_floor_r = LiveTradingLoop._scalp_cost_floor_r(trade, initial_risk)
        break_even_trigger_r = max(
            settings.scalp_break_even_trigger_r,
            cost_floor_r,
        )
        if favorable_r >= break_even_trigger_r:
            offset = (
                trade.entry_price * LiveTradingLoop._scalp_cost_floor_bps() / 10_000
            )
            candidates.append(
                trade.entry_price + offset
                if trade.side == "long"
                else trade.entry_price - offset
            )
        if favorable_r >= settings.scalp_trailing_trigger_r:
            distance = initial_risk * settings.scalp_trailing_distance_r
            candidates.append(
                peak - distance if trade.side == "long" else peak + distance
            )
        proposed = max(candidates) if trade.side == "long" else min(candidates)
        improvement = (
            proposed - current_stop if trade.side == "long" else current_stop - proposed
        )
        minimum = price * settings.scalp_stop_update_min_bps / 10_000
        if improvement < minimum:
            return None
        if (trade.side == "long" and proposed >= price) or (
            trade.side == "short" and proposed <= price
        ):
            return None
        return round(proposed, 8)

    async def _refresh_account(self, required: bool = False):
        self._last_account_refresh_at = time.monotonic()
        try:
            account = await asyncio.wait_for(
                get_account_info(self.client), timeout=30.0
            )
        except Exception as e:
            await self._record_connectivity_failure(e)
            self._account_refresh_failures += 1
            logger.warning(
                "Account refresh failed ({} consecutive): {}",
                self._account_refresh_failures,
                self._describe_exception(e),
            )
            if required:
                raise RuntimeError(
                    "Initial account refresh failed in trade mode"
                ) from e
            if (
                self._account_refresh_failures >= 3
                and not self._account_degradation_alerted
            ):
                reason = (
                    "Account data unavailable after three scheduled refreshes; "
                    "new entries remain paused"
                )
                self._account_degradation_alerted = True
                self.audit_store.safe_record_event(
                    "account_refresh_degraded",
                    reason,
                    severity="critical",
                    mode=self.mode,
                    payload={
                        "consecutive_failures": self._account_refresh_failures,
                        "error": self._describe_exception(e),
                    },
                )
                await self.alerter.error_alert(reason)
            return

        self._record_connectivity_success()
        if self._account_degradation_alerted:
            self.audit_store.safe_record_event(
                "account_refresh_recovered",
                "Account data refresh recovered",
                mode=self.mode,
                payload={
                    "previous_consecutive_failures": self._account_refresh_failures
                },
            )
        self._account_refresh_failures = 0
        self._account_degradation_alerted = False
        previous_peak = self.portfolio.peak_equity
        self.portfolio.update_account(account)
        logger.info(
            "Account snapshot: url={} equity={:.2f} wallet={:.2f} "
            "available={:.2f} upnl={:.2f} margin={:.2%}",
            settings.binance_api_url,
            account.total_equity,
            account.wallet_balance,
            account.available_balance,
            account.unrealized_pnl,
            account.margin_ratio,
        )
        if self.portfolio.peak_equity > previous_peak:
            self.audit_store.set_control(
                "peak_equity",
                str(self.portfolio.peak_equity),
                "highest observed account equity",
            )
        logger.debug(f"Account equity refreshed: {account.total_equity:.2f}")

    async def _refresh_account_if_due(self) -> None:
        elapsed = time.monotonic() - self._last_account_refresh_at
        if elapsed < self._account_refresh_interval_seconds:
            return
        await self._refresh_account()

    def _restore_persisted_risk_state(self) -> None:
        self.portfolio.restore_risk_state(self.audit_store.load_closed_trades())
        raw_peak = self.audit_store.get_control("peak_equity", "0")
        try:
            self.portfolio.peak_equity = max(float(raw_peak), 0.0)
        except ValueError:
            logger.warning("Ignoring invalid persisted peak_equity control")
            self.portfolio.peak_equity = 0.0

    @classmethod
    def _describe_exception(cls, exc: Exception) -> str:
        chain = []
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = cls._redact_sensitive_url_parts(str(current))
            chain.append(f"{type(current).__name__}: {message}")
            current = current.__cause__ or current.__context__
        return " -> ".join(chain)

    @staticmethod
    def _redact_sensitive_url_parts(message: str) -> str:
        return redact_text(message)

    async def _reconcile_if_due(self):
        elapsed = time.monotonic() - self._last_reconciliation_at
        if elapsed < self._reconciliation_interval_seconds:
            return
        try:
            reconciled = await asyncio.wait_for(
                self.pos_mgr.reconcile_exchange_state(
                    auto_close_unmanaged=settings.auto_close_unmanaged_positions
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Exchange reconciliation timed out after 60s")
            self._last_reconciliation_at = time.monotonic()
            return
        self._last_reconciliation_at = time.monotonic()
        if reconciled is None:
            logger.warning(
                "Runtime reconciliation was inconclusive; keeping protected "
                "positions and retrying at the next interval"
            )
            return
        if reconciled is False:
            level = self.audit_store.get_trading_level()
            logger.warning(
                f"Runtime reconciliation failed; trading level={level}. "
                "New entries are blocked, existing positions retain protective orders."
            )

    async def _cleanup_restored_position_exposure(self) -> None:
        if not settings.restored_position_exposure_cleanup_enabled:
            return
        open_trades = self.pos_mgr.open_trades
        if not open_trades:
            return

        keys_to_close: set[str] = set()

        for symbol in sorted({trade.symbol for trade in open_trades.values()}):
            symbol_trades = [
                (key, trade)
                for key, trade in open_trades.items()
                if trade.symbol == symbol
            ]
            keys_to_close.update(
                self._excess_position_keys(
                    symbol_trades,
                    settings.max_positions_per_symbol,
                )
            )

        for side in ("long", "short"):
            side_trades = [
                (key, trade) for key, trade in open_trades.items() if trade.side == side
            ]
            keys_to_close.update(
                self._excess_position_keys(
                    side_trades,
                    settings.max_same_direction_positions,
                )
            )

        if not keys_to_close:
            return

        logger.warning(
            "Closing {} restored position(s) that exceed current exposure limits: {}",
            len(keys_to_close),
            ", ".join(sorted(keys_to_close)),
        )
        self.audit_store.safe_record_event(
            "restored_position_exposure_cleanup",
            "Closing restored positions that exceed current exposure limits",
            severity="warning",
            mode=self.mode,
            payload={
                "position_keys": sorted(keys_to_close),
                "max_positions_per_symbol": settings.max_positions_per_symbol,
                "max_same_direction_positions": settings.max_same_direction_positions,
            },
        )

        for position_key in sorted(
            keys_to_close,
            key=lambda key: self._open_trade_sort_key(open_trades[key]),
        ):
            if position_key not in open_trades:
                continue
            await self.pos_mgr.exit_position(
                position_key,
                "startup restored exposure limit",
            )

    @classmethod
    def _excess_position_keys(
        cls,
        trades: list[tuple[str, object]],
        limit: int,
    ) -> list[str]:
        if len(trades) <= limit:
            return []
        ordered = sorted(trades, key=lambda item: cls._open_trade_sort_key(item[1]))
        return [key for key, _trade in ordered[: len(trades) - limit]]

    async def _reserve_directional_entry(self, side: str) -> tuple[bool, str]:
        async with self._entry_reservation_lock:
            open_count = sum(
                1 for trade in self.pos_mgr.open_trades.values() if trade.side == side
            )
            pending_count = self._pending_entry_sides.get(side, 0)
            effective_count = open_count + pending_count
            if effective_count >= settings.max_same_direction_positions:
                return (
                    False,
                    (
                        f"same-direction limit reached: {effective_count} "
                        f"existing/pending {side}(s)"
                    ),
                )
            self._pending_entry_sides[side] = pending_count + 1
            return True, ""

    async def _release_directional_entry(self, side: str) -> None:
        async with self._entry_reservation_lock:
            self._pending_entry_sides[side] = max(
                self._pending_entry_sides.get(side, 0) - 1,
                0,
            )

    @staticmethod
    def _scan_key(symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        return timeframe_seconds(timeframe)

    @classmethod
    def _higher_confirmation_timeframe(cls, timeframe: str) -> str | None:
        """The nearest configured timeframe strictly above `timeframe`.

        Used for multi-timeframe confirmation: a signal on `timeframe` is
        checked against the regime on the next timeframe up, not a fixed
        timeframe regardless of what's being evaluated (e.g. using 15m
        context for a 1h/4h signal would be checking noise against noise).
        """
        try:
            current_seconds = cls._timeframe_seconds(timeframe)
        except (KeyError, ValueError, IndexError):
            return None
        candidates = []
        for candidate in settings.timeframes_list:
            try:
                candidate_seconds = cls._timeframe_seconds(candidate)
            except (KeyError, ValueError, IndexError):
                continue
            if candidate_seconds > current_seconds:
                candidates.append((candidate_seconds, candidate))
        if not candidates:
            return None
        return min(candidates)[1]

    @classmethod
    def _closed_candle_latency_seconds(cls, candle: dict) -> float:
        timestamp = pd.Timestamp(candle["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        closed_at = timestamp + pd.Timedelta(
            seconds=cls._timeframe_seconds(candle["timeframe"])
        )
        return max(
            (pd.Timestamp.now(tz="UTC") - closed_at).total_seconds(),
            0.0,
        )

    @classmethod
    def _next_candle_scan_due(
        cls,
        timeframe: str,
        *,
        monotonic_now: float,
        epoch_now: float,
    ) -> float:
        interval = cls._timeframe_seconds(timeframe)
        next_boundary = (int(epoch_now) // interval + 1) * interval
        grace = settings.candle_close_grace_seconds
        if timeframe in {"1m", "3m"} and not settings.scalp_streaming_enabled:
            grace = max(
                grace,
                settings.scalp_rest_candle_close_grace_seconds,
            )
        delay = next_boundary - epoch_now + grace
        return monotonic_now + max(delay, settings.scan_sleep_seconds)

    # Called on each new closed candle.
    @staticmethod
    def _model_scope_key(symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}:{timeframe}"

    async def _on_candle(self, candle: dict, df_ind=None):
        logger.info(
            f"Candle: {candle['symbol']} {candle['timeframe']} "
            f"close={candle['close']:.2f}"
        )
        try:
            resolved = self.audit_store.resolve_signal_observations(
                symbol=candle["symbol"],
                timeframe=candle["timeframe"],
                candle_timestamp=candle["timestamp"],
                outcome_price=candle["close"],
            )
            if resolved:
                logger.debug(
                    "Resolved {} prior signal observation(s) for {} {}",
                    resolved,
                    candle["symbol"],
                    candle["timeframe"],
                )
        except Exception as exc:
            logger.warning(
                "Could not resolve signal observations for {} {}: {}",
                candle["symbol"],
                candle["timeframe"],
                self._describe_exception(exc),
            )
        if df_ind is not None and "trend_regime" in df_ind.columns:
            regime = df_ind["trend_regime"].iloc[-1]
            if regime == regime:
                self._market_regimes[
                    self._model_scope_key(
                        candle["symbol"],
                        candle["timeframe"],
                    )
                ] = int(regime)
        await self._execute_trade(candle, df_ind=df_ind)

    # Generate signal and enter position if criteria are met
    async def _execute_trade(self, candle: dict, df_ind=None):  # noqa: C901
        if self._network_outage_active():
            logger.info(
                "Signal skipped for {}:{}: network outage cooldown active",
                candle["symbol"],
                candle["timeframe"],
            )
            return
        symbol = candle["symbol"]
        scope = self._model_scope_key(symbol, candle["timeframe"])
        timeframe = candle["timeframe"]
        logger.debug(f"Evaluating {symbol} {timeframe} at {candle['close']:.2f}")

        if self._entry_runtime_blocked(candle, scope):
            return

        # Generate multiple strategy signals per candle
        try:
            if df_ind is None:
                from src.indicators.compute import compute_all_indicators

                df = await self.client.fetch_ohlcv(symbol, timeframe, limit=200)
                df_ind = await asyncio.to_thread(compute_all_indicators, df)

            from src.signals.regime import detect_regime, regime_appropriate_strategies

            regime = detect_regime(df_ind)
            higher_confirmation_tf = self._higher_confirmation_timeframe(timeframe)
            htf_bias = (
                self._market_regimes.get(
                    self._model_scope_key(symbol, higher_confirmation_tf), 0
                )
                if higher_confirmation_tf
                else 0
            )
            signals = generate_with_context(
                self.aggregator,
                df_ind,
                htf_bias,
            )

            if not signals:
                logger.debug(f"No signals for {scope}")
                return
            signals = self._ordered_strategy_signals(signals)

            atr = (
                df_ind["atr"].iloc[-1]
                if "atr" in df_ind.columns
                else df_ind["close"].iloc[-1] * 0.01
            )
            if atr <= 0 or atr != atr:
                atr = df_ind["close"].iloc[-1] * 0.01

            higher_regime = self._market_regimes.get(
                self._model_scope_key(symbol, "1h")
            )
            if higher_regime is None and "trend_regime" in df_ind.columns:
                higher_regime = int(df_ind["trend_regime"].iloc[-1])

            allowed_by_regime = regime_appropriate_strategies(regime)

            for signal in signals:
                strategy = signal.strategy
                if signal.direction == 0:
                    continue

                policy = self.strategy_registry.get(strategy)
                minimum_confidence = self._strategy_minimum_confidence(strategy)
                if not self.strategy_registry.is_enabled(strategy):
                    reason = f"{strategy} strategy is disabled"
                    logger.debug(f"Signal skipped for {scope} {strategy}: {reason}")
                    self._record_signal_observation(
                        candle,
                        signal,
                        strategy=strategy,
                        minimum_confidence=minimum_confidence,
                        decision="disabled",
                        reason=reason,
                    )
                    continue
                if self._is_strategy_scope_disabled(scope, strategy):
                    reason = f"{scope}:{strategy} is disabled by performance review"
                    self._record_strategy_scope_disabled(candle, scope, strategy)
                    self._record_signal_observation(
                        candle,
                        signal,
                        strategy=strategy,
                        minimum_confidence=minimum_confidence,
                        decision="disabled",
                        reason=reason,
                    )
                    continue
                if not policy.supports(timeframe):
                    logger.debug(
                        f"{policy.name} does not support {timeframe}; skipping"
                    )
                    continue

                adjusted_confidence = signal.confidence
                if settings.signal_source_gate_enabled:
                    source_gate = self.signal_gate.evaluate(
                        signal.ta_source,
                        strategy,
                        timeframe,
                    )
                    if not source_gate.passed:
                        reason = f"source gate blocked: {source_gate.reason}"
                        logger.info(
                            f"Signal skipped for {scope} {strategy}: "
                            f"{reason}; samples={source_gate.total_samples} "
                            f"accuracy={source_gate.win_rate:.2%} "
                            f"avg_bps={source_gate.avg_directional_bps:.2f}"
                        )
                        self._record_signal_observation(
                            candle,
                            signal,
                            strategy=strategy,
                            minimum_confidence=minimum_confidence,
                            decision="source_gate_rejected",
                            reason=reason,
                        )
                        continue
                    # SignalGate.evaluate() sets confidence_multiplier < 1.0
                    # in exactly the cases where passed is also False (both
                    # driven by the same win_rate_failed/edge_failed check) -
                    # so this branch was dead: by the time control reaches
                    # here, passed was True, which means confidence_multiplier
                    # is always exactly 1.0. Removed rather than "fixed" into
                    # a real graduated-penalty feature, since that would be a
                    # behavior change to live signal filtering with no
                    # evidence backing where the pass/fail boundary should
                    # sit - SignalGate is a hard pass/fail gate in practice
                    # today, not a soft one.

                if adjusted_confidence < minimum_confidence:
                    detail = (
                        f"{strategy} confidence {adjusted_confidence:.4f} "
                        f"< {minimum_confidence:.4f}"
                    )
                    reason = signal.decision_reason or detail
                    logger.debug(f"Signal skipped for {scope} {strategy}: {detail}")
                    self._record_signal_observation(
                        candle,
                        signal,
                        strategy=strategy,
                        minimum_confidence=minimum_confidence,
                        decision="skipped",
                        reason=reason,
                    )
                    continue

                if strategy not in allowed_by_regime:
                    reason = (
                        f"{regime.market_type} regime does not permit {strategy} "
                        f"(allowed: {', '.join(allowed_by_regime) or 'none'})"
                    )
                    if settings.regime_filter_enforced:
                        logger.debug(f"Signal skipped for {scope} {strategy}: {reason}")
                        self._record_signal_observation(
                            candle,
                            signal,
                            strategy=strategy,
                            minimum_confidence=minimum_confidence,
                            decision="regime_rejected",
                            reason=reason,
                        )
                        continue
                    logger.debug(
                        f"{reason}; evaluating independent strategy rules "
                        f"(regime_filter_enforced=False)"
                    )

                position_key = self.pos_mgr.trades.position_key(
                    symbol,
                    timeframe,
                    strategy,
                )
                if position_key in self.pos_mgr.open_trades:
                    logger.debug(
                        f"Already in {strategy} position for {scope}; skipping"
                    )
                    continue

                if strategy == "scalp":
                    latency = self._closed_candle_latency_seconds(candle)
                    source = str(candle.get("market_data_source") or "rest")
                    latency_limit = (
                        settings.scalp_max_signal_latency_seconds
                        if source == "websocket"
                        else settings.scalp_rest_max_signal_latency_seconds
                    )
                    if latency > latency_limit:
                        reason = (
                            f"scalp signal stale: {latency:.2f}s "
                            f"> {latency_limit:.2f}s source={source}"
                        )
                        logger.info(f"Signal skipped for {scope}: {reason}")
                        self._record_signal_observation(
                            candle,
                            signal,
                            strategy=strategy,
                            minimum_confidence=minimum_confidence,
                            decision="stale",
                            reason=reason,
                        )
                        continue

                quality = self.strategy_quality.evaluate(
                    df_ind,
                    signal.direction,
                    timeframe=timeframe,
                    higher_timeframe_regime=higher_regime,
                    strategy=strategy,
                    signal_source=signal.ta_source,
                    atr_mult_sl=policy.atr_stop_multiplier,
                    reward_risk_ratio=policy.reward_risk_ratio,
                )
                if not quality.accepted:
                    quality_decision = (
                        "rejected"
                        if settings.strategy_quality_gate_enforced
                        else "advisory"
                    )
                    logger.info(
                        f"Strategy quality {quality_decision} {scope} {strategy}: "
                        f"score={quality.score:.2f} reason={quality.reason}"
                    )
                    self.audit_store.safe_record_event(
                        f"signal_quality_{quality_decision}",
                        f"Strategy quality {quality_decision} {scope} {strategy}",
                        symbol=symbol,
                        mode=self.mode,
                        payload={
                            "scope": scope,
                            "strategy": strategy,
                            "direction": signal.direction,
                            "confidence": signal.confidence,
                            "quality_score": quality.score,
                            "reason": quality.reason,
                            "metrics": quality.metrics,
                            "enforced": settings.strategy_quality_gate_enforced,
                        },
                    )
                    if settings.strategy_quality_gate_enforced:
                        self._record_signal_observation(
                            candle,
                            signal,
                            strategy=strategy,
                            minimum_confidence=minimum_confidence,
                            decision="quality_rejected",
                            reason=quality.reason,
                            quality=quality,
                        )
                        continue

                same_dir_label = "long" if signal.direction == 1 else "short"
                reserved, reason = await self._reserve_directional_entry(same_dir_label)
                if not reserved:
                    logger.info("Signal skipped for {}: {}", scope, reason)
                    self._record_signal_observation(
                        candle,
                        signal,
                        strategy=strategy,
                        minimum_confidence=minimum_confidence,
                        decision="risk_rejected",
                        reason=reason,
                        quality=quality,
                    )
                    continue

                try:
                    reversal_allowed, reversal_reason = (
                        self._same_symbol_reversal_allowed(
                            symbol=symbol,
                            timeframe=timeframe,
                            signal=signal,
                            quality=quality,
                        )
                    )
                    if not reversal_allowed:
                        logger.info(
                            "Signal skipped for {}: {}",
                            scope,
                            reversal_reason,
                        )
                        self._record_signal_observation(
                            candle,
                            signal,
                            strategy=strategy,
                            minimum_confidence=minimum_confidence,
                            decision="reversal_deferred",
                            reason=reversal_reason,
                            quality=quality,
                        )
                        continue

                    policy_allowed, policy_reason = self._entry_policy_allows_reversal(
                        symbol,
                        strategy,
                        signal,
                    )
                    if not policy_allowed:
                        logger.info("Signal skipped for {}: {}", scope, policy_reason)
                        self._record_signal_observation(
                            candle,
                            signal,
                            strategy=strategy,
                            minimum_confidence=minimum_confidence,
                            decision="risk_rejected",
                            reason=policy_reason,
                            quality=quality,
                        )
                        continue

                    reverse_ready, reversed_position = (
                        await self._close_opposite_symbol_trades_if_needed(
                            symbol,
                            signal,
                            strategy,
                            scope,
                        )
                    )
                    if not reverse_ready:
                        reason = "opposite-side position could not be closed"
                        self._record_signal_observation(
                            candle,
                            signal,
                            strategy=strategy,
                            minimum_confidence=minimum_confidence,
                            decision="risk_rejected",
                            reason=reason,
                            quality=quality,
                        )
                        continue

                    self.audit_store.safe_record_event(
                        "signal_accepted",
                        f"Accepted {scope} {strategy} signal",
                        symbol=symbol,
                        mode=self.mode,
                        payload={
                            "scope": scope,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "direction": signal.direction,
                            "confidence": adjusted_confidence,
                            "ta_source": signal.ta_source,
                            "decision_reason": signal.decision_reason,
                            "quality_score": quality.score,
                            "quality_reason": quality.reason,
                            "quality_metrics": quality.metrics,
                            "atr": float(atr),
                            "signal_price": float(candle["close"]),
                            "signal_timestamp": str(candle["timestamp"]),
                        },
                    )
                    observation_id = self._record_signal_observation(
                        candle,
                        signal,
                        strategy=strategy,
                        minimum_confidence=minimum_confidence,
                        decision="accepted",
                        reason=signal.decision_reason or quality.reason,
                        quality=quality,
                    )

                    if signal.direction == 1:
                        opened = await self.pos_mgr.enter_long(
                            symbol,
                            candle["close"],
                            atr,
                            settings.max_leverage,
                            timeframe=timeframe,
                            signal_timestamp=candle["timestamp"],
                            strategy=strategy,
                            ignore_reentry_cooldown=reversed_position,
                            market_context=df_ind.iloc[-1].to_dict(),
                        )
                    else:
                        opened = await self.pos_mgr.enter_short(
                            symbol,
                            candle["close"],
                            atr,
                            settings.max_leverage,
                            timeframe=timeframe,
                            signal_timestamp=candle["timestamp"],
                            strategy=strategy,
                            ignore_reentry_cooldown=reversed_position,
                            market_context=df_ind.iloc[-1].to_dict(),
                        )
                    if observation_id:
                        try:
                            self.audit_store.update_signal_execution(
                                observation_id,
                                "opened" if opened else "blocked_or_failed",
                            )
                        except Exception as exc:
                            logger.warning(
                                "Could not update signal execution status for {}: {}",
                                scope,
                                self._describe_exception(exc),
                            )
                    if opened:
                        logger.info(
                            f"Entered {strategy} {scope} at {candle['close']:.2f}"
                        )
                        break
                finally:
                    await self._release_directional_entry(same_dir_label)

        except Exception as e:
            error = self._describe_exception(e)
            logger.exception(f"Trade execution error for {symbol}: {error}")
            self.audit_store.safe_record_event(
                "trade_execution_error",
                f"Trade execution error for {symbol}",
                severity="error",
                symbol=symbol,
                mode=self.mode,
                payload={
                    "error_type": type(e).__name__,
                    "error": error,
                    "timeframe": timeframe,
                },
            )
            await self.alerter.trade_failed_alert(self.mode, symbol, error)

    def _record_signal_observation(
        self,
        candle: dict,
        signal,
        *,
        strategy: str | None = None,
        minimum_confidence: float,
        decision: str,
        reason: str,
        quality=None,
    ) -> str | None:
        try:
            return self.audit_store.record_signal_observation(
                candle_timestamp=candle["timestamp"],
                symbol=candle["symbol"],
                timeframe=candle["timeframe"],
                strategy=strategy or signal.strategy,
                direction=signal.direction,
                confidence=signal.confidence,
                minimum_confidence=minimum_confidence,
                decision=decision,
                reason=reason,
                signal_price=candle["close"],
                ta_source=signal.ta_source,
                quality_score=quality.score if quality is not None else None,
                quality_reason=quality.reason if quality is not None else None,
                metrics=quality.metrics if quality is not None else {},
            )
        except Exception as exc:
            logger.warning(
                "Could not persist signal observation for {} {}: {}",
                candle["symbol"],
                candle["timeframe"],
                self._describe_exception(exc),
            )
            return None

    @staticmethod
    def _strategy_minimum_confidence(strategy: str) -> float:
        return strategy_minimum_confidence(strategy)

    @staticmethod
    def _ordered_strategy_signals(signals):
        # Fixed tiebreaker only: the strongest signal this cycle should be
        # attempted first regardless of which strategy produced it. Sorting
        # by this priority ahead of confidence let a barely-qualifying
        # low-priority-table signal (e.g. breakout at 0.21) execute before a
        # much stronger one from another strategy (e.g. trend at 0.90).
        priority = {
            "scalp": 0,
            "breakout": 1,
            "trend": 2,
            "transition": 3,
            "range": 4,
            "reversal": 5,
            "countertrend": 6,
        }
        best_by_strategy = {}
        for signal in signals:
            current = best_by_strategy.get(signal.strategy)
            if current is None or signal.confidence > current.confidence:
                best_by_strategy[signal.strategy] = signal
        return sorted(
            best_by_strategy.values(),
            key=lambda signal: (
                -signal.confidence,
                priority.get(signal.strategy, 99),
            ),
        )

    @staticmethod
    def _strategy_scope_key(scope: str, strategy: str) -> str:
        return f"{scope}:{strategy}"

    @staticmethod
    def _is_strategy_scope_disabled(scope: str, strategy: str) -> bool:
        disabled = settings.disabled_strategy_scopes_set
        return (
            scope in disabled
            or LiveTradingLoop._strategy_scope_key(
                scope,
                strategy,
            )
            in disabled
        )

    def _record_strategy_scope_disabled(
        self,
        candle: dict,
        scope: str,
        strategy: str,
    ) -> None:
        disabled_key = self._strategy_scope_key(scope, strategy)
        if disabled_key in self._disabled_scope_log:
            return
        logger.warning(
            f"Strategy scope {disabled_key} is disabled by performance review"
        )
        self.audit_store.safe_record_event(
            "strategy_scope_disabled",
            f"Strategy scope {disabled_key} skipped",
            severity="warning",
            symbol=candle["symbol"],
            mode=self.mode,
            payload={
                "timeframe": candle["timeframe"],
                "scope": scope,
                "strategy": strategy,
            },
        )
        self._disabled_scope_log.add(disabled_key)

    def _same_symbol_reversal_allowed(
        self,
        *,
        symbol: str,
        timeframe: str,
        signal,
        quality,
    ) -> tuple[bool, str]:
        if not settings.same_symbol_reversal_guard_enabled:
            return True, ""

        target_side = "long" if signal.direction == 1 else "short"
        candidate_seconds = self._timeframe_seconds(timeframe)
        opposite_trades = [
            trade
            for trade in self.pos_mgr.open_trades.values()
            if trade.symbol == symbol and trade.side != target_side
        ]
        if not opposite_trades:
            return True, ""

        if not quality.accepted:
            return (
                False,
                (
                    "opposite-side reversal requires accepted strategy quality; "
                    f"quality rejected: {quality.reason}"
                ),
            )

        for trade in opposite_trades:
            trade_timeframe = str(trade.timeframe or timeframe)
            try:
                trade_seconds = self._timeframe_seconds(trade_timeframe)
            except ValueError:
                trade_seconds = candidate_seconds

            if candidate_seconds >= trade_seconds:
                continue

            reason = self._lower_timeframe_reversal_block_reason(
                trade=trade,
                trade_timeframe=trade_timeframe,
                trade_seconds=trade_seconds,
                candidate_timeframe=timeframe,
                candidate_seconds=candidate_seconds,
                target_side=target_side,
                signal=signal,
                quality=quality,
            )
            if reason:
                return False, reason

        return True, ""

    def _lower_timeframe_reversal_block_reason(
        self,
        *,
        trade,
        trade_timeframe: str,
        trade_seconds: int,
        candidate_timeframe: str,
        candidate_seconds: int,
        target_side: str,
        signal,
        quality,
    ) -> str:
        timeframe_ratio = trade_seconds / max(candidate_seconds, 1)
        max_ratio = settings.lower_timeframe_reversal_max_timeframe_ratio
        if timeframe_ratio > max_ratio:
            return (
                f"lower-timeframe {candidate_timeframe} {target_side} signal cannot "
                f"reverse {trade_timeframe} {trade.side} {trade.strategy} "
                f"position; timeframe ratio {timeframe_ratio:.1f} > {max_ratio:.1f}"
            )

        held_seconds = self._open_trade_age_seconds(trade)
        min_hold = settings.lower_timeframe_reversal_min_hold_seconds
        if held_seconds < min_hold:
            return (
                f"lower-timeframe {candidate_timeframe} {target_side} signal cannot "
                f"reverse {trade_timeframe} {trade.side} {trade.strategy} "
                f"position before {min_hold}s hold time"
            )

        min_confidence = settings.lower_timeframe_reversal_min_confidence
        if signal.confidence < min_confidence:
            return (
                f"lower-timeframe {candidate_timeframe} {target_side} reversal "
                f"confidence {signal.confidence:.2f} < {min_confidence:.2f}"
            )

        min_quality = settings.lower_timeframe_reversal_min_quality_score
        if not quality.accepted or quality.score < min_quality:
            return (
                f"lower-timeframe {candidate_timeframe} {target_side} reversal "
                f"requires confirmed quality score >= {min_quality:.2f}"
            )
        return ""

    def _entry_policy_allows_reversal(
        self,
        symbol: str,
        strategy: str,
        signal,
    ) -> tuple[bool, str]:
        target_side = "long" if signal.direction == 1 else "short"
        has_opposite_position = any(
            trade.symbol == symbol and trade.side != target_side
            for trade in self.pos_mgr.open_trades.values()
        )
        if not has_opposite_position:
            return True, ""

        allowed, reason = self.portfolio.can_trade(symbol, strategy)
        if allowed:
            return True, ""
        return False, f"entry policy blocks reversal: {reason}"

    @staticmethod
    def _open_trade_age_seconds(trade) -> float:
        opened_at = trade.timestamp
        if isinstance(opened_at, pd.Timestamp):
            opened_at = opened_at.to_pydatetime()
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - opened_at).total_seconds())

    @staticmethod
    def _open_trade_sort_key(trade) -> tuple[datetime, int]:
        opened_at = trade.timestamp
        if isinstance(opened_at, pd.Timestamp):
            opened_at = opened_at.to_pydatetime()
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        timeframe = str(trade.timeframe or "1m")
        try:
            interval = timeframe_seconds(timeframe)
        except ValueError:
            interval = 0
        return opened_at, interval

    async def _close_opposite_symbol_trades_if_needed(
        self,
        symbol: str,
        signal,
        strategy: str,
        scope: str,
    ) -> tuple[bool, bool]:
        if not settings.reverse_on_opposite_signal:
            return True, False

        lock = self._reversal_locks.get(symbol)
        if lock is None:
            lock = self._reversal_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            target_side = "long" if signal.direction == 1 else "short"
            opposite = [
                (key, trade)
                for key, trade in self.pos_mgr.open_trades.items()
                if trade.symbol == symbol and trade.side != target_side
            ]
            if not opposite:
                return True, False

            reason = f"reverse to {target_side} on {strategy} signal"
            reversed_position = False
            for position_key, trade in opposite:
                if position_key not in self.pos_mgr.open_trades:
                    continue
                logger.info(
                    "Closing opposite {} {} before {} entry for {}",
                    trade.side,
                    position_key,
                    target_side,
                    scope,
                )
                await self.pos_mgr.exit_position(position_key, reason)
                self._cleanup_scalp_position(position_key)
                reversed_position = True

            remaining = [
                key
                for key, trade in self.pos_mgr.open_trades.items()
                if trade.symbol == symbol and trade.side != target_side
            ]
            if remaining:
                logger.warning(
                    "Opposite-side position(s) still open for {} after reversal "
                    "close: {}",
                    symbol,
                    ", ".join(remaining),
                )
                return False, reversed_position
            return True, reversed_position

    def _entry_runtime_blocked(self, candle: dict, scope: str) -> bool:
        symbol = candle["symbol"]
        if self._account_refresh_failures:
            reason = (
                "account snapshot unavailable; new entries paused until "
                "the next successful refresh"
            )
            logger.warning(f"Skipping {symbol} entry: {reason}")
            self.audit_store.safe_record_event(
                "entry_blocked_account_stale",
                reason,
                severity="warning",
                symbol=symbol,
                mode=self.mode,
                payload={
                    "timeframe": candle["timeframe"],
                    "consecutive_failures": self._account_refresh_failures,
                },
            )
            return True

        if scope in settings.disabled_strategy_scopes_set:
            if scope not in self._disabled_scope_log:
                logger.warning(
                    f"Strategy scope {scope} is disabled by performance review"
                )
                self.audit_store.safe_record_event(
                    "strategy_scope_disabled",
                    f"Strategy scope {scope} skipped",
                    severity="warning",
                    symbol=symbol,
                    mode=self.mode,
                    payload={
                        "timeframe": candle["timeframe"],
                        "scope": scope,
                    },
                )
                self._disabled_scope_log.add(scope)
            return True
        return False

    async def _handle_telegram_command(
        self, command: str, arguments: str, user_id: str, update_id: int = 0
    ) -> str:
        actor = f"telegram user {user_id or 'unknown'}"
        if update_id:
            last_update_id = int(
                self.audit_store.get_control("telegram_last_update_id", "0")
            )
            if update_id <= last_update_id:
                logger.warning(f"Ignored replayed Telegram update {update_id}")
                return ""
            self.audit_store.set_control(
                "telegram_last_update_id", str(update_id), actor
            )
        self.audit_store.safe_record_event(
            "telegram_command",
            f"Telegram command {command}",
            mode=self.mode,
            payload={
                "command": command,
                "user_id": user_id,
                "update_id": update_id,
            },
        )
        if command == "/status":
            return self._telegram_status()
        if command == "/positions":
            return self._telegram_positions()
        if command in {
            "/pause",
            "/resume",
            "/emergency_stop",
            "/clear_emergency",
        }:
            return self._apply_telegram_control(command, arguments, actor)
        if command == "/help":
            return self._telegram_help()
        return (
            "<b>Unknown Command</b>\nUse <code>/help</code> to list supported commands."
        )

    def _telegram_status(self) -> str:
        allowed, allowed_reason = self.audit_store.trading_allowed()
        equity = (
            f"{self.portfolio.account.total_equity:.2f}"
            if self.portfolio.account
            else "unavailable"
        )
        configured_scalp_streams = (
            len(settings.symbols_list)
            * len({"1m", "3m"} & set(settings.timeframes_list))
            if settings.scalp_streaming_enabled
            else 0
        )
        healthy_scalp_streams = sum(
            self._scalp_stream_is_healthy(key, time.monotonic())
            for key in self._scalp_stream_heartbeat
        )
        return (
            "<b>Bot Status</b>\n"
            f"Environment: {html.escape(settings.binance_environment)}\n"
            f"Trading: {'ENABLED' if allowed else 'BLOCKED'}\n"
            f"Reason: {html.escape(allowed_reason)}\n"
            f"Equity: {equity}\n"
            f"Mode: TA-only\n"
            f"Scalp feed: "
            f"{'WebSocket' if settings.scalp_streaming_enabled else 'REST fast lane'}"
            f" ({healthy_scalp_streams}/{configured_scalp_streams} streams)\n"
            f"Open trades: {len(self.pos_mgr.open_trades)}\n"
            f"Alert queue: {self.alerter.pending_messages}"
        )

    def _telegram_positions(self) -> str:
        if not self.pos_mgr.open_trades:
            return "<b>Open Positions</b>\nNone"
        lines = ["<b>Open Positions</b>"]
        for key, trade in self.pos_mgr.open_trades.items():
            lines.append(
                f"{html.escape(key)}: {html.escape(trade.side.upper())} "
                f"qty={trade.quantity:.6f} entry={trade.entry_price:.2f}"
            )
        return "\n".join(lines)

    def _apply_telegram_control(self, command: str, arguments: str, actor: str) -> str:
        reason = arguments or actor
        if command == "/pause":
            self.audit_store.pause_trading(reason)
            return (
                "<b>Trading Paused</b>\n"
                f"Reason: {html.escape(reason)}\n"
                "Existing positions retain their protective orders."
            )
        if command == "/resume":
            self.audit_store.resume_trading(reason)
            return "<b>Trading Resumed</b>"
        if command == "/emergency_stop":
            self.audit_store.activate_emergency_stop(reason)
            return (
                "<b>Emergency Stop Activated</b>\n"
                "New entries are blocked. Existing positions retain their "
                "protective orders."
            )
        if arguments != "CONFIRM":
            return (
                "<b>Confirmation Required</b>\n"
                "Use <code>/clear_emergency CONFIRM</code>."
            )
        self.audit_store.clear_emergency_stop(actor)
        return "<b>Emergency Stop Cleared</b>"

    @staticmethod
    def _telegram_help() -> str:
        return (
            "<b>Bot Commands</b>\n"
            "<code>/status</code> - health and trading state\n"
            "<code>/positions</code> - audited open positions\n"
            "<code>/pause [reason]</code> - block new entries\n"
            "<code>/resume [reason]</code> - remove manual pause\n"
            "<code>/emergency_stop [reason]</code> - emergency block\n"
            "<code>/clear_emergency CONFIRM</code> - clear emergency block"
        )

    # Shut down streams, close positions, close connection
    async def stop(  # noqa: C901
        self,
        reason: str = "normal shutdown",
        *,
        close_positions: bool | None = None,
    ):
        if self._stopped:
            return
        self._stopped = True
        await self._stop_heartbeat_publisher()
        # Must precede the "stopping" write below: a surviving bootstrap
        # thread would overwrite that record with a stale "bootstrapping"
        # one, and the supervisor reads exactly that record's reason field
        # to decide whether the child exited fatally.
        self._stop_bootstrap_heartbeat()
        await self._emit_heartbeat("stopping", mode=self.mode, reason=reason)
        logger.info("Stopping trading loop...")
        for task in self._scalp_stream_tasks:
            task.cancel()
        for task in self._scalp_stream_tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._scalp_stream_tasks.clear()
        should_close_positions = (
            reason == "fatal error" if close_positions is None else close_positions
        )
        try:
            if should_close_positions:
                # TradeExecutor.close_all() now catches per-symbol failures
                # internally and returns False instead of raising (it also
                # activates its own emergency stop) - the except clause
                # below is kept as a defensive fallback for anything that
                # still escapes (e.g. from PositionManager itself), but the
                # bool return is the path that actually fires in practice
                # now and must not be silently discarded.
                try:
                    closed = await self.pos_mgr.close_all()
                except Exception as exc:
                    closed = False
                    logger.critical(
                        "Could not close positions during shutdown; preserving "
                        "protected positions for restart reconciliation: {}",
                        self._describe_exception(exc),
                    )
                    self.audit_store.safe_record_event(
                        "shutdown_close_all_failed",
                        "Could not close all positions during shutdown",
                        severity="critical",
                        mode=self.mode,
                        payload={
                            "reason": reason,
                            "positions": list(self.pos_mgr.open_trades),
                            "error": self._describe_exception(exc),
                        },
                    )
                if not closed:
                    logger.critical(
                        "Could not close all positions during shutdown; "
                        "preserving protected positions for restart "
                        "reconciliation"
                    )
                    self.audit_store.safe_record_event(
                        "shutdown_close_all_failed",
                        "Could not close all positions during shutdown",
                        severity="critical",
                        mode=self.mode,
                        payload={
                            "reason": reason,
                            "positions": list(self.pos_mgr.open_trades),
                        },
                    )
            elif self.pos_mgr.open_trades:
                self.audit_store.safe_record_event(
                    "positions_preserved_on_shutdown",
                    "Protected positions preserved for restart reconciliation",
                    severity="warning",
                    mode=self.mode,
                    payload={
                        "reason": reason,
                        "positions": list(self.pos_mgr.open_trades),
                    },
                )
                logger.warning(
                    f"Preserving {len(self.pos_mgr.open_trades)} protected "
                    "position(s) on graceful shutdown"
                )
        finally:
            try:
                await self.client.close()
            finally:
                await self.alerter.shutdown_alert(
                    self.mode, settings.binance_environment, reason
                )
                await self.alerter.stop()
                await self._emit_heartbeat("stopped", mode=self.mode, reason=reason)
