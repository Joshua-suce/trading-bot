# Unified trading loop for Binance Demo Trading and mainnet.
import asyncio
import html
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from src.audit import AuditStore
from src.config import settings
from src.exchange.account import get_account_info
from src.exchange.client import ExchangeClient
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.live.data_quality import OHLCVQualityValidator, timeframe_seconds
from src.models.classifier import XGBoostClassifier
from src.models.ensemble import ModelEnsemble
from src.monitoring.alerter import Alerter
from src.risk.portfolio import PortfolioManager
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager
from src.security import redact_text
from src.signals.aggregator import SignalAggregator


class LiveTradingLoop:
    # Initialise all components with optional pre-trained ensemble
    def __init__(
        self,
        ensemble: Optional[ModelEnsemble] = None,
        *,
        audit_store: AuditStore | None = None,
    ):
        self.mode = "trade"
        self.audit_store = audit_store or AuditStore()
        self.client = ExchangeClient()
        self.order_mgr = OrderManager(self.client, audit_store=self.audit_store)
        self.portfolio = PortfolioManager()
        self.sizer = PositionSizer(self.portfolio)
        self.sl_mgr = StopLossManager()
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
        self.ensemble = ensemble or ModelEnsemble()
        self.aggregator = SignalAggregator(self.ensemble)
        self._scoped_aggregators = (
            {} if ensemble is not None else self._load_scoped_aggregators()
        )
        self.data_quality = OHLCVQualityValidator()
        self._last_processed_candles: dict[str, object] = {}
        self._next_scan_due: dict[str, float] = {}
        self._account_refresh_failures = 0
        self._account_degradation_alerted = False
        self._last_account_refresh_at = 0.0
        self._account_refresh_interval_seconds = (
            settings.account_refresh_interval_seconds
        )
        self._last_reconciliation_at = 0.0
        self._reconciliation_interval_seconds = settings.reconciliation_interval_seconds
        self._stopped = False

    # Connect, fetch account, then scan every configured timeframe in sequence
    async def start(self):
        try:
            await self.alerter.start(self._handle_telegram_command)
            await self.alerter.initializing_alert(
                self.mode, settings.binance_environment
            )
            await self.client.connect()
            logger.info(
                "Starting trading loop on Binance "
                f"{settings.binance_environment.upper()}"
            )
            self.audit_store.safe_record_event(
                "bot_started",
                f"Trading loop started in {self.mode} mode",
                mode=self.mode,
                payload={
                    "symbols": settings.symbols_list,
                    "timeframes": settings.timeframes_list,
                    "ml_model_ready": bool(self._scoped_aggregators)
                    or self.ensemble.is_ready(),
                    "ml_model_scopes": sorted(self._scoped_aggregators),
                },
            )

            self._restore_persisted_risk_state()
            await self._refresh_account(required=True)
            if self.portfolio.account:
                logger.info(
                    f"Account equity: {self.portfolio.account.total_equity:.2f}"
                )
            self.pos_mgr.restore_open_trades_from_audit()
            self.pos_mgr.restore_recent_exit_cooldowns()
            reconciled = await self.pos_mgr.reconcile_exchange_state()
            if reconciled is None:
                await self.stop(
                    "startup reconciliation unavailable",
                    close_positions=False,
                )
                return
            if reconciled is False:
                raise RuntimeError(
                    "Startup reconciliation failed; emergency stop activated"
                )
            self._last_reconciliation_at = time.monotonic()

            for symbol in settings.symbols_list:
                await self.client.set_leverage(symbol, settings.max_leverage)

            await self.alerter.startup_alert(
                self.mode,
                settings.binance_environment,
                settings.symbols_list,
            )

            while True:
                await self._refresh_account_if_due()
                await self._reconcile_if_due()
                await self._scan_due_timeframes_once()
                await asyncio.sleep(settings.scan_sleep_seconds)
        except asyncio.CancelledError:
            await self.stop("cancelled")
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

    async def _scan_timeframes_once(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
    ):
        scan_symbols = symbols or settings.symbols_list
        scan_timeframes = timeframes or settings.timeframes_list
        for symbol in scan_symbols:
            for timeframe in scan_timeframes:
                await self._process_timeframe(symbol, timeframe)

    async def _scan_due_timeframes_once(self):
        now = time.monotonic()
        for symbol, timeframe in self._due_scan_pairs(now):
            await self._process_timeframe(symbol, timeframe)
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
                if now >= self._next_scan_due.get(key, 0.0):
                    due.append((symbol, timeframe))
        return due

    async def _process_timeframe(self, symbol: str, timeframe: str):
        try:
            from src.indicators.compute import compute_all_indicators

            limit = settings.min_ohlcv_candles + 1
            df = await self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
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
            df_ind = compute_all_indicators(closed_df)
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
            }
            await self._on_candle(candle, df_ind=df_ind)
        except Exception as e:
            error = self._describe_exception(e)
            logger.error(f"Timeframe scan error for {symbol} {timeframe}: {error}")
            await self.alerter.trade_failed_alert(self.mode, symbol, error)

    async def _refresh_account(self, required: bool = False):
        self._last_account_refresh_at = time.monotonic()
        try:
            account = await get_account_info(self.client)
        except Exception as e:
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
        reconciled = await self.pos_mgr.reconcile_exchange_state()
        self._last_reconciliation_at = time.monotonic()
        if reconciled is None:
            logger.warning(
                "Runtime reconciliation was inconclusive; keeping protected "
                "positions and retrying at the next interval"
            )
            return
        if reconciled is False:
            raise RuntimeError("Runtime reconciliation failed; emergency stop active")

    @staticmethod
    def _scan_key(symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        return timeframe_seconds(timeframe)

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
        delay = next_boundary - epoch_now + settings.candle_close_grace_seconds
        return monotonic_now + max(delay, settings.scan_sleep_seconds)

    @staticmethod
    def _load_scoped_aggregators() -> dict[str, SignalAggregator]:
        model_dir = Path(settings.model_dir).expanduser()
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parents[2] / model_dir
        aggregators: dict[str, SignalAggregator] = {}
        for symbol in settings.symbols_list:
            for timeframe in settings.timeframes_list:
                model_path = model_dir / f"xgb_{symbol}_{timeframe}.json"
                if not model_path.exists():
                    continue
                try:
                    model = XGBoostClassifier()
                    model.load(str(model_path))
                    expected_scope = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                    }
                    if model.metadata != expected_scope:
                        raise ValueError(
                            f"scope metadata {model.metadata} does not match "
                            f"{expected_scope}"
                        )
                except Exception as exc:
                    logger.error(
                        "Could not load XGBoost model from {}: {}",
                        model_path,
                        redact_text(exc),
                    )
                    continue
                key = LiveTradingLoop._model_scope_key(symbol, timeframe)
                aggregators[key] = SignalAggregator(ModelEnsemble(xgb_model=model))
                logger.info(
                    "Loaded XGBoost trading model for {} {} from {}",
                    symbol,
                    timeframe,
                    model_path,
                )
        if not aggregators:
            logger.warning(
                "No compatible scoped XGBoost models found in {}; "
                "TA-only signals are active",
                model_dir,
            )
        return aggregators

    @staticmethod
    def _model_scope_key(symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}:{timeframe}"

    def _aggregator_for(self, symbol: str, timeframe: str) -> SignalAggregator:
        return self._scoped_aggregators.get(
            self._model_scope_key(symbol, timeframe),
            self.aggregator,
        )

    # Called on each new closed candle.
    async def _on_candle(self, candle: dict, df_ind=None):
        logger.info(
            f"Candle: {candle['symbol']} {candle['timeframe']} "
            f"close={candle['close']:.2f}"
        )
        await self._execute_trade(candle, df_ind=df_ind)

    # Generate signal and enter position if criteria are met
    async def _execute_trade(self, candle: dict, df_ind=None):
        symbol = candle["symbol"]
        logger.debug(
            f"Evaluating {symbol} {candle['timeframe']} at {candle['close']:.2f}"
        )

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
            return

        # Only one leg per symbol/timeframe; other timeframes may pyramid.
        position_key = self.pos_mgr.position_key(symbol, candle["timeframe"])
        if position_key in self.pos_mgr.open_trades:
            logger.debug(
                f"Already in position for {symbol} {candle['timeframe']}; "
                "skipping new entry"
            )
            return

        # Generate signal
        try:
            if df_ind is None:
                from src.indicators.compute import compute_all_indicators

                df = await self.client.fetch_ohlcv(
                    symbol, candle["timeframe"], limit=200
                )
                df_ind = compute_all_indicators(df)
            signal = self._aggregator_for(
                symbol,
                candle["timeframe"],
            ).generate(df_ind)

            if signal.direction == 0 or signal.confidence < 0.4:
                logger.debug(
                    f"Signal skipped for {symbol}: direction={signal.direction} "
                    f"confidence={signal.confidence:.2f} ta={signal.ta_source}"
                )
                return

            atr = (
                df_ind["atr"].iloc[-1]
                if "atr" in df_ind.columns
                else df_ind["close"].iloc[-1] * 0.01
            )
            if atr <= 0 or atr != atr:
                atr = df_ind["close"].iloc[-1] * 0.01

            if signal.direction == 1:
                await self.pos_mgr.enter_long(
                    symbol,
                    candle["close"],
                    atr,
                    settings.max_leverage,
                    timeframe=candle["timeframe"],
                )
            elif signal.direction == -1:
                await self.pos_mgr.enter_short(
                    symbol,
                    candle["close"],
                    atr,
                    settings.max_leverage,
                    timeframe=candle["timeframe"],
                )

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
                    "timeframe": candle["timeframe"],
                },
            )
            await self.alerter.trade_failed_alert(self.mode, symbol, error)

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
            "<b>Unknown Command</b>\n"
            "Use <code>/help</code> to list supported commands."
        )

    def _telegram_status(self) -> str:
        allowed, allowed_reason = self.audit_store.trading_allowed()
        equity = (
            f"{self.portfolio.account.total_equity:.2f}"
            if self.portfolio.account
            else "unavailable"
        )
        return (
            "<b>Bot Status</b>\n"
            f"Environment: {html.escape(settings.binance_environment)}\n"
            f"Trading: {'ENABLED' if allowed else 'BLOCKED'}\n"
            f"Reason: {html.escape(allowed_reason)}\n"
            f"Equity: {equity}\n"
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
    async def stop(
        self,
        reason: str = "normal shutdown",
        *,
        close_positions: bool | None = None,
    ):
        if self._stopped:
            return
        self._stopped = True
        logger.info("Stopping trading loop...")
        should_close_positions = (
            reason == "fatal error" if close_positions is None else close_positions
        )
        try:
            if should_close_positions:
                await self.pos_mgr.close_all()
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
