# Unified trading loop for Binance Demo Trading and mainnet.
import asyncio
import html
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.audit import AuditStore
from src.config import settings
from src.exchange.account import get_account_info
from src.exchange.client import BinanceDemoAccountInactiveError, ExchangeClient
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.live.data_quality import OHLCVQualityValidator, timeframe_seconds
from src.models.auto_trainer import AutomaticModelTrainer
from src.models.classifier import XGBoostClassifier
from src.models.ensemble import ModelEnsemble
from src.models.feature_engineer import LABEL_SCHEMA
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
from src.strategies import StrategyRegistry


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
        self.ensemble = ensemble or ModelEnsemble()
        if settings.force_ta_only:
            self.aggregator = SignalAggregator(ModelEnsemble())
            self._scoped_aggregators = {}
        else:
            ml_callbacks = self._make_ml_callbacks()
            self.aggregator = SignalAggregator(
                self.ensemble,
                ta_weight=settings.ta_weight,
                ml_weight=settings.ml_weight,
                require_confluence=settings.require_signal_confluence,
                on_ml_degraded=ml_callbacks["on_degraded"],
                on_ml_recovered=ml_callbacks["on_recovered"],
            )
            self._scoped_aggregators = (
                {}
                if ensemble is not None
                else self._load_scoped_aggregators(alerter=self.alerter)
            )
        self.data_quality = OHLCVQualityValidator()
        self.strategy_quality = strategy_quality_gate_from_settings()
        self.strategy_registry = StrategyRegistry(settings)
        self._market_regimes: dict[str, int] = {}
        self._last_processed_candles: dict[str, object] = {}
        self._disabled_scope_log: set[str] = set()
        self._next_scan_due: dict[str, float] = {}
        self._account_refresh_failures = 0
        self._account_degradation_alerted = False
        self._last_account_refresh_at = 0.0
        self._account_refresh_interval_seconds = (
            settings.account_refresh_interval_seconds
        )
        self._last_reconciliation_at = 0.0
        self._reconciliation_interval_seconds = settings.reconciliation_interval_seconds
        self._auto_trainer: AutomaticModelTrainer | None = None
        self._auto_trainer_task: asyncio.Task | None = None
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
        self._stopped = False
        self._heartbeat = RuntimeHeartbeat(settings.runtime_heartbeat_path)
        self._heartbeat_task: asyncio.Task | None = None

    # Connect, fetch account, then scan every configured timeframe in sequence
    async def start(self):
        try:
            await self.alerter.start(self._handle_telegram_command)
            await self.alerter.initializing_alert(
                self.mode, settings.binance_environment
            )
            await self.client.connect()
            self._write_heartbeat(
                "starting",
                mode=self.mode,
                environment=settings.binance_environment,
            )
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
            self.pos_mgr.trades.restore_open_trades_from_audit()
            self.pos_mgr.trades.restore_recent_exit_cooldowns()
            reconciled = await self.pos_mgr.reconcile_exchange_state(
                auto_close_unmanaged=True
            )
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
            interrupted_requests = (
                self.audit_store.fail_interrupted_manual_trade_requests()
            )
            if interrupted_requests:
                logger.critical(
                    "{} interrupted dashboard trade request(s) require review",
                    interrupted_requests,
                )
            self._last_reconciliation_at = time.monotonic()

            await self._set_configured_leverage()

            await self.alerter.startup_alert(
                self.mode,
                settings.binance_environment,
                settings.symbols_list,
            )
            self._start_automatic_retraining()
            self._start_scalp_streams()
            self._start_heartbeat_publisher()

            while True:
                await self._refresh_account_if_due()
                await self._reconcile_if_due()
                await self._manage_scalp_positions_if_due()
                await self._process_manual_trade_requests()
                await self._scan_due_timeframes_once()
                self._write_heartbeat(
                    "running",
                    mode=self.mode,
                    environment=settings.binance_environment,
                    open_positions=len(self.pos_mgr.open_trades),
                )
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

    def _write_heartbeat(self, state: str, **details: object) -> None:
        try:
            self._heartbeat.write(state, **details)
        except Exception as exc:
            logger.warning(
                "Runtime heartbeat update failed (state={}): {}",
                state,
                self._describe_exception(exc),
            )

    def _start_heartbeat_publisher(self) -> None:
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_publisher())

    async def _heartbeat_publisher(self) -> None:
        interval = max(
            1.0,
            min(15.0, settings.supervisor_heartbeat_stale_seconds / 3),
        )
        while not self._stopped:
            try:
                self._write_heartbeat(
                    "running",
                    mode=self.mode,
                    environment=settings.binance_environment,
                    open_positions=len(self.pos_mgr.open_trades),
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
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
        if scope in settings.disabled_strategy_scopes_set:
            return {
                "success": False,
                "reason": "strategy scope is disabled by performance review",
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
        indicators = compute_all_indicators(closed_df)
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
            context_indicators = compute_all_indicators(context_df.iloc[:-1].copy())
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
        await asyncio.gather(
            *(
                self._bounded_process_timeframe(symbol, timeframe)
                for symbol, timeframe in pairs
            )
        )
        for symbol, timeframe in pairs:
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
    ) -> None:
        async with self._market_data_semaphore:
            await self._process_timeframe(symbol, timeframe)

    async def _process_timeframe(self, symbol: str, timeframe: str):
        try:
            df = await self._fetch_rest_market_frame(symbol, timeframe)
            self._cache_market_frame(symbol, timeframe, df)
            await self._process_market_frame(
                symbol,
                timeframe,
                df,
                market_data_source="rest",
            )
        except Exception as e:
            await self._report_timeframe_error(symbol, timeframe, e)

    async def _fetch_rest_market_frame(
        self,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        limit = settings.min_ohlcv_candles + 1
        attempts = (
            settings.scalp_rest_freshness_attempts if timeframe in {"1m", "3m"} else 1
        )
        frame = pd.DataFrame()
        for attempt in range(1, attempts + 1):
            frame = await self.client.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit,
            )
            if timeframe not in {"1m", "3m"} or len(frame) < 2:
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
                    "REST scalp candle stale for {} {}: {:.2f}s; refetching ({}/{})",
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
        await self.alerter.data_feed_alert(
            self.mode,
            symbol,
            timeframe,
            error,
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

    async def _manage_scalp_positions_if_due(self) -> None:
        now = time.monotonic()
        if (
            now - self._last_scalp_management_at
            < settings.scalp_management_interval_seconds
        ):
            return
        self._last_scalp_management_at = now
        positions = [
            (key, trade)
            for key, trade in list(self.pos_mgr.open_trades.items())
            if trade.strategy == "scalp"
        ]
        stale_positions = [
            (key, trade)
            for key, trade in list(self.pos_mgr.open_trades.items())
            if trade.strategy != "scalp" and self._strategy_max_hold_reached(trade)
        ]
        await asyncio.gather(
            *(self._manage_scalp_position(key, trade) for key, trade in positions),
            *(
                self.pos_mgr.exit_position(key, "maximum hold")
                for key, _ in stale_positions
            ),
        )

    async def _manage_scalp_position(self, position_key: str, trade) -> None:
        try:
            if self._scalp_max_hold_reached(trade):
                await self.pos_mgr.exit_position(position_key, "scalp maximum hold")
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
        reconciled = await self.pos_mgr.reconcile_exchange_state(
            auto_close_unmanaged=settings.auto_close_unmanaged_positions
        )
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

    @staticmethod
    def _scan_key(symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        return timeframe_seconds(timeframe)

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

    @staticmethod
    def _load_scoped_aggregators(
        *,
        alerter: Alerter | None = None,
    ) -> dict[str, SignalAggregator]:
        model_dir = Path(settings.model_dir).expanduser()
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parents[2] / model_dir
        aggregators: dict[str, SignalAggregator] = {}
        for symbol in settings.symbols_list:
            for timeframe in settings.timeframes_list:
                scope = LiveTradingLoop._model_scope_key(symbol, timeframe)
                if scope in settings.disabled_strategy_scopes_set:
                    logger.info(
                        "Skipping ML model load for disabled strategy scope {}",
                        scope,
                    )
                    continue
                model_path = model_dir / f"xgb_{symbol}_{timeframe}.json"
                if not model_path.exists():
                    continue
                try:
                    model = XGBoostClassifier()
                    model.load(str(model_path))
                    if not LiveTradingLoop._model_is_compatible(
                        model,
                        symbol,
                        timeframe,
                    ):
                        continue
                except Exception as exc:
                    logger.error(
                        "Could not load XGBoost model from {}: {}",
                        model_path,
                        redact_text(exc),
                    )
                    continue
                ml_callbacks = LiveTradingLoop._make_ml_callbacks(alerter)
                aggregators[scope] = SignalAggregator(
                    ModelEnsemble(
                        xgb_model=model,
                        confidence_threshold=settings.ml_confidence_threshold,
                    ),
                    ta_weight=settings.ta_weight,
                    ml_weight=settings.ml_weight,
                    require_confluence=settings.require_signal_confluence,
                    on_ml_degraded=ml_callbacks["on_degraded"],
                    on_ml_recovered=ml_callbacks["on_recovered"],
                )
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
    def _model_is_compatible(
        model: XGBoostClassifier,
        symbol: str,
        timeframe: str,
    ) -> bool:
        expected_scope = {"symbol": symbol, "timeframe": timeframe}
        actual_scope = {key: model.metadata.get(key) for key in expected_scope}
        if actual_scope != expected_scope:
            raise ValueError(
                f"scope metadata {model.metadata} does not match {expected_scope}"
            )
        expected_training = {
            "label_schema": LABEL_SCHEMA,
            "prediction_horizon": settings.prediction_horizon,
            "label_atr_multiplier": settings.ml_label_atr_multiplier,
            "label_min_return": settings.ml_effective_label_min_return,
        }
        try:
            compatible = (
                model.metadata.get("label_schema")
                == expected_training["label_schema"]
                and int(model.metadata.get("prediction_horizon", -1))
                == expected_training["prediction_horizon"]
                and float(model.metadata.get("label_atr_multiplier", -1))
                == expected_training["label_atr_multiplier"]
                and float(model.metadata.get("label_min_return", -1))
                == expected_training["label_min_return"]
            )
        except (TypeError, ValueError):
            compatible = False
        if compatible:
            return True
        if settings.auto_retrain_enabled:
            logger.info(
                "Skipping economically incompatible ML model for {} {}; "
                "automatic retraining will replace it",
                symbol,
                timeframe,
            )
        else:
            logger.warning(
                "Skipping economically incompatible ML model for {} {}; "
                "retrain it before enabling ML predictions for this scope",
                symbol,
                timeframe,
            )
        return False

    @staticmethod
    def _make_ml_callbacks(
        alerter: Alerter | None = None,
    ) -> dict[str, Callable[[], None]]:
        def _degraded():
            if alerter:
                import asyncio

                asyncio.create_task(
                    alerter.error_alert("ML model degraded; TA-only fallback active")
                )

        def _recovered():
            if alerter:
                import asyncio

                asyncio.create_task(
                    alerter.error_alert(
                        "ML model recovered; full signal fusion restored"
                    )
                )

        return {"on_degraded": _degraded, "on_recovered": _recovered}

    @staticmethod
    def _model_scope_key(symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}:{timeframe}"

    def _aggregator_for(self, symbol: str, timeframe: str) -> SignalAggregator:
        return self._scoped_aggregators.get(
            self._model_scope_key(symbol, timeframe),
            self.aggregator,
        )

    def _start_automatic_retraining(self) -> None:
        if settings.force_ta_only or not settings.auto_retrain_enabled:
            return
        self._auto_trainer = AutomaticModelTrainer(
            self.client,
            self.audit_store,
            self._activate_retrained_model,
        )
        self._auto_trainer_task = asyncio.create_task(
            self._auto_trainer.run_forever(),
            name="automatic-model-retraining",
        )
        logger.info(
            "Automatic ML retraining enabled: refresh={}h check={}s limit={}",
            settings.model_update_interval_hours,
            settings.auto_retrain_check_interval_seconds,
            settings.auto_retrain_candle_limit,
        )

    def _activate_retrained_model(
        self,
        symbol: str,
        timeframe: str,
        model: XGBoostClassifier,
    ) -> None:
        callbacks = self._make_ml_callbacks(self.alerter)
        aggregator = SignalAggregator(
            ModelEnsemble(
                xgb_model=model,
                confidence_threshold=settings.ml_confidence_threshold,
            ),
            ta_weight=settings.ta_weight,
            ml_weight=settings.ml_weight,
            require_confluence=settings.require_signal_confluence,
            on_ml_degraded=callbacks["on_degraded"],
            on_ml_recovered=callbacks["on_recovered"],
        )
        scope = self._model_scope_key(symbol, timeframe)
        self._scoped_aggregators[scope] = aggregator
        logger.info("Activated retrained ML model for {}", scope)

    # Called on each new closed candle.
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

    @staticmethod
    def _ordered_strategies(primary: str, timeframe: str) -> list[str]:
        priority = ["trend", "breakout", "transition", "reversal", "range", "countertrend"]
        seen: set[str] = set()
        order: list[str] = []
        for s in [primary] + priority:
            if s not in seen:
                order.append(s)
                seen.add(s)
        if timeframe in {"1m", "3m"} and "scalp" not in seen:
            order.append("scalp")
        return order

    # Generate signal and enter position if criteria are met
    async def _execute_trade(self, candle: dict, df_ind=None):  # noqa: C901
        symbol = candle["symbol"]
        scope = self._model_scope_key(symbol, candle["timeframe"])
        timeframe = candle["timeframe"]
        logger.debug(
            f"Evaluating {symbol} {timeframe} at {candle['close']:.2f}"
        )

        if self._entry_runtime_blocked(candle, scope):
            return

        # Generate signal once; evaluate against all strategies independently
        try:
            if df_ind is None:
                from src.indicators.compute import compute_all_indicators

                df = await self.client.fetch_ohlcv(
                    symbol, timeframe, limit=200
                )
                df_ind = compute_all_indicators(df)
            htf_key = self._model_scope_key(symbol, "15m")
            htf_bias = self._market_regimes.get(htf_key, 0)
            signal = generate_with_context(
                self._aggregator_for(symbol, timeframe),
                df_ind,
                htf_bias,
            )

            if signal.direction == 0:
                logger.debug(f"No signal direction for {scope}")
                return

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

            strategies_to_try = self._ordered_strategies(signal.strategy, timeframe)

            for strategy_idx, strategy in enumerate(strategies_to_try):
                position_key = self.pos_mgr.trades.position_key(symbol, timeframe, strategy)
                if position_key in self.pos_mgr.open_trades:
                    logger.debug(
                        f"Already in {strategy} position for {scope}; skipping"
                    )
                    continue

                policy = self.strategy_registry.get(strategy)
                if not policy.supports(timeframe):
                    logger.debug(
                        f"{policy.name} does not support {timeframe}; skipping"
                    )
                    continue

                minimum_confidence = self._strategy_minimum_confidence(strategy)
                if signal.confidence < minimum_confidence:
                    reason = f"{strategy} confidence {signal.confidence:.4f} < {minimum_confidence:.4f}"
                    logger.debug(f"Signal skipped for {scope} {strategy}: {reason}")
                    self._record_signal_observation(
                        candle, signal,
                        minimum_confidence=minimum_confidence,
                        decision="skipped",
                        reason=reason,
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
                        reason = f"scalp signal stale: {latency:.2f}s > {latency_limit:.2f}s source={source}"
                        logger.info(f"Signal skipped for {scope}: {reason}")
                        self._record_signal_observation(
                            candle, signal,
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
                )
                if not quality.accepted:
                    logger.debug(
                        f"Strategy quality rejected {scope} {strategy}: "
                        f"score={quality.score:.2f} reason={quality.reason}"
                    )
                    self.audit_store.safe_record_event(
                        "signal_quality_rejected",
                        f"Strategy quality rejected {scope} {strategy}",
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
                        },
                    )
                    self._record_signal_observation(
                        candle, signal,
                        minimum_confidence=minimum_confidence,
                        decision="quality_rejected",
                        reason=quality.reason,
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
                        "confidence": signal.confidence,
                        "ta_source": signal.ta_source,
                        "ml_strength": signal.ml_strength,
                        "ml_confidence": signal.ml_confidence,
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
                    candle, signal,
                    minimum_confidence=minimum_confidence,
                    decision="accepted",
                    reason=signal.decision_reason or quality.reason,
                    quality=quality,
                )

                if signal.direction == 1:
                    opened = await self.pos_mgr.enter_long(
                        symbol, candle["close"], atr,
                        settings.max_leverage,
                        timeframe=timeframe,
                        signal_timestamp=candle["timestamp"],
                        strategy=strategy,
                    )
                else:
                    opened = await self.pos_mgr.enter_short(
                        symbol, candle["close"], atr,
                        settings.max_leverage,
                        timeframe=timeframe,
                        signal_timestamp=candle["timestamp"],
                        strategy=strategy,
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
                strategy=signal.strategy,
                direction=signal.direction,
                confidence=signal.confidence,
                minimum_confidence=minimum_confidence,
                decision=decision,
                reason=reason,
                signal_price=candle["close"],
                ta_source=signal.ta_source,
                ml_strength=signal.ml_strength,
                ml_confidence=signal.ml_confidence,
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
        if settings.force_ta_only:
            ml_status = "N/A (TA-only)"
        else:
            healthy = all(
                agg.ml_healthy
                for agg in [self.aggregator] + list(self._scoped_aggregators.values())
            )
            ml_status = "ok" if healthy else "degraded"
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
            f"ML: {ml_status}\n"
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
    async def stop(
        self,
        reason: str = "normal shutdown",
        *,
        close_positions: bool | None = None,
    ):
        if self._stopped:
            return
        self._stopped = True
        await self._stop_heartbeat_publisher()
        self._write_heartbeat("stopping", mode=self.mode, reason=reason)
        logger.info("Stopping trading loop...")
        for task in self._scalp_stream_tasks:
            task.cancel()
        for task in self._scalp_stream_tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._scalp_stream_tasks.clear()
        if self._auto_trainer:
            await self._auto_trainer.stop()
        if self._auto_trainer_task:
            try:
                await asyncio.wait_for(self._auto_trainer_task, timeout=30.0)
            except TimeoutError:
                logger.warning("Automatic retraining did not stop within 30 seconds")
                self._auto_trainer_task.cancel()
            except asyncio.CancelledError:
                pass
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
                self._write_heartbeat("stopped", mode=self.mode, reason=reason)
