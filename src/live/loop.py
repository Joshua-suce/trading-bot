# Unified trading loop for Binance Demo Trading and mainnet.
import asyncio
import re
import time
from typing import Optional

from loguru import logger

from src.audit import AuditStore
from src.config import settings
from src.exchange.account import get_account_info
from src.exchange.client import ExchangeClient
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.live.data_quality import OHLCVQualityValidator, timeframe_seconds
from src.models.ensemble import ModelEnsemble
from src.monitoring.alerter import Alerter
from src.risk.portfolio import PortfolioManager
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager
from src.signals.aggregator import SignalAggregator


class LiveTradingLoop:
    # Initialise all components with optional pre-trained ensemble
    def __init__(self, ensemble: Optional[ModelEnsemble] = None):
        self.mode = "trade"
        self.audit_store = AuditStore()
        self.client = ExchangeClient()
        self.order_mgr = OrderManager(self.client, audit_store=self.audit_store)
        self.portfolio = PortfolioManager()
        self.sizer = PositionSizer(self.portfolio)
        self.sl_mgr = StopLossManager()
        self.alerter = Alerter(
            telegram_token=settings.telegram_bot_token,
            telegram_chat_id=settings.telegram_chat_id,
            discord_webhook=settings.discord_webhook_url,
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
        self.data_quality = OHLCVQualityValidator()
        self._last_processed_candles: dict[str, object] = {}
        self._next_scan_due: dict[str, float] = {}
        self._account_refresh_failures = 0
        self._last_reconciliation_at = 0.0
        self._reconciliation_interval_seconds = settings.reconciliation_interval_seconds

    # Connect, fetch account, then scan every configured timeframe in sequence
    async def start(self):
        try:
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
                },
            )

            await self._refresh_account(required=True)
            if self.portfolio.account:
                logger.info(
                    f"Account equity: {self.portfolio.account.total_equity:.2f}"
                )
            await self.alerter.startup_alert(
                self.mode,
                settings.binance_environment,
                settings.symbols_list,
            )

            self.pos_mgr.restore_open_trades_from_audit()
            reconciled = await self.pos_mgr.reconcile_exchange_state()
            if not reconciled:
                raise RuntimeError(
                    "Startup reconciliation failed; emergency stop activated"
                )
            self._last_reconciliation_at = time.monotonic()

            for symbol in settings.symbols_list:
                await self.client.set_leverage(symbol, settings.max_leverage)

            while True:
                await self._scan_due_timeframes_once()
                await self._refresh_account()
                await self._reconcile_if_due()
                await asyncio.sleep(settings.scan_sleep_seconds)
        except asyncio.CancelledError:
            await self.stop()
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            self.audit_store.safe_record_event(
                "bot_error",
                "Live loop crashed",
                severity="critical",
                mode=self.mode,
                payload={"error": str(e)},
            )
            await self.stop()

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
                now + self._timeframe_seconds(timeframe)
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
            logger.error(f"Timeframe scan error for {symbol} {timeframe}: {e}")
            await self.alerter.trade_failed_alert(self.mode, symbol, str(e))

    async def _refresh_account(self, required: bool = False):
        try:
            account = await get_account_info(self.client)
        except Exception as e:
            self._account_refresh_failures += 1
            logger.warning(
                "Account refresh failed ({} consecutive): {}",
                self._account_refresh_failures,
                self._describe_exception(e),
            )
            if required or self._account_refresh_failures >= 3:
                raise RuntimeError(
                    "Account refresh failed repeatedly in trade mode"
                ) from e
            return

        self._account_refresh_failures = 0
        self.portfolio.update_account(account)
        logger.debug(f"Account equity refreshed: {account.total_equity:.2f}")

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
        return re.sub(
            r"((?:[?&])?signature=)[^&\s]+",
            r"\1<redacted>",
            message,
        )

    async def _reconcile_if_due(self):
        elapsed = time.monotonic() - self._last_reconciliation_at
        if elapsed < self._reconciliation_interval_seconds:
            return
        reconciled = await self.pos_mgr.reconcile_exchange_state()
        self._last_reconciliation_at = time.monotonic()
        if not reconciled:
            raise RuntimeError("Runtime reconciliation failed; emergency stop active")

    @staticmethod
    def _scan_key(symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        return timeframe_seconds(timeframe)

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

        # Check if already in position
        position_key = self.pos_mgr.position_key(symbol, candle["timeframe"])
        if position_key in self.pos_mgr.open_trades:
            logger.debug(f"Already in position for {symbol}; skipping new entry")
            return

        # Generate signal
        try:
            if df_ind is None:
                from src.indicators.compute import compute_all_indicators

                df = await self.client.fetch_ohlcv(
                    symbol, candle["timeframe"], limit=200
                )
                df_ind = compute_all_indicators(df)
            signal = self.aggregator.generate(df_ind)

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
            logger.error(f"Trade execution error for {symbol}: {e}")
            await self.alerter.trade_failed_alert(self.mode, symbol, str(e))

    # Shut down streams, close positions, close connection
    async def stop(self):
        logger.info("Stopping trading loop...")
        await self.pos_mgr.close_all()
        await self.client.close()
