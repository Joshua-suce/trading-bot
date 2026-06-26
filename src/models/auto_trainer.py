import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from src.audit import AuditStore
from src.config import settings
from src.exchange.client import ExchangeClient
from src.models.classifier import XGBoostClassifier
from src.models.feature_engineer import LABEL_SCHEMA
from src.models.trainer import ModelTrainer
from src.security import redact_text

ModelReadyCallback = Callable[[str, str, XGBoostClassifier], None]


class AutomaticModelTrainer:
    """Sequential background retraining with failure-isolated model activation."""

    def __init__(
        self,
        client: ExchangeClient,
        audit_store: AuditStore,
        on_model_ready: ModelReadyCallback,
    ):
        self.client = client
        self.audit_store = audit_store
        self.on_model_ready = on_model_ready
        self._stop_event = asyncio.Event()
        self._training_lock = asyncio.Lock()

    async def run_forever(self) -> None:
        if settings.auto_retrain_on_startup:
            await self._run_cycle_safely()

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=settings.auto_retrain_check_interval_seconds,
                )
            except TimeoutError:
                await self._run_cycle_safely()

    async def stop(self) -> None:
        self._stop_event.set()

    async def _run_cycle_safely(self) -> None:
        try:
            counts = await self.run_due_cycle()
            if counts["trained"] or counts["failed"]:
                logger.info(
                    "Automatic ML retraining cycle complete: "
                    "trained={} failed={} skipped={}",
                    counts["trained"],
                    counts["failed"],
                    counts["skipped"],
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Automatic ML retraining cycle failed unexpectedly: {}",
                redact_text(exc),
            )
            self.audit_store.safe_record_event(
                "ml_retraining_cycle_failed",
                "Automatic retraining scheduler failed; it will retry",
                severity="error",
                mode="trade",
                payload={"error": redact_text(exc)},
            )

    async def run_due_cycle(self) -> dict[str, int]:
        if self._training_lock.locked():
            return {"trained": 0, "failed": 0, "skipped": 0}

        counts = {"trained": 0, "failed": 0, "skipped": 0}
        async with self._training_lock:
            for symbol in settings.symbols_list:
                for timeframe in settings.timeframes_list:
                    if self._stop_event.is_set():
                        return counts
                    scope = f"{symbol}:{timeframe}"
                    if scope in settings.disabled_strategy_scopes_set:
                        counts["skipped"] += 1
                        continue
                    if not self._model_is_due(symbol, timeframe):
                        counts["skipped"] += 1
                        continue
                    if await self._train_scope(symbol, timeframe):
                        counts["trained"] += 1
                    else:
                        counts["failed"] += 1
                    if settings.auto_retrain_scope_delay_seconds:
                        await self._sleep_or_stop(
                            settings.auto_retrain_scope_delay_seconds
                        )
        return counts

    async def _train_scope(self, symbol: str, timeframe: str) -> bool:
        started = time.monotonic()
        logger.info("Automatic ML retraining started for {} {}", symbol, timeframe)
        self.audit_store.safe_record_event(
            "ml_retraining_started",
            "Automatic model retraining started",
            symbol=symbol,
            mode="trade",
            payload={"timeframe": timeframe},
        )
        try:
            df = await self.client.fetch_ohlcv(
                symbol,
                timeframe,
                limit=settings.auto_retrain_candle_limit,
            )
            if len(df) < self._minimum_training_rows():
                raise ValueError(
                    f"insufficient candles: {len(df)} < "
                    f"{self._minimum_training_rows()}"
                )

            # Binance's final row can still be forming. Training only sees closed data.
            closed_df = df.iloc[:-1].copy()
            trainer = ModelTrainer()
            ensemble = await asyncio.to_thread(
                trainer.train_ensemble,
                closed_df,
                symbol=symbol,
                timeframe=timeframe,
            )
            model = ensemble.xgb
            if model is None or not model.is_trained:
                raise RuntimeError("training did not produce a usable XGBoost model")

            self.on_model_ready(symbol, timeframe, model)
            elapsed = time.monotonic() - started
            self.audit_store.safe_record_event(
                "ml_retraining_completed",
                "Automatic model retraining completed",
                symbol=symbol,
                mode="trade",
                payload={
                    "timeframe": timeframe,
                    "candles": len(closed_df),
                    "duration_seconds": round(elapsed, 3),
                    "classes": model.classes_.tolist(),
                    "label_schema": model.metadata.get("label_schema"),
                },
            )
            logger.info(
                "Automatic ML retraining completed for {} {} in {:.1f}s",
                symbol,
                timeframe,
                elapsed,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = redact_text(exc)
            self.audit_store.safe_record_event(
                "ml_retraining_failed",
                "Automatic model retraining failed; previous model retained",
                severity="error",
                symbol=symbol,
                mode="trade",
                payload={"timeframe": timeframe, "error": error},
            )
            logger.error(
                "Automatic ML retraining failed for {} {}: {}; "
                "previous model retained",
                symbol,
                timeframe,
                error,
            )
            return False

    def _model_is_due(self, symbol: str, timeframe: str) -> bool:
        model_path = self._model_path(symbol, timeframe)
        metadata_path = model_path.with_name(f"{model_path.stem}.meta.json")
        if not model_path.exists() or not metadata_path.exists():
            return True

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_metadata = metadata.get("metadata") or {}
            if model_metadata.get("label_schema") != LABEL_SCHEMA:
                return True
            if int(model_metadata.get("prediction_horizon", -1)) != (
                settings.prediction_horizon
            ):
                return True
            if float(model_metadata.get("label_atr_multiplier", -1)) != (
                settings.ml_label_atr_multiplier
            ):
                return True
            if float(model_metadata.get("label_min_return", -1)) != (
                settings.ml_effective_label_min_return
            ):
                return True
        except (OSError, ValueError, TypeError):
            return True

        age_seconds = max(time.time() - model_path.stat().st_mtime, 0.0)
        return age_seconds >= settings.model_update_interval_hours * 3600

    @staticmethod
    def _minimum_training_rows() -> int:
        return max(
            300,
            settings.min_ohlcv_candles
            + settings.prediction_horizon
            + settings.feature_lookback
            + 1,
        )

    @staticmethod
    def _model_path(symbol: str, timeframe: str) -> Path:
        model_dir = Path(settings.model_dir).expanduser()
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parents[2] / model_dir
        return model_dir / f"xgb_{symbol.upper()}_{timeframe}.json"

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except TimeoutError:
            return
