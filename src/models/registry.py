import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelPromotion:
    symbol: str
    timeframe: str
    promoted_at: str
    validation: dict
    previous_available: bool


class ModelArtifactRegistry:
    def __init__(self, model_dir: str) -> None:
        self.root = Path(model_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def candidate_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"xgb_{symbol.upper()}_{timeframe}.candidate.json"

    def active_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"xgb_{symbol.upper()}_{timeframe}.json"

    def promote(
        self,
        symbol: str,
        timeframe: str,
        *,
        validation: dict,
    ) -> ModelPromotion:
        if not validation.get("approved"):
            raise ValueError("candidate validation has not been approved")
        candidate = self.candidate_path(symbol, timeframe)
        candidate_meta = self._metadata_path(candidate)
        if not candidate.exists() or not candidate_meta.exists():
            raise FileNotFoundError("candidate model or metadata is missing")
        active = self.active_path(symbol, timeframe)
        active_meta = self._metadata_path(active)
        previous = self._previous_path(active)
        previous_meta = self._metadata_path(previous)
        had_previous = active.exists() and active_meta.exists()
        if had_previous:
            shutil.copy2(active, previous)
            shutil.copy2(active_meta, previous_meta)
        try:
            candidate.replace(active)
            candidate_meta.replace(active_meta)
        except Exception:
            if had_previous:
                shutil.copy2(previous, active)
                shutil.copy2(previous_meta, active_meta)
            raise
        promotion = ModelPromotion(
            symbol=symbol.upper(),
            timeframe=timeframe,
            promoted_at=datetime.now(timezone.utc).isoformat(),
            validation=validation,
            previous_available=had_previous,
        )
        self._record_path(active).write_text(
            json.dumps(promotion.__dict__, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return promotion

    def rollback(self, symbol: str, timeframe: str) -> None:
        active = self.active_path(symbol, timeframe)
        previous = self._previous_path(active)
        previous_meta = self._metadata_path(previous)
        if not previous.exists() or not previous_meta.exists():
            raise FileNotFoundError("previous model is unavailable for rollback")
        shutil.copy2(previous, active)
        shutil.copy2(previous_meta, self._metadata_path(active))

    @staticmethod
    def _metadata_path(model_path: Path) -> Path:
        return model_path.with_name(f"{model_path.stem}.meta.json")

    @staticmethod
    def _previous_path(active_path: Path) -> Path:
        return active_path.with_name(f"{active_path.stem}.previous.json")

    @staticmethod
    def _record_path(active_path: Path) -> Path:
        return active_path.with_name(f"{active_path.stem}.promotion.json")
