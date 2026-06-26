import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.backtest.metrics import BacktestMetrics
from src.backtest.walk_forward import WalkForwardReport
from src.config import Settings, settings


@dataclass(frozen=True)
class StrategyApproval:
    approved: bool
    approved_at: str
    approved_by: str
    reason: str
    symbol: str
    timeframe: str
    metrics: dict[str, Any]
    validation: dict[str, Any]

    @property
    def approved_datetime(self) -> datetime:
        parsed = datetime.fromisoformat(self.approved_at)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed


class StrategyApprovalStore:
    def __init__(self, path: str | None = None, cfg: Settings = settings) -> None:
        self.path = Path(path or cfg.strategy_approval_path)
        self.cfg = cfg

    def write(
        self,
        *,
        metrics: BacktestMetrics,
        symbol: str,
        timeframe: str,
        approved_by: str,
        reason: str,
        validation: dict[str, Any] | None = None,
    ) -> StrategyApproval:
        approval = StrategyApproval(
            approved=True,
            approved_at=datetime.now(timezone.utc).isoformat(),
            approved_by=approved_by,
            reason=reason,
            symbol=symbol,
            timeframe=timeframe,
            metrics=asdict(metrics),
            validation=validation or {},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(approval), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return approval

    def write_walk_forward(
        self,
        *,
        report: WalkForwardReport,
        approved_by: str,
        reason: str,
    ) -> StrategyApproval:
        if not report.approved:
            raise ValueError(f"walk-forward report is not approved: {report.reason}")
        return self.write(
            metrics=report.aggregate_metrics,
            symbol=report.symbol,
            timeframe=report.timeframe,
            approved_by=approved_by,
            reason=reason,
            validation={"method": "walk_forward", **report.to_dict()},
        )

    def revoke(self, reason: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "approved": False,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approved_by": "system",
            "reason": reason,
            "symbol": "",
            "timeframe": "",
            "metrics": {},
            "validation": {},
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> StrategyApproval | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return StrategyApproval(
            approved=bool(data.get("approved")),
            approved_at=str(data.get("approved_at", "")),
            approved_by=str(data.get("approved_by", "")),
            reason=str(data.get("reason", "")),
            symbol=str(data.get("symbol", "")),
            timeframe=str(data.get("timeframe", "")),
            metrics=dict(data.get("metrics", {})),
            validation=dict(data.get("validation", {})),
        )

    def validate_for_mainnet(self) -> tuple[bool, str]:
        if not self.cfg.require_strategy_approval:
            return True, "strategy approval not required"
        approval = self.load()
        if approval is None:
            return False, f"missing strategy approval: {self.path}"
        if not approval.approved:
            return False, f"strategy approval revoked: {approval.reason}"

        age_seconds = (
            datetime.now(timezone.utc) - approval.approved_datetime
        ).total_seconds()
        max_age_seconds = self.cfg.strategy_approval_max_age_hours * 3600
        if age_seconds > max_age_seconds:
            return False, "strategy approval expired"

        metrics = approval.metrics
        total_trades = int(metrics.get("total_trades", 0))
        profit_factor = float(metrics.get("profit_factor", 0.0))
        drawdown_pct = float(metrics.get("max_drawdown_pct", 100.0))
        if total_trades < self.cfg.min_approval_trades:
            return False, f"approval has too few trades: {total_trades}"
        if profit_factor < self.cfg.min_approval_profit_factor:
            return False, f"profit factor below approval threshold: {profit_factor}"
        if drawdown_pct > self.cfg.max_approval_drawdown_pct:
            return False, f"drawdown above approval threshold: {drawdown_pct}"
        if self.cfg.require_walk_forward_approval:
            valid, reason = self._validate_walk_forward(approval.validation)
            if not valid:
                return valid, reason
        return True, "strategy approval valid"

    def _validate_walk_forward(
        self,
        validation: dict[str, Any],
    ) -> tuple[bool, str]:
        if validation.get("method") != "walk_forward":
            return False, "approval is missing walk-forward validation"
        if not validation.get("approved"):
            return False, "walk-forward validation did not pass"
        folds = validation.get("folds") or []
        if len(folds) < self.cfg.min_approval_oos_folds:
            return False, f"approval has too few out-of-sample folds: {len(folds)}"
        profitable_ratio = float(validation.get("profitable_fold_ratio", 0.0))
        if profitable_ratio < self.cfg.min_approval_profitable_fold_ratio:
            return False, "profitable fold ratio below approval threshold"
        return True, "walk-forward validation valid"
