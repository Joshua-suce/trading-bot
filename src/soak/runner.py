import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd

from src.audit import AuditStore
from src.exchange.account import AccountInfo
from src.exchange.client import ExchangeClient
from src.live.loop import LiveTradingLoop
from src.monitoring.alerter import Alerter
from src.signals.aggregator import FinalSignal, SignalAggregator


class SoakAlerter:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def startup_alert(self, mode: str, environment: str, symbols: list[str]):
        self.messages.append(
            {"type": "startup", "mode": mode, "environment": environment}
        )

    async def trade_opened_alert(self, *args):
        self.messages.append({"type": "trade_opened", "args": args})

    async def trade_completed_alert(self, *args):
        self.messages.append({"type": "trade_completed", "args": args})

    async def trade_failed_alert(self, mode: str, symbol: str, reason: str):
        self.messages.append(
            {"type": "trade_failed", "mode": mode, "symbol": symbol, "reason": reason}
        )

    async def error_alert(self, error: str):
        self.messages.append({"type": "error", "error": error})


class SoakClient:
    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        periods = max(limit, 60)
        index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"), periods=periods, freq="5min"
        )
        return pd.DataFrame(
            {
                "open": [100.0 + i * 0.05 for i in range(periods)],
                "high": [101.0 + i * 0.05 for i in range(periods)],
                "low": [99.0 + i * 0.05 for i in range(periods)],
                "close": [100.5 + i * 0.05 for i in range(periods)],
                "volume": [10.0 + i for i in range(periods)],
            },
            index=index,
        )


class SoakAggregator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, _df) -> FinalSignal:
        self.calls += 1
        direction = 1 if self.calls == 1 else 0
        confidence = 0.9 if direction else 0.0
        return FinalSignal(
            direction=direction,
            confidence=confidence,
            ta_source="soak",
            ml_strength=0.0,
            ml_confidence=0.0,
        )


@dataclass(frozen=True)
class SoakReport:
    status: str
    iterations: int
    opened_trades: int
    completed_trades: int
    failed_trades: int
    audit_events: int
    open_trades_after_shutdown: int
    report_path: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class SoakRunner:
    def __init__(
        self,
        *,
        audit_path: str,
        report_path: str,
        iterations: int = 2,
    ) -> None:
        self.audit_store = AuditStore(audit_path)
        self.report_path = Path(report_path)
        self.iterations = iterations

    async def run(self) -> SoakReport:
        bot = LiveTradingLoop(mode="paper")
        bot.audit_store = self.audit_store
        soak_alerter = SoakAlerter()
        bot.client = cast(ExchangeClient, SoakClient())
        bot.alerter = cast(Alerter, soak_alerter)
        bot.aggregator = cast(SignalAggregator, SoakAggregator())
        bot.order_mgr.audit_store = self.audit_store
        bot.pos_mgr.audit_store = self.audit_store
        bot.pos_mgr.alerter = bot.alerter
        bot.portfolio.update_account(
            AccountInfo(
                total_equity=10_000.0,
                wallet_balance=10_000.0,
                available_balance=10_000.0,
                unrealized_pnl=0.0,
                margin_ratio=0.0,
            )
        )

        await bot.alerter.startup_alert("paper", "soak", ["BTCUSDT"])
        self.audit_store.record_event(
            "soak_started",
            "Offline paper soak started",
            mode="paper",
            payload={"iterations": self.iterations},
        )
        for _ in range(self.iterations):
            await bot._scan_timeframes_once(symbols=["BTCUSDT"], timeframes=["5m"])

        await bot.stop()
        self.audit_store.record_event(
            "soak_completed",
            "Offline paper soak completed",
            mode="paper",
        )
        return self._build_report(bot, soak_alerter)

    def _build_report(self, bot: LiveTradingLoop, alerter: SoakAlerter) -> SoakReport:
        events = self.audit_store.load_recent_events(500)
        alert_types = [message["type"] for message in alerter.messages]
        report = SoakReport(
            status="ok" if not self.audit_store.load_open_trades("paper") else "failed",
            iterations=self.iterations,
            opened_trades=alert_types.count("trade_opened"),
            completed_trades=alert_types.count("trade_completed"),
            failed_trades=alert_types.count("trade_failed"),
            audit_events=len(events),
            open_trades_after_shutdown=len(self.audit_store.load_open_trades("paper")),
            report_path=str(self.report_path),
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report
