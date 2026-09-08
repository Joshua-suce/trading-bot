import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd

from src.audit import AuditStore
from src.config import settings
from src.exchange.account import AccountInfo
from src.exchange.client import ExchangeClient
from src.execution.order_manager import OrderManager
from src.live.data_quality import timeframe_seconds
from src.live.loop import LiveTradingLoop
from src.monitoring.alerter import Alerter
from src.monitoring.heartbeat import RuntimeHeartbeat
from src.signals.aggregator import FinalSignal, SignalAggregator


class SoakAlerter:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.pending_messages = 0

    async def start(self, command_handler=None):
        return None

    async def initializing_alert(self, mode: str, environment: str):
        self.messages.append(
            {"type": "initializing", "mode": mode, "environment": environment}
        )

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

    async def data_feed_alert(
        self,
        mode: str,
        symbol: str,
        timeframe: str,
        reason: str,
    ):
        self.messages.append(
            {
                "type": "data_feed",
                "mode": mode,
                "symbol": symbol,
                "timeframe": timeframe,
                "reason": reason,
            }
        )

    async def error_alert(self, error: str):
        self.messages.append({"type": "error", "error": error})

    async def shutdown_alert(self, mode: str, environment: str, reason: str):
        self.messages.append(
            {
                "type": "shutdown",
                "mode": mode,
                "environment": environment,
                "reason": reason,
            }
        )

    async def stop(self):
        return None


class SoakClient:
    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        periods = max(limit, 60)
        cadence = pd.Timedelta(seconds=timeframe_seconds(timeframe))
        index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"),
            periods=periods,
            freq=cadence,
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

    async def fetch_ticker(self, symbol: str) -> dict:
        return {"last": 110.45, "bid": 110.44, "ask": 110.46}

    async def fetch_market(self, symbol: str) -> dict:
        return {"precision": {"amount": 0.001}}

    async def fetch_funding_rate(self, symbol: str) -> float:
        return 0.0

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return {"bids": [[110.44, 1000.0]], "asks": [[110.46, 1000.0]]}


class SoakOrderManager:
    def __init__(self) -> None:
        self.counter = 0

    async def market_order(
        self, symbol: str, side: str, quantity: float, reduce_only: bool = False
    ) -> dict:
        self.counter += 1
        return {
            "id": f"soak-market-{self.counter}",
            "filled": quantity,
            "average": 110.5 if reduce_only else 110.4,
        }

    async def limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool = True,
    ) -> dict:
        self.counter += 1
        return {
            "id": f"soak-limit-{self.counter}",
            "filled": quantity,
            "average": price,
            "status": "closed",
        }

    async def stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        price: float | None = None,
    ) -> dict:
        self.counter += 1
        return {"id": f"soak-stop-{self.counter}"}

    async def take_profit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
        self.counter += 1
        return {"id": f"soak-target-{self.counter}"}

    async def cancel_all_orders(self, symbol: str) -> None:
        return None

    async def cancel_order(
        self, symbol: str, order_id: str, conditional: bool = False
    ) -> None:
        return None


class SoakAggregator:
    def __init__(self, strategy: str) -> None:
        self.calls = 0
        self.strategy = strategy

    def generate(self, _df, higher_trend_bias: int = 0) -> list[FinalSignal]:
        self.calls += 1
        direction = 1 if self.calls == 1 else 0
        confidence = 0.9 if direction else 0.0
        return [
            FinalSignal(
                direction=direction,
                confidence=confidence,
                ta_source="soak",
                strategy=self.strategy,
            )
        ]


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
        bot = LiveTradingLoop(audit_store=self.audit_store)
        bot._heartbeat = RuntimeHeartbeat(
            str(self.report_path.with_name("soak_heartbeat.json"))
        )
        bot.mode = "soak"
        soak_alerter = SoakAlerter()
        bot.client = cast(ExchangeClient, SoakClient())
        bot.alerter = cast(Alerter, soak_alerter)
        symbol, timeframe = self._enabled_scope()
        strategy = "scalp" if timeframe in {"1m", "3m"} else "trend"
        bot.aggregator = cast(SignalAggregator, SoakAggregator(strategy))
        soak_orders = SoakOrderManager()
        bot.order_mgr = cast(OrderManager, soak_orders)
        bot.pos_mgr.rebind_runtime(
            bot.client,
            cast(OrderManager, soak_orders),
            bot.alerter,
            "soak",
        )
        bot.portfolio.update_account(
            AccountInfo(
                total_equity=10_000.0,
                wallet_balance=10_000.0,
                available_balance=10_000.0,
                unrealized_pnl=0.0,
                margin_ratio=0.0,
            )
        )

        bot._market_regimes[f"{symbol}:1h"] = 1
        await bot.alerter.startup_alert("soak", "offline", [symbol])
        self.audit_store.record_event(
            "soak_started",
            "Offline trade-path soak started",
            mode="soak",
            payload={"iterations": self.iterations},
        )
        for _ in range(self.iterations):
            await bot._scan_timeframes_once(
                symbols=[symbol],
                timeframes=[timeframe],
            )

        await bot.stop("soak complete", close_positions=True)
        self.audit_store.record_event(
            "soak_completed",
            "Offline trade-path soak completed",
            mode="soak",
        )
        return self._build_report(bot, soak_alerter)

    @staticmethod
    def _enabled_scope() -> tuple[str, str]:
        disabled = settings.disabled_strategy_scopes_set
        for symbol in settings.symbols_list:
            for timeframe in settings.timeframes_list:
                if f"{symbol}:{timeframe}" not in disabled:
                    return symbol, timeframe
        raise RuntimeError("offline soak requires at least one enabled strategy scope")

    def _build_report(self, bot: LiveTradingLoop, alerter: SoakAlerter) -> SoakReport:
        events = self.audit_store.load_recent_events(500)
        alert_types = [message["type"] for message in alerter.messages]
        report = SoakReport(
            status="ok" if not self.audit_store.load_open_trades("soak") else "failed",
            iterations=self.iterations,
            opened_trades=alert_types.count("trade_opened"),
            completed_trades=alert_types.count("trade_completed"),
            failed_trades=alert_types.count("trade_failed"),
            audit_events=len(events),
            open_trades_after_shutdown=len(self.audit_store.load_open_trades("soak")),
            report_path=str(self.report_path),
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report
