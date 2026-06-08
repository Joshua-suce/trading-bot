# CLI entry point — parses arguments, sets up logging, dispatches to the right mode
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.monitoring.logger import setup_logging
from src.preflight import run_preflight


# Build and parse CLI arguments for mode, symbol, timeframe, etc.
def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="AI Trading Bot for Binance Futures")
    parser.add_argument(
        "--mode",
        type=str,
        default="trade",
        choices=[
            "trade",
            "train",
            "dashboard",
            "admin",
            "health",
            "history",
            "soak",
        ],
        help="Trading mode",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="Timeframe (default: 1h)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of candles to fetch (default: 500)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--admin-action",
        type=str,
        default="status",
        choices=[
            "status",
            "pause",
            "resume",
            "emergency-stop",
            "clear-emergency",
            "strategy-status",
            "revoke-strategy",
            "test-alert",
        ],
        help="Admin control action when --mode admin is used",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Reason recorded for admin control changes",
    )
    parser.add_argument(
        "--soak-iterations",
        type=int,
        default=2,
        help="Number of offline soak scan iterations",
    )
    parser.add_argument(
        "--soak-report",
        type=str,
        default="data/reports/soak_report.json",
        help="Path for soak report JSON",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=100,
        help="Maximum trade-history records to return",
    )
    parser.add_argument(
        "--history-status",
        choices=["all", "open", "closed"],
        default="all",
        help="Filter trade history by lifecycle status",
    )
    parser.add_argument(
        "--history-symbol",
        type=str,
        default="",
        help="Optional symbol filter for trade history",
    )
    parser.add_argument(
        "--history-format",
        choices=["table", "json", "csv"],
        default="table",
        help="Trade-history output format",
    )
    parser.add_argument(
        "--history-output",
        type=str,
        default="",
        help="Optional file path for JSON or CSV history output",
    )
    return parser.parse_args(argv)


# Fetch OHLCV and train ML models (XGBoost + optional LSTM ensemble)
async def run_train(symbol: str, timeframe: str, limit: int):
    from src.exchange.client import ExchangeClient
    from src.models.trainer import ModelTrainer

    logger.info(f"Training models on {symbol} {timeframe}...")
    async with ExchangeClient() as client:
        df = await client.fetch_ohlcv(symbol, timeframe, limit=limit)

    trainer = ModelTrainer()
    ensemble = trainer.train_ensemble(
        df,
        symbol=symbol,
        timeframe=timeframe,
    )
    logger.info("Training complete!")
    return ensemble


# Connect to the configured exchange environment and execute signals.
async def run_trade():
    from src.live.loop import LiveTradingLoop

    loop = LiveTradingLoop()
    await loop.start()


# Launch the Streamlit dashboard as a subprocess
def run_dashboard():
    import subprocess
    import sys

    dashboard_path = str(Path(__file__).parent / "monitoring" / "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


def run_admin(action: str, reason: str = ""):
    from src.audit import AuditStore
    from src.governance import StrategyApprovalStore

    audit_store = AuditStore()
    approval_store = StrategyApprovalStore()
    if action == "test-alert":
        asyncio.run(run_alert_test())
        return
    if action == "pause":
        audit_store.pause_trading(reason or "manual pause")
        logger.warning("Trading manually paused")
    elif action == "resume":
        audit_store.resume_trading(reason or "manual resume")
        logger.warning("Trading manually resumed")
    elif action == "emergency-stop":
        audit_store.activate_emergency_stop(reason or "manual emergency stop")
        logger.critical("Emergency stop activated")
    elif action == "clear-emergency":
        audit_store.clear_emergency_stop(reason or "manual emergency clear")
        logger.warning("Emergency stop cleared")
    elif action == "revoke-strategy":
        approval_store.revoke(reason or "manual strategy revocation")
        audit_store.safe_record_event(
            "strategy_approval_revoked",
            reason or "manual strategy revocation",
            severity="warning",
        )
        logger.warning("Strategy approval revoked")

    allowed, allowed_reason = audit_store.trading_allowed()
    logger.info(f"Trading allowed: {allowed} ({allowed_reason})")
    approved, approval_reason = approval_store.validate_for_mainnet()
    logger.info(f"Strategy approval valid: {approved} ({approval_reason})")
    approval = approval_store.load()
    if approval:
        logger.info(
            f"Strategy approval: approved={approval.approved} "
            f"symbol={approval.symbol} timeframe={approval.timeframe} "
            f"approved_at={approval.approved_at} by={approval.approved_by}"
        )
    controls = audit_store.get_controls()
    if controls:
        logger.info(f"Controls: {controls}")
    open_trades = audit_store.load_open_trades()
    logger.info(f"Audited open trades: {len(open_trades)}")
    for trade in open_trades:
        logger.info(
            f"  {trade['symbol']} {trade['side']} qty={trade['quantity']} "
            f"entry={trade['entry_price']} status={trade['status']}"
        )


async def run_alert_test() -> None:
    from src.config import settings
    from src.monitoring.alerter import Alerter

    alerter = Alerter(
        telegram_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
        discord_webhook=settings.discord_webhook_url,
        queue_size=settings.telegram_alert_queue_size,
        delivery_timeout=settings.telegram_delivery_timeout_seconds,
    )
    if not alerter.enabled:
        raise RuntimeError(
            "No alert channel configured. Set Telegram credentials or Discord webhook."
        )
    await alerter.start()
    await alerter.send(
        "<b>Trading Bot Alert Test</b>\n"
        f"Environment: {settings.binance_environment}\n"
        "Status: delivery channel is operational",
        "test",
    )
    await alerter.stop()
    if not alerter.delivery_successes or alerter.delivery_failures:
        raise RuntimeError("Alert test failed; inspect redacted logs for details")
    logger.info("Alert test delivered successfully")


def run_health() -> int:
    from src.monitoring.health import HealthChecker

    report = HealthChecker().check()
    print(report.to_json())
    return 1 if report.status == "critical" else 0


def run_history(
    *,
    limit: int,
    status: str,
    output_format: str,
    output_path: str = "",
    symbol: str | None = None,
) -> int:
    from src.audit import AuditStore
    from src.monitoring.history import (
        history_payload,
        render_history,
        write_history_output,
    )

    audit_store = AuditStore()
    payload = history_payload(
        audit_store,
        limit=limit,
        status=status,
        symbol=symbol,
    )
    content = render_history(payload, output_format)
    if output_path:
        path = write_history_output(content, output_path)
        logger.info(f"Trade history written to {path}")
    else:
        print(content)
    return 0


async def run_soak(iterations: int, report_path: str) -> int:
    from src.config import settings
    from src.soak import SoakRunner

    runner = SoakRunner(
        audit_path=settings.audit_db_path,
        report_path=report_path,
        iterations=iterations,
    )
    report = await runner.run()
    print(report.to_json())
    return 0 if report.status == "ok" else 1


# Top-level dispatch: parse args, set up logging, route to the chosen mode
def main():
    args = parse_args()
    setup_logging("DEBUG" if args.verbose else None)

    logger.info(f"Starting AI Trading Bot - Mode: {args.mode}")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    preflight = run_preflight(args.mode)
    logger.info(f"Preflight passed: environment={preflight.environment}")

    if args.mode == "soak":
        raise SystemExit(asyncio.run(run_soak(args.soak_iterations, args.soak_report)))
    if args.mode == "train":
        asyncio.run(run_train(args.symbol, args.timeframe, args.limit))
    elif args.mode == "trade":
        asyncio.run(run_trade())
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "admin":
        run_admin(args.admin_action, args.reason)
    elif args.mode == "health":
        raise SystemExit(run_health())
    elif args.mode == "history":
        raise SystemExit(
            run_history(
                limit=args.history_limit,
                status=args.history_status,
                output_format=args.history_format,
                output_path=args.history_output,
                symbol=args.history_symbol or None,
            )
        )
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(2)


if __name__ == "__main__":
    main()
