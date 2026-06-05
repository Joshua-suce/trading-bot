from dataclasses import dataclass

from src.config import Settings, settings
from src.governance import StrategyApprovalStore


@dataclass(frozen=True)
class PreflightResult:
    mode: str
    environment: str
    credentials_required: bool


def run_preflight(
    mode: str, *, offline: bool = False, cfg: Settings = settings
) -> PreflightResult:
    environment = "demo" if cfg.binance_testnet else "mainnet"

    if mode in {"paper", "live"} and not cfg.has_exchange_credentials:
        raise RuntimeError(
            f"{mode} mode requires BINANCE_API_KEY and BINANCE_API_SECRET "
            f"for {environment}."
        )

    if mode == "paper" and not cfg.binance_testnet and not cfg.allow_mainnet_paper:
        raise RuntimeError(
            "Paper mode is pointed at Binance mainnet. Set BINANCE_TESTNET=true "
            "or explicitly set ALLOW_MAINNET_PAPER=true."
        )

    if mode == "live" and not cfg.binance_testnet and not cfg.allow_live_trading:
        raise RuntimeError(
            "Live mainnet trading is disabled. Set ALLOW_LIVE_TRADING=true only "
            "after keys, risk limits, alerts, and deployment controls are verified."
        )

    if mode == "live":
        approved, reason = StrategyApprovalStore(cfg=cfg).validate_for_live()
        if not approved:
            raise RuntimeError(f"Live mode blocked by strategy governance: {reason}")

    return PreflightResult(
        mode=mode,
        environment=environment,
        credentials_required=mode in {"paper", "live"} and not offline,
    )
