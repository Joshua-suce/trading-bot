from dataclasses import dataclass

from src.config import Settings, settings
from src.governance import StrategyApprovalStore


@dataclass(frozen=True)
class PreflightResult:
    mode: str
    environment: str
    credentials_required: bool


def run_preflight(mode: str, *, cfg: Settings = settings) -> PreflightResult:
    environment = cfg.binance_environment

    if mode == "trade" and not cfg.has_exchange_credentials:
        raise RuntimeError(
            "trade mode requires BINANCE_API_KEY and BINANCE_API_SECRET "
            f"for {environment}."
        )

    if mode == "trade" and environment == "mainnet" and not cfg.allow_mainnet_trading:
        raise RuntimeError(
            "Mainnet trading is disabled. Set ALLOW_MAINNET_TRADING=true only "
            "after keys, risk limits, alerts, and deployment controls are verified."
        )

    if mode == "trade" and environment == "mainnet":
        approved, reason = StrategyApprovalStore(cfg=cfg).validate_for_mainnet()
        if not approved:
            raise RuntimeError(
                f"Mainnet trading blocked by strategy governance: {reason}"
            )

    return PreflightResult(
        mode=mode,
        environment=environment,
        credentials_required=mode == "trade",
    )
