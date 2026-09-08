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

    if mode == "trade" and cfg.require_signal_confluence:
        # No live decision path (SignalAggregator/quality_gate/decision_policy)
        # reads require_signal_confluence or ml_confidence_threshold, and no
        # ML model is wired into live trading at all right now - the live
        # signal path is TA-only. An operator who set this expecting trades
        # to require ML confirmation is silently getting TA-only signals
        # with no confluence check whatsoever, which is worse than either
        # having the feature or clearly not offering it.
        raise RuntimeError(
            "REQUIRE_SIGNAL_CONFLUENCE=true has no effect: no ML model is "
            "wired into live trading, so this setting cannot enforce "
            "TA/ML agreement. Set REQUIRE_SIGNAL_CONFLUENCE=false to "
            "acknowledge trading is TA-only."
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
