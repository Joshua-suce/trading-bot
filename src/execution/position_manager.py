# Position manager — manages position lifecycle: entry, SL/TP placement, exit
from src.audit import AuditStore
from src.config import settings  # noqa: F401 - compatibility hook for runtime tests
from src.exchange.client import ExchangeClient
from src.execution.exposure_limiter import ExposureLimiter
from src.execution.fill_resolver import FillResolver
from src.execution.order_manager import OrderManager
from src.execution.position_reconciler import PositionReconciler
from src.execution.position_registry import PositionRegistry
from src.execution.protection_manager import ProtectionManager
from src.execution.trade_executor import TradeExecutor
from src.monitoring.alerter import Alerter
from src.risk.portfolio import PortfolioManager
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager


class PositionManager:
    # Wire together all dependencies for position management
    def __init__(
        self,
        client: ExchangeClient,
        order_mgr: OrderManager,
        pos_sizer: PositionSizer,
        sl_mgr: StopLossManager,
        portfolio: PortfolioManager,
        alerter: Alerter | None = None,
        mode: str = "trade",
        audit_store: AuditStore | None = None,
    ):
        self.client = client
        self.orders = order_mgr
        self.sizer = pos_sizer
        self.sl_manager = sl_mgr
        self.portfolio = portfolio
        self.alerter = alerter
        self.mode = mode
        self.audit_store = audit_store or AuditStore()
        self._fill_resolver = FillResolver(self.client)
        self.protection = ProtectionManager(
            client=self.client,
            orders=self.orders,
            audit_store=self.audit_store,
            fill_resolver=self._fill_resolver,
            alerter=self.alerter,
            mode=self.mode,
        )
        self.trades = PositionRegistry(
            portfolio=self.portfolio,
            audit_store=self.audit_store,
            protection=self.protection,
        )
        self._exposure_limiter = ExposureLimiter(
            portfolio=self.portfolio,
            open_trades=self.trades.open_trades,
        )
        self.reconciler = PositionReconciler(
            client=self.client,
            orders=self.orders,
            fill_resolver=self._fill_resolver,
            protection=self.protection,
            trades=self.trades,
            portfolio=self.portfolio,
            alerter=self.alerter,
            audit_store=self.audit_store,
            mode=self.mode,
        )
        self.executor = TradeExecutor(
            client=self.client,
            orders=self.orders,
            fill_resolver=self._fill_resolver,
            sl_manager=self.sl_manager,
            sizer=self.sizer,
            portfolio=self.portfolio,
            alerter=self.alerter,
            audit_store=self.audit_store,
            mode=self.mode,
            trades=self.trades,
            protection=self.protection,
            check_exposure_limits=self._check_exposure_limits,
            fail_reconciliation=self._fail_reconciliation,
            finalize_trade_leg=self._finalize_trade_leg,
        )

    @property
    def open_trades(self):
        return self.trades.open_trades

    def rebind_runtime(self, client, orders, alerter, mode: str) -> None:
        """Replace runtime adapters consistently across execution components."""
        self.client = client
        self.orders = orders
        self.alerter = alerter
        self.mode = mode

        self._fill_resolver.client = client

        self.protection.client = client
        self.protection.orders = orders
        self.protection.fill_resolver = self._fill_resolver
        self.protection.alerter = alerter
        self.protection.mode = mode

        self.reconciler.client = client
        self.reconciler.orders = orders
        self.reconciler._fill_resolver = self._fill_resolver
        self.reconciler.protection = self.protection
        self.reconciler.alerter = alerter
        self.reconciler.mode = mode

        self.executor.client = client
        self.executor.orders = orders
        self.executor.entry_router.client = client
        self.executor.entry_router.orders = orders
        self.executor._fill_resolver = self._fill_resolver
        self.executor.protection = self.protection
        self.executor.alerter = alerter
        self.executor.mode = mode

    # Enter a long position: check limits, size, place market order + SL/TP
    async def enter_long(
        self,
        symbol: str,
        price: float,
        atr: float,
        leverage: int = 1,
        timeframe: str | None = None,
        signal_timestamp=None,
        strategy: str | None = None,
        ignore_reentry_cooldown: bool = False,
        market_context: dict | None = None,
    ) -> bool:
        return await self.executor.enter_long(
            symbol,
            price,
            atr,
            leverage,
            timeframe,
            signal_timestamp=signal_timestamp,
            strategy=strategy,
            ignore_reentry_cooldown=ignore_reentry_cooldown,
            market_context=market_context,
        )

    async def enter_short(
        self,
        symbol: str,
        price: float,
        atr: float,
        leverage: int = 1,
        timeframe: str | None = None,
        signal_timestamp=None,
        strategy: str | None = None,
        ignore_reentry_cooldown: bool = False,
        market_context: dict | None = None,
    ) -> bool:
        return await self.executor.enter_short(
            symbol,
            price,
            atr,
            leverage,
            timeframe,
            signal_timestamp=signal_timestamp,
            strategy=strategy,
            ignore_reentry_cooldown=ignore_reentry_cooldown,
            market_context=market_context,
        )

    # Exit a position: market order, record PnL, cancel related SL/TP orders
    async def exit_position(self, position_key: str, reason: str = "manual"):
        await self.executor.exit_position(position_key, reason)

    async def partial_exit_position(
        self,
        position_key: str,
        fraction: float,
        reason: str,
    ) -> bool:
        return await self.executor.partial_exit_position(
            position_key,
            fraction,
            reason,
        )

    # Close every open position (e.g. on shutdown)
    async def close_all(self):
        await self.executor.close_all()

    async def close_symbol(self, symbol: str, reason: str = "manual") -> bool:
        return await self.executor._close_symbol_positions(symbol, reason)

    async def reconcile_exchange_state(
        self, *, auto_close_unmanaged: bool = False
    ) -> bool | None:
        return await self.reconciler.reconcile_exchange_state(
            self.executor._notify_trade_completed,
            auto_close_unmanaged=auto_close_unmanaged,
        )

    async def _fail_reconciliation(
        self,
        event_type: str,
        reason: str,
        *,
        symbol: str | None = None,
        correlation_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        await self.reconciler._fail_reconciliation(
            event_type,
            reason,
            symbol=symbol,
            correlation_id=correlation_id,
            payload=payload,
        )

    async def _finalize_trade_leg(
        self,
        position_key: str,
        *,
        exit_price: float,
        reason: str,
        correlation_id: str,
    ) -> None:
        await self.reconciler._finalize_trade_leg(
            position_key,
            exit_price=exit_price,
            reason=reason,
            correlation_id=correlation_id,
            notify_trade_completed=self.executor._notify_trade_completed,
        )

    def _check_exposure_limits(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        position_key: str,
        *,
        timeframe: str | None = None,
    ) -> tuple[bool, str]:
        return self._exposure_limiter.check_exposure_limits(
            symbol,
            side,
            entry_price,
            quantity,
            position_key,
            trades_for_symbol=self.trades.trades_for_symbol,
            open_notional=self.trades.open_notional,
            timeframe=timeframe,
        )
