# Async ccxt exchange client — wraps Binance Futures REST + WebSocket APIs
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import List, Optional, TypeVar

import aiohttp
import ccxt.async_support as ccxt
import ccxt.pro as ccxt_pro
import pandas as pd
from ccxt.base.errors import AuthenticationError, InvalidNonce, NetworkError
from loguru import logger

from src.config import settings

T = TypeVar("T")


class ExchangeClient:
    # Initialise config and enable Binance Demo Trading when testnet mode is on
    def __init__(self) -> None:
        self.config = settings.exchange_config
        self._rest: Optional[ccxt.Exchange] = None
        self._ws: Optional[ccxt_pro.Exchange] = None
        self._rest_session: Optional[aiohttp.ClientSession] = None
        self._ws_session: Optional[aiohttp.ClientSession] = None

    # Context manager entry — connect then return self
    async def __aenter__(self):
        await self.connect()
        return self

    # Context manager exit — close both connections
    async def __aexit__(self, *args):
        await self.close()

    # Create REST + WebSocket clients and verify exchange connectivity/auth
    async def connect(self) -> None:
        attempts = settings.exchange_connect_attempts
        for attempt in range(1, attempts + 1):
            try:
                await self._connect_once()
                break
            except AuthenticationError as exc:
                await self.close()
                environment = settings.binance_environment
                logger.error("Binance authentication failed: {}", exc)
                raise RuntimeError(
                    f"Binance API authentication failed for {environment}. "
                    "Check that BINANCE_API_KEY and BINANCE_API_SECRET were created "
                    "for this environment and have Futures permissions."
                ) from exc
            except NetworkError as exc:
                await self.close()
                if attempt == attempts:
                    raise
                delay = settings.exchange_connect_backoff_seconds * attempt
                logger.warning(
                    "Binance connection failed attempt {}/{}: {}. "
                    "Retrying in {:.1f}s",
                    attempt,
                    attempts,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

        if settings.binance_demo:
            logger.info("Connected to Binance Futures DEMO")
        else:
            logger.warning("Connected to Binance Futures MAINNET")

    async def _connect_once(self) -> None:
        self._rest_session = self._create_aiohttp_session()
        self._ws_session = self._create_aiohttp_session()
        self._rest = ccxt.binanceusdm({**self.config, "session": self._rest_session})
        self._ws = ccxt_pro.binanceusdm({**self.config, "session": self._ws_session})

        # Binance deprecated Futures testnet — use demo environment instead
        if settings.binance_demo:
            self._enable_demo_trading(self._rest)
            self._enable_demo_trading(self._ws)

        # load_markets checks public connectivity; fetch_balance checks auth.
        await self._rest.load_markets()
        if self.config.get("apiKey") and self.config.get("secret"):
            await self._rest.fetch_balance()

    @staticmethod
    def _enable_demo_trading(exchange):
        enable_demo = getattr(exchange, "enable_demo_trading", None)
        if not callable(enable_demo):
            raise RuntimeError(
                "Installed ccxt does not support Binance Demo Trading. "
                "Upgrade ccxt to version 4.5.6 or newer."
            )
        enable_demo(True)

    @staticmethod
    def _create_aiohttp_session() -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.ThreadedResolver(),
            enable_cleanup_closed=True,
        )
        return aiohttp.ClientSession(connector=connector)

    # Gracefully close both REST and WebSocket connections
    async def close(self):
        resources = (
            self._rest,
            self._ws,
            self._rest_session,
            self._ws_session,
        )
        self._rest = None
        self._ws = None
        self._rest_session = None
        self._ws_session = None
        for resource in resources:
            if resource is None:
                continue
            with suppress(Exception):
                await resource.close()

    async def _read_with_retries(
        self, operation: Callable[[], Awaitable[T]], label: str
    ) -> T:
        attempts = settings.exchange_read_attempts
        for attempt in range(1, attempts + 1):
            try:
                return await operation()
            except InvalidNonce as exc:
                if attempt == attempts:
                    raise
                await self._resync_time()
                await self._read_retry_delay(label, exc, attempt, attempts)
            except NetworkError as exc:
                if attempt == attempts:
                    raise
                await self._read_retry_delay(label, exc, attempt, attempts)
        raise RuntimeError(f"Unreachable retry state for {label}")

    async def _resync_time(self) -> None:
        try:
            await self.rest.load_time_difference()
            logger.warning("Resynchronized Binance server time after InvalidNonce")
        except Exception as exc:
            logger.warning(
                "Binance time resynchronization failed: {}",
                type(exc).__name__,
            )

    @staticmethod
    async def _read_retry_delay(
        label: str, exc: Exception, attempt: int, attempts: int
    ) -> None:
        delay = settings.exchange_read_backoff_seconds * attempt
        logger.warning(
            "{} failed attempt {}/{}: {}. Retrying in {:.1f}s",
            label,
            attempt,
            attempts,
            type(exc).__name__,
            delay,
        )
        await asyncio.sleep(delay)

    # Lazy-accessor for the REST client (raises if not connected)
    @property
    def rest(self) -> ccxt.Exchange:
        if not self._rest:
            raise RuntimeError(
                "Exchange not connected. Use 'async with ExchangeClient()'"
            )
        return self._rest

    # Lazy-accessor for the WebSocket client (raises if not connected)
    @property
    def ws(self) -> ccxt_pro.Exchange:
        if not self._ws:
            raise RuntimeError(
                "Exchange not connected. Use 'async with ExchangeClient()'"
            )
        return self._ws

    # Fetch historical OHLCV candles and return a DataFrame with a datetime index
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500
    ) -> pd.DataFrame:
        raw = await self._read_with_retries(
            lambda: self.rest.fetch_ohlcv(symbol, timeframe, limit=limit),
            f"OHLCV {symbol} {timeframe}",
        )
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    # Fetch wallet balance (all currencies)
    async def fetch_balance(self) -> dict:
        return await self._read_with_retries(
            lambda: self.rest.fetch_balance(), "account balance"
        )

    # Fetch open positions, optionally filtered by symbol
    async def fetch_positions(self, symbol: Optional[str] = None) -> List[dict]:
        positions = await self._read_with_retries(
            lambda: self.rest.fetch_positions([symbol] if symbol else []),
            "positions",
        )
        return positions

    # Set leverage for a specific symbol
    async def set_leverage(self, symbol: str, leverage: int):
        await self._read_with_retries(
            lambda: self.rest.set_leverage(leverage, symbol),
            f"set leverage {symbol}",
        )

    # Generic order-creation wrapper
    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> dict:
        try:
            return await self.rest.create_order(
                symbol, order_type, side, amount, price, params or {}
            )
        except InvalidNonce:
            await self._resync_time()
            raise

    # Cancel a specific order by ID
    async def cancel_order(self, id: str, symbol: str, params: Optional[dict] = None):
        return await self._read_with_retries(
            lambda: self.rest.cancel_order(id, symbol, params or {}),
            f"cancel order {symbol}",
        )

    async def cancel_all_orders(self, symbol: str, *, conditional: bool = False):
        params = {"trigger": True} if conditional else {}
        return await self._read_with_retries(
            lambda: self.rest.cancel_all_orders(symbol, params=params),
            f"cancel all orders {symbol}",
        )

    # List all open orders, optionally for one symbol
    async def fetch_open_orders(
        self, symbol: Optional[str] = None, *, conditional: bool = False
    ) -> List[dict]:
        params = {"trigger": True} if conditional else {}
        return await self._read_with_retries(
            lambda: self.rest.fetch_open_orders(symbol, params=params),
            f"open orders {symbol or 'all'}",
        )

    async def fetch_order(
        self, order_id: str, symbol: str, *, conditional: bool = False
    ) -> dict:
        params = {"trigger": True} if conditional else {}
        return await self._read_with_retries(
            lambda: self.rest.fetch_order(order_id, symbol, params=params),
            f"order {symbol}",
        )

    async def fetch_order_by_client_id(
        self, client_order_id: str, symbol: str, *, conditional: bool = False
    ) -> dict:
        params = (
            {"trigger": True, "clientAlgoId": client_order_id}
            if conditional
            else {"origClientOrderId": client_order_id}
        )
        return await self._read_with_retries(
            lambda: self.rest.fetch_order("", symbol, params=params),
            f"order by client ID {symbol}",
        )

    async def fetch_orders(
        self, symbol: str, *, conditional: bool = False, limit: int = 50
    ) -> List[dict]:
        params = {"trigger": True} if conditional else {}
        return await self._read_with_retries(
            lambda: self.rest.fetch_orders(symbol, limit=limit, params=params),
            f"order history {symbol}",
        )

    # Fetch current ticker (24hr stats) for a symbol
    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._read_with_retries(
            lambda: self.rest.fetch_ticker(symbol), f"ticker {symbol}"
        )

    # Load all markets and return the market info for one symbol
    async def fetch_market(self, symbol: str) -> dict:
        markets = await self._read_with_retries(
            lambda: self.rest.load_markets(), "market metadata"
        )
        if symbol in markets:
            return markets[symbol]

        normalized = self._normalize_market_symbol(symbol)
        for market in markets.values():
            market_id = self._normalize_market_symbol(market.get("id", ""))
            unified_symbol = self._normalize_market_symbol(market.get("symbol", ""))
            if normalized in {market_id, unified_symbol}:
                return market

        raise ValueError(
            f"Market metadata not found for {symbol}; "
            "verify the configured symbol is a Binance USD-M Futures market"
        )

    @staticmethod
    def _normalize_market_symbol(symbol: object) -> str:
        return (
            str(symbol or "").split(":", 1)[0].replace("/", "").replace("-", "").upper()
        )

    # Fetch the current funding rate for a perpetual contract
    async def fetch_funding_rate(self, symbol: str) -> float:
        result = await self.rest.fetch_funding_rate(symbol)
        return result["fundingRate"]

    # Stream a single OHLCV update via WebSocket (from ccxt.pro)
    async def watch_ohlcv(self, symbol: str, timeframe: str = "1h") -> pd.DataFrame:
        raw = await self.ws.watch_ohlcv(symbol, timeframe)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
