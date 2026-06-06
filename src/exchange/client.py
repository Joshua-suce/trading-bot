# Async ccxt exchange client — wraps Binance Futures REST + WebSocket APIs
from typing import List, Optional

import aiohttp
import ccxt.async_support as ccxt
import ccxt.pro as ccxt_pro
import pandas as pd
from ccxt.base.errors import AuthenticationError
from loguru import logger

from src.config import settings


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
        self._rest_session = self._create_aiohttp_session()
        self._ws_session = self._create_aiohttp_session()
        self._rest = ccxt.binanceusdm({**self.config, "session": self._rest_session})
        self._ws = ccxt_pro.binanceusdm({**self.config, "session": self._ws_session})

        # Binance deprecated Futures testnet — use demo environment instead
        if settings.binance_demo:
            self._enable_demo_trading(self._rest)
            self._enable_demo_trading(self._ws)

        # Quick authentication / permission check to fail fast with clear message
        try:
            # load_markets checks public connectivity; fetch_balance checks auth.
            await self._rest.load_markets()
            if self.config.get("apiKey") and self.config.get("secret"):
                await self._rest.fetch_balance()
        except AuthenticationError as e:
            logger.error("Binance authentication failed: {}", e)
            await self.close()
            environment = settings.binance_environment
            raise RuntimeError(
                f"Binance API authentication failed for {environment}. "
                "Check that BINANCE_API_KEY and BINANCE_API_SECRET were created "
                "for this environment and have Futures permissions."
            ) from e

        if settings.binance_demo:
            logger.info("Connected to Binance Futures DEMO")
        else:
            logger.warning("Connected to Binance Futures MAINNET")

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
        if self._rest:
            await self._rest.close()
        if self._ws:
            await self._ws.close()
        if self._rest_session and not self._rest_session.closed:
            await self._rest_session.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()

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
        raw = await self.rest.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    # Fetch wallet balance (all currencies)
    async def fetch_balance(self) -> dict:
        return await self.rest.fetch_balance()

    # Fetch open positions, optionally filtered by symbol
    async def fetch_positions(self, symbol: Optional[str] = None) -> List[dict]:
        positions = await self.rest.fetch_positions([symbol] if symbol else [])
        return positions

    # Set leverage for a specific symbol
    async def set_leverage(self, symbol: str, leverage: int):
        await self.rest.set_leverage(leverage, symbol)

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
        return await self.rest.create_order(
            symbol, order_type, side, amount, price, params or {}
        )

    # Cancel a specific order by ID
    async def cancel_order(self, id: str, symbol: str):
        return await self.rest.cancel_order(id, symbol)

    # List all open orders, optionally for one symbol
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        return await self.rest.fetch_open_orders(symbol)

    # Fetch current ticker (24hr stats) for a symbol
    async def fetch_ticker(self, symbol: str) -> dict:
        return await self.rest.fetch_ticker(symbol)

    # Load all markets and return the market info for one symbol
    async def fetch_market(self, symbol: str) -> dict:
        markets = await self.rest.load_markets()
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
