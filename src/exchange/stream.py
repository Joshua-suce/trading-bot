# Real-time candle stream via WebSocket — pushes new candles to a callback
import asyncio
from typing import Awaitable, Callable, Optional

from loguru import logger

from src.exchange.client import ExchangeClient


class CandleStream:
    # Store the exchange client, symbol, timeframe, and user-provided callback
    def __init__(
        self,
        client: ExchangeClient,
        symbol: str,
        timeframe: str,
        callback: Callable[[dict], Awaitable[None]],
    ):
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe
        self.callback = callback
        self._task: Optional[asyncio.Task] = None

    # Launch the background streaming task
    async def start(self):
        logger.info(f"Starting candle stream: {self.symbol} {self.timeframe}")
        self._task = asyncio.create_task(self._run())

    # Infinite loop: watch for new candles and invoke the callback with each one
    async def _run(self):
        while True:
            try:
                df = await self.client.watch_ohlcv(self.symbol, self.timeframe)
                if len(df) >= 2:
                    candle = {
                        "symbol": self.symbol,
                        "timeframe": self.timeframe,
                        "timestamp": df.index[-1],
                        "open": df["open"].iloc[-1],
                        "high": df["high"].iloc[-1],
                        "low": df["low"].iloc[-1],
                        "close": df["close"].iloc[-1],
                        "volume": df["volume"].iloc[-1],
                    }
                    await self.callback(candle)
            except Exception as e:
                logger.error(f"Stream error for {self.symbol}: {e}")
                await asyncio.sleep(5)

    # Cancel the background task to stop streaming
    async def stop(self):
        if self._task:
            self._task.cancel()
            logger.info(f"Stopped candle stream: {self.symbol} {self.timeframe}")
