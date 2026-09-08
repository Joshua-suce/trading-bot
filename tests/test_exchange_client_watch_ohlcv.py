import asyncio

import pytest

from src.exchange.client import ExchangeClient


class HangingWs:
    def __init__(self):
        self.closed = False

    async def watch_ohlcv(self, symbol, timeframe):
        # Simulate a silently-stalled WebSocket connection: never resolves.
        await asyncio.sleep(3600)

    async def close(self):
        self.closed = True


class RespondingWs:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    async def watch_ohlcv(self, symbol, timeframe):
        return self.rows

    async def close(self):
        self.closed = True


def build_client(ws):
    client = ExchangeClient()
    client._ws = ws
    return client


@pytest.mark.asyncio
async def test_watch_ohlcv_times_out_and_closes_stalled_connection(monkeypatch):
    monkeypatch.setattr("src.exchange.client.settings.scalp_stream_fallback_seconds", 0.05)
    ws = HangingWs()
    client = build_client(ws)

    with pytest.raises(asyncio.TimeoutError):
        await client.watch_ohlcv("BTC/USDT:USDT", "1m")

    assert ws.closed is True


@pytest.mark.asyncio
async def test_watch_ohlcv_returns_frame_when_stream_is_healthy():
    ws = RespondingWs([[1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 10.0]])
    client = build_client(ws)

    df = await client.watch_ohlcv("BTC/USDT:USDT", "1m")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert ws.closed is False
