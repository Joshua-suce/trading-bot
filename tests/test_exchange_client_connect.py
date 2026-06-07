import pytest
from ccxt.base.errors import AuthenticationError, InvalidNonce, RequestTimeout

from src.exchange import client as client_module
from src.exchange.client import ExchangeClient


@pytest.mark.asyncio
async def test_connect_retries_transient_network_failure(monkeypatch):
    client = ExchangeClient()
    attempts = 0
    delays = []

    async def connect_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RequestTimeout("temporary timeout")

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(client_module.settings, "exchange_connect_attempts", 3)
    monkeypatch.setattr(client_module.settings, "exchange_connect_backoff_seconds", 2.0)
    monkeypatch.setattr(client, "_connect_once", connect_once)
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    await client.connect()

    assert attempts == 2
    assert delays == [2.0]


@pytest.mark.asyncio
async def test_connect_does_not_retry_authentication_failure(monkeypatch):
    client = ExchangeClient()
    attempts = 0

    async def connect_once():
        nonlocal attempts
        attempts += 1
        raise AuthenticationError("invalid key")

    monkeypatch.setattr(client_module.settings, "exchange_connect_attempts", 3)
    monkeypatch.setattr(client, "_connect_once", connect_once)

    with pytest.raises(RuntimeError, match="authentication failed"):
        await client.connect()

    assert attempts == 1


@pytest.mark.asyncio
async def test_connect_raises_after_network_attempts_exhausted(monkeypatch):
    client = ExchangeClient()
    attempts = 0

    async def connect_once():
        nonlocal attempts
        attempts += 1
        raise RequestTimeout("persistent timeout")

    async def sleep(_delay):
        return None

    monkeypatch.setattr(client_module.settings, "exchange_connect_attempts", 2)
    monkeypatch.setattr(client, "_connect_once", connect_once)
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    with pytest.raises(RequestTimeout, match="persistent timeout"):
        await client.connect()

    assert attempts == 2


@pytest.mark.asyncio
async def test_close_attempts_every_resource_when_one_close_fails():
    closed = []

    class Resource:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        async def close(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    client = ExchangeClient()
    client._rest = Resource("rest", fail=True)
    client._ws = Resource("ws")
    client._rest_session = Resource("rest-session")
    client._ws_session = Resource("ws-session")

    await client.close()

    assert closed == ["rest", "ws", "rest-session", "ws-session"]
    assert client._rest is None
    assert client._ws is None


@pytest.mark.asyncio
async def test_runtime_read_resynchronizes_time_after_invalid_nonce(monkeypatch):
    client = ExchangeClient()
    attempts = 0
    synchronized = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InvalidNonce("clock drift")
        return {"ok": True}

    async def resync():
        synchronized.append(True)

    async def sleep(_delay):
        return None

    monkeypatch.setattr(client_module.settings, "exchange_read_attempts", 3)
    monkeypatch.setattr(client, "_resync_time", resync)
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    assert await client._read_with_retries(operation, "test read") == {"ok": True}
    assert synchronized == [True]
    assert attempts == 2


@pytest.mark.asyncio
async def test_runtime_read_retries_request_timeout(monkeypatch):
    client = ExchangeClient()
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RequestTimeout("temporary")
        return "complete"

    async def sleep(_delay):
        return None

    monkeypatch.setattr(client_module.settings, "exchange_read_attempts", 3)
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)

    assert await client._read_with_retries(operation, "test read") == "complete"
    assert attempts == 3
