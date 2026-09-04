import pytest

from src.monitoring.alerter import Alerter


class FakeResponse:
    def __init__(self, *, should_fail=False, body=None):
        self.should_fail = should_fail
        self.body = body or {"ok": True, "result": {}}

    def raise_for_status(self):
        if self.should_fail:
            raise RuntimeError("HTTP rejected secret-token")

    def json(self):
        return self.body


class FakeAsyncClient:
    posts: list[tuple[str, dict, float]] = []
    failures_before_success = 0

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, timeout=None):
        self.posts.append((url, json, self.timeout))
        should_fail = len(self.posts) <= self.failures_before_success
        return FakeResponse(should_fail=should_fail)

    async def aclose(self):
        return None


class CapturingAlerter(Alerter):
    def __init__(self):
        super().__init__()
        self.messages = []

    async def send(self, message: str, level: str = "info"):
        self.messages.append((level, message))


@pytest.mark.asyncio
async def test_send_posts_to_enabled_channels(monkeypatch):
    FakeAsyncClient.posts = []
    FakeAsyncClient.failures_before_success = 0
    monkeypatch.setattr("src.monitoring.alerter.httpx.AsyncClient", FakeAsyncClient)

    alerter = Alerter(
        telegram_token="secret-token",
        telegram_chat_id="chat-1",
        discord_webhook="https://discord.example/webhook",
    )

    await alerter.send("bot started", "startup")

    assert len(FakeAsyncClient.posts) == 2
    assert FakeAsyncClient.posts[0][0].endswith("/botsecret-token/sendMessage")
    assert FakeAsyncClient.posts[0][1] == {
        "chat_id": "chat-1",
        "text": "bot started",
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    assert FakeAsyncClient.posts[1][1] == {"content": "```\nbot started\n```"}


@pytest.mark.asyncio
async def test_post_with_retries_redacts_secret_after_failures(monkeypatch, caplog):
    async def no_sleep(_seconds):
        return None

    FakeAsyncClient.posts = []
    FakeAsyncClient.failures_before_success = 5
    monkeypatch.setattr("src.monitoring.alerter.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("src.monitoring.alerter.asyncio.sleep", no_sleep)
    alerter = Alerter(telegram_token="secret-token")

    result = await alerter._post_with_retries(
        "Telegram",
        "https://api.telegram.org/botsecret-token/sendMessage",
        {"text": "hello"},
        attempts=2,
    )

    assert result is False
    assert len(FakeAsyncClient.posts) == 2
    assert "secret-token" not in caplog.text


@pytest.mark.asyncio
async def test_alert_helpers_format_operational_notifications():
    alerter = CapturingAlerter()

    await alerter.startup_alert("trade", "demo", ["BTCUSDT", "ETHUSDT"])
    await alerter.trade_opened_alert(
        "trade", "BTCUSDT", "long", 100.0, 0.5, 95.0, 110.0
    )
    await alerter.trade_completed_alert("trade", "BTCUSDT", "long", 108.0, 4.0, "tp")
    await alerter.trade_failed_alert("trade", "ETHUSDT", "risk blocked")
    await alerter.data_feed_alert("trade", "BNBUSDT", "5m", "timeout")
    await alerter.trade_alert("SOLUSDT", "short", 50.0, 1.25, pnl=-1.5)
    await alerter.error_alert("exchange down")
    await alerter.daily_summary(12.5, 0.625, 8)

    levels = [level for level, _message in alerter.messages]
    assert levels == [
        "startup",
        "trade",
        "trade_completed",
        "trade_failed",
        "data_feed",
        "trade",
        "error",
        "daily",
    ]
    assert "Trade Placed" in alerter.messages[1][1]
    assert "Market Data Unavailable" in alerter.messages[4][1]
    assert "PnL: -1.50" in alerter.messages[5][1]


@pytest.mark.asyncio
async def test_started_alerter_queues_and_flushes_delivery(monkeypatch):
    delivered = []
    alerter = Alerter(discord_webhook="https://discord.example/webhook")

    async def capture(message):
        delivered.append(message)
        return True

    monkeypatch.setattr(alerter, "_deliver", capture)

    await alerter.start()
    assert await alerter.send("queued message")
    await alerter.stop()

    assert delivered == ["queued message"]
    assert alerter.pending_messages == 0


@pytest.mark.asyncio
async def test_telegram_commands_require_authorized_chat_and_user():
    responses = []

    async def handler(command, arguments, user_id, update_id):
        responses.append((command, arguments, user_id, update_id))
        return "ack"

    alerter = CapturingAlerter()
    alerter.telegram_chat_id = "chat-1"
    alerter.telegram_allowed_user_ids = {"42"}
    alerter._command_handler = handler

    unauthorized_chat = {
        "message": {
            "chat": {"id": "chat-2"},
            "from": {"id": "42"},
            "text": "/status",
        }
    }
    unauthorized_user = {
        "message": {
            "chat": {"id": "chat-1"},
            "from": {"id": "99"},
            "text": "/status",
        }
    }
    authorized = {
        "update_id": 123,
        "message": {
            "chat": {"id": "chat-1"},
            "from": {"id": "42"},
            "text": "/pause maintenance",
        },
    }

    assert not await alerter.process_telegram_update(unauthorized_chat)
    assert not await alerter.process_telegram_update(unauthorized_user)
    assert await alerter.process_telegram_update(authorized)
    assert responses == [("/pause", "maintenance", "42", 123)]
    assert alerter.messages == [("telegram_command", "ack")]


@pytest.mark.asyncio
async def test_telegram_api_level_rejection_is_delivery_failure(monkeypatch):
    class RejectingClient(FakeAsyncClient):
        async def post(self, url, json, timeout=None):
            return FakeResponse(body={"ok": False, "description": "chat not found"})

    monkeypatch.setattr("src.monitoring.alerter.httpx.AsyncClient", RejectingClient)
    alerter = Alerter(telegram_token="secret-token", telegram_chat_id="missing")

    assert not await alerter.send("test")
    assert alerter.delivery_failures == 1
