import asyncio

import pandas as pd
import pytest

from src.exchange.account import AccountInfo
from src.live import loop as live_loop_module
from src.live.loop import LiveTradingLoop


class NoopAlerter:
    pending_messages = 0

    async def start(self, command_handler=None):
        return None

    async def initializing_alert(self, mode, environment):
        return None

    async def startup_alert(self, mode, environment, symbols):
        return None

    async def trade_failed_alert(self, mode, symbol, reason):
        return None

    async def error_alert(self, error):
        return None

    async def shutdown_alert(self, mode, environment, reason):
        return None

    async def stop(self):
        return None


class NoopClient:
    async def connect(self):
        return None

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_trade_account_refresh_failure_is_non_fatal_before_threshold(monkeypatch):
    bot = LiveTradingLoop()

    async def fail_account(_client):
        raise RuntimeError("temporary demo account outage")

    monkeypatch.setattr(live_loop_module, "get_account_info", fail_account)

    await bot._refresh_account()

    assert bot._account_refresh_failures == 1


@pytest.mark.asyncio
async def test_trade_start_stops_after_initial_account_refresh_failure(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = NoopAlerter()
    reached = []

    async def fail_account(_client):
        raise RuntimeError(
            "binanceusdm GET https://demo-fapi.binance.com/fapi/v3/account"
            "?timestamp=1&signature=secret"
        )

    async def reconcile():
        reached.append("reconcile")
        return True

    async def cancel_scan():
        reached.append("scan")
        raise asyncio.CancelledError

    async def stop(reason):
        reached.append("stop")

    monkeypatch.setattr(live_loop_module, "get_account_info", fail_account)
    monkeypatch.setattr(bot.pos_mgr, "restore_open_trades_from_audit", lambda: 0)
    monkeypatch.setattr(bot.pos_mgr, "reconcile_exchange_state", reconcile)
    monkeypatch.setattr(bot, "_scan_due_timeframes_once", cancel_scan)
    monkeypatch.setattr(bot, "stop", stop)

    await bot.start()

    assert reached == ["stop"]
    assert bot.portfolio.account is None
    assert bot._account_refresh_failures == 1
    assert "secret" not in bot._describe_exception(RuntimeError("signature=secret"))


@pytest.mark.asyncio
async def test_telegram_commands_update_durable_trading_controls(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop()
    bot.audit_store = AuditStore(tmp_path / "telegram-controls.db")
    bot.alerter = NoopAlerter()

    paused = await bot._handle_telegram_command("/pause", "maintenance", "42")
    allowed, reason = bot.audit_store.trading_allowed()
    assert not allowed
    assert "pause" in reason
    assert "Trading Paused" in paused

    resumed = await bot._handle_telegram_command("/resume", "", "42")
    assert bot.audit_store.trading_allowed() == (True, "ok")
    assert "Trading Resumed" in resumed

    await bot._handle_telegram_command("/emergency_stop", "incident", "42")
    blocked, reason = bot.audit_store.trading_allowed()
    assert not blocked
    assert "emergency" in reason

    confirmation = await bot._handle_telegram_command("/clear_emergency", "", "42")
    assert "Confirmation Required" in confirmation
    assert not bot.audit_store.trading_allowed()[0]

    cleared = await bot._handle_telegram_command("/clear_emergency", "CONFIRM", "42")
    assert bot.audit_store.trading_allowed() == (True, "ok")
    assert "Emergency Stop Cleared" in cleared


@pytest.mark.asyncio
async def test_telegram_status_and_help_commands(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop()
    bot.audit_store = AuditStore(tmp_path / "telegram-status.db")
    bot.alerter = NoopAlerter()

    status = await bot._handle_telegram_command("/status", "", "42")
    help_text = await bot._handle_telegram_command("/help", "", "42")
    unknown = await bot._handle_telegram_command("/nope", "", "42")

    assert "Trading: ENABLED" in status
    assert "Open trades: 0" in status
    assert "/emergency_stop" in help_text
    assert "Unknown Command" in unknown


@pytest.mark.asyncio
async def test_telegram_command_replay_is_ignored(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop()
    bot.audit_store = AuditStore(tmp_path / "telegram-replay.db")
    bot.alerter = NoopAlerter()

    first = await bot._handle_telegram_command(
        "/pause", "maintenance", "42", update_id=100
    )
    replay = await bot._handle_telegram_command(
        "/resume", "stale replay", "42", update_id=100
    )

    assert "Trading Paused" in first
    assert replay == ""
    assert not bot.audit_store.trading_allowed()[0]


@pytest.mark.asyncio
async def test_trade_account_refresh_fails_after_three_attempts(monkeypatch):
    bot = LiveTradingLoop()

    async def fail_account(_client):
        raise RuntimeError("temporary demo account outage")

    monkeypatch.setattr(live_loop_module, "get_account_info", fail_account)

    await bot._refresh_account()
    await bot._refresh_account()
    with pytest.raises(RuntimeError, match="failed repeatedly"):
        await bot._refresh_account()


@pytest.mark.asyncio
async def test_account_refresh_updates_portfolio(monkeypatch):
    bot = LiveTradingLoop()
    account = AccountInfo(
        total_equity=1234.0,
        wallet_balance=1200.0,
        available_balance=1000.0,
        unrealized_pnl=34.0,
        margin_ratio=0.1,
    )

    async def get_account(_client):
        return account

    monkeypatch.setattr(live_loop_module, "get_account_info", get_account)

    await bot._refresh_account()

    assert bot._account_refresh_failures == 0
    assert bot.portfolio.account == account


class FakeSequentialClient:
    def __init__(self):
        self.calls = []
        self.last_index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"), periods=201, freq="5min"
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=200):
        self.calls.append((symbol, timeframe, limit))
        index = self.last_index
        return pd.DataFrame(
            {
                "open": [100.0 + i * 0.1 for i in range(len(index))],
                "high": [101.0 + i * 0.1 for i in range(len(index))],
                "low": [99.0 + i * 0.1 for i in range(len(index))],
                "close": [100.5 + i * 0.1 for i in range(len(index))],
                "volume": [10.0 + i for i in range(len(index))],
            },
            index=index,
        )


def test_due_scan_pairs_only_returns_due_items(monkeypatch):
    bot = LiveTradingLoop()
    monkeypatch.setattr(live_loop_module.settings, "symbols", "BTCUSDT")
    monkeypatch.setattr(live_loop_module.settings, "timeframes", "5m,15m")
    bot._next_scan_due["BTCUSDT_15m"] = 999.0

    due = bot._due_scan_pairs(100.0)

    assert due == [("BTCUSDT", "5m")]


def test_timeframe_seconds_parses_supported_units():
    assert LiveTradingLoop._timeframe_seconds("5m") == 300
    assert LiveTradingLoop._timeframe_seconds("1h") == 3600
    assert LiveTradingLoop._timeframe_seconds("1d") == 86_400


@pytest.mark.asyncio
async def test_scan_timeframes_runs_in_configured_sequence(monkeypatch):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    fake_client = FakeSequentialClient()
    bot.client = fake_client
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append((candle["symbol"], candle["timeframe"]))

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    await bot._scan_timeframes_once(
        symbols=["BTCUSDT"],
        timeframes=["5m", "15m", "30m"],
    )

    assert fake_client.calls == [
        ("BTCUSDT", "5m", 201),
        ("BTCUSDT", "15m", 201),
        ("BTCUSDT", "30m", 201),
    ]
    assert processed == [
        ("BTCUSDT", "5m"),
        ("BTCUSDT", "15m"),
        ("BTCUSDT", "30m"),
    ]


@pytest.mark.asyncio
async def test_scan_timeframes_skips_duplicate_candle(monkeypatch):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    bot.client = FakeSequentialClient()
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append((candle["symbol"], candle["timeframe"]))

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    await bot._scan_timeframes_once(symbols=["BTCUSDT"], timeframes=["5m"])
    await bot._scan_timeframes_once(symbols=["BTCUSDT"], timeframes=["5m"])

    assert processed == [("BTCUSDT", "5m")]


@pytest.mark.asyncio
async def test_process_timeframe_uses_last_closed_candle(monkeypatch):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    bot.client = FakeSequentialClient()
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append(candle)

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    await bot._process_timeframe("BTCUSDT", "5m")

    assert processed[0]["timestamp"] == bot.client.last_index[-2]
    assert processed[0]["close"] == 100.5 + 199 * 0.1


@pytest.mark.asyncio
async def test_process_timeframe_rejects_bad_data(monkeypatch):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()
    bot.client = FakeSequentialClient()
    processed = []

    async def capture_candle(candle, df_ind=None):
        processed.append(candle)

    monkeypatch.setattr(bot, "_on_candle", capture_candle)

    async def bad_fetch(symbol, timeframe, limit=200):
        df = await FakeSequentialClient().fetch_ohlcv(symbol, timeframe, limit)
        df.loc[df.index[-2], "volume"] = 0.0
        return df

    monkeypatch.setattr(bot.client, "fetch_ohlcv", bad_fetch)

    await bot._process_timeframe("BTCUSDT", "5m")

    assert processed == []
    events = bot.audit_store.load_recent_events(5)
    assert any(event["event_type"] == "data_quality_rejected" for event in events)
