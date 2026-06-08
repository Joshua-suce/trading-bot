import asyncio

import numpy as np
import pandas as pd
import pytest

from src.exchange.account import AccountInfo
from src.live import loop as live_loop_module
from src.live.loop import LiveTradingLoop


@pytest.fixture(autouse=True)
def isolate_live_loop_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "audit_db_path",
        str(tmp_path / "live-loop-audit.db"),
    )


def test_live_loop_uses_injected_audit_store(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "injected.db")
    bot = LiveTradingLoop(audit_store=audit)

    assert bot.audit_store is audit
    assert bot.order_mgr.audit_store is audit
    assert bot.pos_mgr.audit_store is audit


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


class LifecycleAlerter(NoopAlerter):
    def __init__(self):
        self.shutdown_reasons = []

    async def shutdown_alert(self, mode, environment, reason):
        self.shutdown_reasons.append(reason)


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
async def test_startup_reconciliation_unavailable_preserves_positions(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = NoopAlerter()
    stopped = []

    async def refresh_account(required=False):
        return None

    async def reconcile():
        return None

    async def stop(reason, close_positions=None):
        stopped.append((reason, close_positions))

    monkeypatch.setattr(bot, "_refresh_account", refresh_account)
    monkeypatch.setattr(bot.pos_mgr, "restore_open_trades_from_audit", lambda: 1)
    monkeypatch.setattr(bot.pos_mgr, "reconcile_exchange_state", reconcile)
    monkeypatch.setattr(bot, "stop", stop)

    await bot.start()

    assert stopped == [("startup reconciliation unavailable", False)]


@pytest.mark.asyncio
async def test_graceful_shutdown_preserves_protected_positions(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = LifecycleAlerter()
    bot.pos_mgr.open_trades["BTCUSDT"] = object()
    closed = []

    async def close_all():
        closed.append(True)

    monkeypatch.setattr(bot.pos_mgr, "close_all", close_all)

    await bot.stop("cancelled")

    assert closed == []
    assert bot.pos_mgr.open_trades
    events = bot.audit_store.load_recent_events(2)
    assert any(
        event["event_type"] == "positions_preserved_on_shutdown" for event in events
    )


@pytest.mark.asyncio
async def test_fatal_shutdown_flattens_positions(monkeypatch):
    bot = LiveTradingLoop()
    bot.client = NoopClient()
    bot.alerter = LifecycleAlerter()
    closed = []

    async def close_all():
        closed.append(True)

    monkeypatch.setattr(bot.pos_mgr, "close_all", close_all)

    await bot.stop("fatal error")

    assert closed == [True]


@pytest.mark.asyncio
async def test_telegram_commands_update_durable_trading_controls(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "telegram-controls.db"))
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

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "telegram-status.db"))
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

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "telegram-replay.db"))
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
async def test_trade_account_refresh_degrades_without_stopping_after_three_attempts(
    monkeypatch,
):
    bot = LiveTradingLoop()
    bot.alerter = NoopAlerter()

    async def fail_account(_client):
        raise RuntimeError("temporary demo account outage")

    monkeypatch.setattr(live_loop_module, "get_account_info", fail_account)

    await bot._refresh_account()
    await bot._refresh_account()
    await bot._refresh_account()

    assert bot._account_refresh_failures == 3
    assert bot._account_degradation_alerted is True
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "account_refresh_degraded"


@pytest.mark.asyncio
async def test_account_refresh_recovery_clears_degraded_state(monkeypatch):
    bot = LiveTradingLoop()
    bot._account_refresh_failures = 3
    bot._account_degradation_alerted = True
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
    assert bot._account_degradation_alerted is False
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "account_refresh_recovered"


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
    assert bot.audit_store.get_control("peak_equity") == "1234.0"


@pytest.mark.asyncio
async def test_account_refresh_if_due_is_throttled(monkeypatch):
    bot = LiveTradingLoop()
    calls = []

    async def refresh(required=False):
        calls.append(required)
        bot._last_account_refresh_at = 100.0

    monkeypatch.setattr(bot, "_refresh_account", refresh)
    monkeypatch.setattr(live_loop_module.time, "monotonic", lambda: 100.0)

    await bot._refresh_account_if_due()
    await bot._refresh_account_if_due()

    assert calls == [False]


@pytest.mark.asyncio
async def test_failed_account_refresh_blocks_new_entries(tmp_path):
    from src.audit import AuditStore

    bot = LiveTradingLoop(audit_store=AuditStore(tmp_path / "stale-account.db"))
    bot._account_refresh_failures = 1
    generated = []

    class Aggregator:
        def generate(self, _df):
            generated.append(True)

    bot.aggregator = Aggregator()
    await bot._execute_trade(
        {"symbol": "BTCUSDT", "timeframe": "5m", "close": 100.0},
        df_ind=pd.DataFrame({"close": [100.0]}),
    )

    assert generated == []
    event = bot.audit_store.load_recent_events(1)[0]
    assert event["event_type"] == "entry_blocked_account_stale"


def test_live_loop_restores_persisted_peak_equity(tmp_path):
    from src.audit import AuditStore

    audit = AuditStore(tmp_path / "risk-state.db")
    audit.set_control("peak_equity", "6000.0", "test")
    bot = LiveTradingLoop(audit_store=audit)

    bot._restore_persisted_risk_state()

    assert bot.portfolio.peak_equity == 6000.0


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


def test_next_scan_aligns_to_exchange_candle_boundary(monkeypatch):
    monkeypatch.setattr(
        live_loop_module.settings,
        "candle_close_grace_seconds",
        2.0,
    )
    monkeypatch.setattr(live_loop_module.settings, "scan_sleep_seconds", 5.0)

    due = LiveTradingLoop._next_candle_scan_due(
        "15m",
        monotonic_now=100.0,
        epoch_now=12 * 3600 + 7 * 60,
    )

    assert due == 100.0 + 8 * 60 + 2.0


def test_live_loop_loads_configured_xgboost_model(monkeypatch, tmp_path):
    model_path = tmp_path / "xgb_classifier.json"
    model_path.write_text("{}", encoding="utf-8")

    class FakeClassifier:
        def load(self, path):
            assert path == str(model_path)

        def predict_with_confidence(self, X):
            return np.array([1]), np.array([0.9])

    monkeypatch.setattr(live_loop_module.settings, "model_dir", str(tmp_path))
    monkeypatch.setattr(live_loop_module, "XGBoostClassifier", FakeClassifier)

    ensemble = LiveTradingLoop._load_ensemble()

    assert ensemble.is_ready()


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
