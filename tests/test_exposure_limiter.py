from datetime import datetime

from pytest import MonkeyPatch

from src.config import settings
from src.exchange.account import AccountInfo
from src.execution.exposure_limiter import ExposureLimiter
from src.risk.portfolio import PortfolioManager, TradeRecord


def make_trade(
    symbol: str = "BTCUSDT",
    side: str = "long",
    entry_price: float = 100.0,
    quantity: float = 1.0,
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        timestamp=datetime.now(),
    )


def make_limiter(total_equity: float = 5000.0) -> ExposureLimiter:
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountInfo(
            total_equity=total_equity,
            wallet_balance=total_equity,
            available_balance=total_equity,
            unrealized_pnl=0.0,
            margin_ratio=0.0,
        )
    )
    return ExposureLimiter(portfolio=portfolio, open_trades={})


def helper(trades: list[tuple[str, TradeRecord]]):
    def _trades_for_symbol(symbol: str):
        return [(k, t) for k, t in trades if t.symbol == symbol]

    def _open_notional(symbol: str | None = None):
        candidates = [(k, t) for k, t in trades if symbol is None or t.symbol == symbol]
        return sum(t.entry_price * t.quantity for _, t in candidates)

    return _trades_for_symbol, _open_notional


class TestExposureLimiterConcurrentEntryRace:
    # A passed check isn't recorded into open_trades until many awaits
    # later (order submission, fill validation, protective-order
    # placement). These simulate two symbols whose entry attempts are
    # in flight concurrently in the same scan batch: neither has reached
    # open_trades yet when the second one's check runs.

    def test_reservation_from_in_flight_entry_counts_toward_total_notional(
        self, monkeypatch: MonkeyPatch
    ):
        # Isolate the total-notional check: disable the symbol and
        # correlated caps (BTCUSDT/ETHUSDT are correlated by default) so
        # only the dimension under test can reject.
        monkeypatch.setattr(settings, "correlated_symbols", "")
        monkeypatch.setattr(settings, "max_symbol_open_notional_pct", 1.0)
        monkeypatch.setattr(settings, "max_total_open_notional_pct", 0.20)
        limiter = make_limiter(total_equity=5_000.0)  # total cap = $1000
        sym_fn, not_fn = helper([])  # nothing in open_trades yet for either

        first_ok, first_reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            600.0,
            1.0,
            "BTCUSDT:5m:trend",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert first_ok, first_reason

        second_ok, second_reason = limiter.check_exposure_limits(
            "ETHUSDT",
            "long",
            500.0,
            1.0,
            "ETHUSDT:5m:trend",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )

        assert not second_ok
        assert "total exposure limit exceeded" in second_reason

    def test_released_reservation_no_longer_blocks_a_later_entry(
        self, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setattr(settings, "correlated_symbols", "")
        monkeypatch.setattr(settings, "max_symbol_open_notional_pct", 1.0)
        monkeypatch.setattr(settings, "max_total_open_notional_pct", 0.20)
        limiter = make_limiter(total_equity=5_000.0)
        sym_fn, not_fn = helper([])

        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            600.0,
            1.0,
            "BTCUSDT:5m:trend",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert ok, reason

        limiter.release_reservation("BTCUSDT:5m:trend")

        ok, reason = limiter.check_exposure_limits(
            "ETHUSDT",
            "long",
            500.0,
            1.0,
            "ETHUSDT:5m:trend",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert ok, reason

    def test_reservation_counts_toward_leg_count_limit(
        self, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setattr(settings, "max_positions_per_symbol", 1)
        limiter = make_limiter()
        sym_fn, not_fn = helper([])

        first_ok, first_reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT:5m:trend",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert first_ok, first_reason

        second_ok, second_reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT:15m:trend",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )

        assert not second_ok
        assert "max trade legs" in second_reason


class TestExposureLimiter:
    def test_ok_when_no_trades_and_equity_available(self):
        limiter = make_limiter()
        sym_fn, not_fn = helper([])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert ok
        assert reason == "ok"

    def test_blocks_same_side_correlated_exposure(self, monkeypatch: MonkeyPatch):
        monkeypatch.setattr(
            settings,
            "correlated_symbols",
            "BTCUSDT,ETHUSDT,BNBUSDT",
        )
        monkeypatch.setattr(settings, "max_correlated_open_notional_pct", 0.15)
        trade = make_trade(
            symbol="ETHUSDT",
            side="long",
            entry_price=500.0,
            quantity=1.0,
        )
        limiter = make_limiter(total_equity=5_000.0)
        limiter.open_trades["ETHUSDT"] = trade
        sym_fn, not_fn = helper([("ETHUSDT", trade)])

        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            300.0,
            1.0,
            "BTCUSDT",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )

        assert ok is False
        assert "correlated exposure" in reason

    def test_opposite_side_does_not_consume_correlated_directional_budget(
        self,
        monkeypatch: MonkeyPatch,
    ):
        monkeypatch.setattr(settings, "max_correlated_open_notional_pct", 0.15)
        trade = make_trade(
            symbol="ETHUSDT",
            side="short",
            entry_price=500.0,
            quantity=1.0,
        )
        limiter = make_limiter(total_equity=5_000.0)
        limiter.open_trades["ETHUSDT"] = trade
        sym_fn, not_fn = helper([("ETHUSDT", trade)])

        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )

        assert ok is True
        assert reason == "ok"

    def test_blocks_duplicate_position_key(self):
        trade = make_trade()
        limiter = make_limiter()
        limiter.open_trades["BTCUSDT"] = trade
        sym_fn, not_fn = helper([("BTCUSDT", trade)])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert not ok
        assert "already open" in reason

    def test_blocks_when_max_open_positions_reached(self, monkeypatch: MonkeyPatch):
        monkeypatch.setattr(settings, "max_open_positions", 1)
        trade = make_trade()
        limiter = make_limiter()
        limiter.open_trades["ETHUSDT"] = trade
        sym_fn, not_fn = helper([("ETHUSDT", trade)])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert not ok
        assert "max open positions" in reason

    def test_blocks_when_max_per_symbol_reached(self, monkeypatch: MonkeyPatch):
        monkeypatch.setattr(settings, "max_positions_per_symbol", 1)
        trade1 = make_trade(symbol="BTCUSDT", side="long", entry_price=100.0)
        limiter = make_limiter()
        limiter.open_trades["BTCUSDT-5m"] = trade1
        sym_fn, not_fn = helper([("BTCUSDT-5m", trade1)])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT-15m",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert not ok
        assert "max trade legs for BTCUSDT" in reason

    def test_blocks_opposite_side_entry(self):
        trade = make_trade(side="long", entry_price=100.0)
        limiter = make_limiter()
        limiter.open_trades["BTCUSDT-5m"] = trade
        sym_fn, not_fn = helper([("BTCUSDT-5m", trade)])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "short",
            100.0,
            1.0,
            "BTCUSDT-15m",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert not ok
        assert "opposite-side" in reason

    def test_blocks_when_equity_unavailable(self):
        limiter = make_limiter(total_equity=0)
        sym_fn, not_fn = helper([])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert not ok
        assert "equity unavailable" in reason

    def test_blocks_when_total_exposure_exceeded(self, monkeypatch: MonkeyPatch):
        monkeypatch.setattr(settings, "max_total_open_notional_pct", 0.1)
        monkeypatch.setattr(settings, "max_symbol_open_notional_pct", 1.0)
        trade = make_trade(symbol="ETHUSDT", entry_price=100.0, quantity=5.0)
        limiter = make_limiter(total_equity=5000.0)
        limiter.open_trades["ETHUSDT"] = trade
        sym_fn, not_fn = helper([("ETHUSDT", trade)])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert not ok
        assert "total exposure limit exceeded" in reason

    def test_blocks_when_symbol_exposure_exceeded(self, monkeypatch: MonkeyPatch):
        monkeypatch.setattr(settings, "max_symbol_open_notional_pct", 0.02)
        monkeypatch.setattr(settings, "max_total_open_notional_pct", 1.0)
        trade = make_trade(symbol="BTCUSDT", entry_price=100.0, quantity=2.0)
        limiter = make_limiter(total_equity=5000.0)
        limiter.open_trades["BTCUSDT-5m"] = trade
        sym_fn, not_fn = helper([("BTCUSDT-5m", trade)])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT-15m",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert not ok
        assert "symbol exposure limit exceeded" in reason

    def test_passes_when_all_limits_satisfied(self, monkeypatch: MonkeyPatch):
        monkeypatch.setattr(settings, "max_open_positions", 5)
        monkeypatch.setattr(settings, "max_positions_per_symbol", 2)
        monkeypatch.setattr(settings, "max_total_open_notional_pct", 0.5)
        monkeypatch.setattr(settings, "max_symbol_open_notional_pct", 0.3)
        trade = make_trade(symbol="ETHUSDT", entry_price=50.0, quantity=2.0)
        limiter = make_limiter(total_equity=5000.0)
        limiter.open_trades["ETHUSDT"] = trade
        sym_fn, not_fn = helper([("ETHUSDT", trade)])
        ok, reason = limiter.check_exposure_limits(
            "BTCUSDT",
            "long",
            100.0,
            1.0,
            "BTCUSDT-15m",
            trades_for_symbol=sym_fn,
            open_notional=not_fn,
        )
        assert ok
        assert reason == "ok"
