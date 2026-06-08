import csv
import io
import json
from datetime import datetime, timezone

from src.audit import AuditStore
from src.monitoring.history import history_payload, render_history
from src.risk.portfolio import TradeRecord


def _closed_trade(symbol: str = "BTCUSDT") -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        side="long",
        entry_price=100.0,
        quantity=2.0,
        timestamp=datetime(2026, 6, 8, tzinfo=timezone.utc),
        timeframe="5m",
        exit_price=105.0,
        pnl=10.0,
        pnl_pct=0.05,
        exit_reason="take_profit",
    )


def test_closed_trade_is_recovered_when_open_record_is_missing(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))

    audit.record_closed_trade(
        _closed_trade(),
        mode="trade",
        correlation_id="recovered-close",
    )

    trades = audit.load_trade_history(status="closed")
    assert len(trades) == 1
    assert trades[0]["correlation_id"] == "recovered-close"
    assert trades[0]["status"] == "closed"
    assert trades[0]["pnl"] == 10.0
    assert audit.load_recent_events(1)[0]["event_type"] == "trade_closed"


def test_history_payload_filters_and_summarizes(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    audit.record_closed_trade(
        _closed_trade("BTCUSDT"),
        mode="trade",
        correlation_id="btc-win",
    )
    losing_trade = _closed_trade("ETHUSDT")
    losing_trade.side = "short"
    losing_trade.exit_price = 102.0
    losing_trade.pnl = -4.0
    losing_trade.pnl_pct = -0.02
    losing_trade.exit_reason = "stop_loss"
    audit.record_closed_trade(
        losing_trade,
        mode="trade",
        correlation_id="eth-loss",
    )

    payload = history_payload(
        audit,
        limit=100,
        status="closed",
        symbol=None,
    )

    assert payload["summary"]["records"] == 2
    assert payload["summary"]["wins"] == 1
    assert payload["summary"]["losses"] == 1
    assert payload["summary"]["net_pnl"] == 6.0
    assert payload["summary"]["profit_factor"] == 2.5

    eth_payload = history_payload(
        audit,
        limit=100,
        status="closed",
        symbol="ethusdt",
    )
    assert [trade["symbol"] for trade in eth_payload["trades"]] == ["ETHUSDT"]


def test_history_renders_json_csv_and_table(tmp_path):
    audit = AuditStore(str(tmp_path / "audit.db"))
    audit.record_closed_trade(
        _closed_trade(),
        mode="trade",
        correlation_id="rendered",
    )
    payload = history_payload(audit, limit=10, status="all", symbol=None)

    json_output = json.loads(render_history(payload, "json"))
    assert json_output["trades"][0]["correlation_id"] == "rendered"

    csv_rows = list(csv.DictReader(io.StringIO(render_history(payload, "csv"))))
    assert csv_rows[0]["symbol"] == "BTCUSDT"

    table = render_history(payload, "table")
    assert "Audit DB:" in table
    assert "BTCUSDT" in table
    assert "Net PnL: 10.00000000" in table
