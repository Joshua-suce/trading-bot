import json

import pytest

from src.config import settings
from src.main import parse_args
from src.soak import SoakRunner


@pytest.mark.asyncio
async def test_offline_soak_runner_cleans_up_open_trades(tmp_path, monkeypatch):
    # Isolate from whatever DISABLED_STRATEGY_SCOPES happens to be set to in
    # the local .env (e.g. scopes disabled after a real backtest review) -
    # this test exercises soak/shutdown lifecycle mechanics via a specific
    # scope, not the scope-disable feature itself, so it must not depend on
    # ambient config.
    monkeypatch.setattr(settings, "disabled_strategy_scopes", "")
    report_path = tmp_path / "soak_report.json"
    runner = SoakRunner(
        audit_path=str(tmp_path / "audit.db"),
        report_path=str(report_path),
        iterations=2,
    )

    report = await runner.run()

    assert report.status == "ok"
    assert report.opened_trades == 1
    assert report.completed_trades == 1
    assert report.open_trades_after_shutdown == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"


def test_parse_args_soak_mode():
    args = parse_args(
        [
            "--mode",
            "soak",
            "--soak-iterations",
            "3",
            "--soak-report",
            "tmp/report.json",
        ]
    )

    assert args.mode == "soak"
    assert args.soak_iterations == 3
    assert args.soak_report == "tmp/report.json"
