import json

import pytest

from src.main import parse_args
from src.soak import SoakRunner


@pytest.mark.asyncio
async def test_offline_soak_runner_cleans_up_open_trades(tmp_path):
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
