import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]
    env: dict[str, str] | None = None


def _checks(audit_path: str, soak_report_path: str) -> list[Check]:
    python = sys.executable
    isolated_env = {"AUDIT_DB_PATH": audit_path}
    checks = [
        Check("black", [python, "-m", "black", "--check", "src", "tests", "scripts"]),
        Check(
            "isort",
            [python, "-m", "isort", "--check-only", "src", "tests", "scripts"],
        ),
        Check("ruff", [python, "-m", "ruff", "check", "src", "tests", "scripts"]),
        Check("secret-scan", [python, "scripts/secret_scan.py"]),
        Check("mypy", [python, "-m", "mypy", "src"]),
        Check(
            "pytest-coverage",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "--cov=src",
                "--cov-report=term",
                "--cov-fail-under=75",
            ],
        ),
        Check(
            "health",
            [python, "-m", "src.main", "--mode", "health"],
            env=isolated_env,
        ),
        Check(
            "offline-soak",
            [
                python,
                "-m",
                "src.main",
                "--mode",
                "soak",
                "--soak-iterations",
                "2",
                "--soak-report",
                soak_report_path,
            ],
            env=isolated_env,
        ),
    ]
    return checks


def run_check(check: Check) -> int:
    print(f"\n==> {check.name}: {' '.join(check.command)}", flush=True)
    env = os.environ.copy()
    env.update(check.env or {})
    result = subprocess.run(check.command, check=False, env=env)
    if result.returncode:
        print(f"FAILED: {check.name} exited {result.returncode}", flush=True)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production readiness checks")
    parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="trading-bot-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        checks = _checks(
            str(temp_path / "audit.db"),
            str(temp_path / "soak_report.json"),
        )
        for check in checks:
            code = run_check(check)
            if code:
                return code
    print("\nAll smoke checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
