import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    command: list[str]


def _checks(include_slow: bool) -> list[Check]:
    python = sys.executable
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
        Check("health", [python, "-m", "src.main", "--mode", "health"]),
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
                "data/reports/smoke_soak_report.json",
            ],
        ),
    ]
    if include_slow:
        checks.append(
            Check(
                "offline-backtest",
                [
                    python,
                    "-m",
                    "src.main",
                    "--mode",
                    "backtest",
                    "--offline",
                    "--limit",
                    "120",
                ],
            )
        )
    return checks


def run_check(check: Check) -> int:
    print(f"\n==> {check.name}: {' '.join(check.command)}", flush=True)
    result = subprocess.run(check.command, check=False)
    if result.returncode:
        print(f"FAILED: {check.name} exited {result.returncode}", flush=True)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production readiness checks")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slower checks such as offline backtest",
    )
    args = parser.parse_args(argv)

    for check in _checks(include_slow=not args.fast):
        code = run_check(check)
        if code:
            return code
    print("\nAll smoke checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
