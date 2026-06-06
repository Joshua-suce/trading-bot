import argparse
import re
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "data",
    "logs",
    "dist",
    "build",
}
EXCLUDED_FILES = {
    ".env",
    ".env.local",
}
TEXT_EXTENSIONS = {
    ".py",
    ".cmd",
    ".ps1",
    ".sh",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".example",
    ".dockerignore",
    ".gitignore",
    "",
}
SECRET_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "telegram-token": re.compile(r"\b[0-9]{6,}:[A-Za-z0-9_-]{30,}\b"),
    "hardcoded-secret": re.compile(
        r"(?i)\b(api[_-]?key|api[_-]?secret|secret|token|webhook|password)\b"
        r"\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
    ),
}
ALLOWLIST_PATTERNS = (
    re.compile(r"BINANCE_API_KEY="),
    re.compile(r"BINANCE_API_SECRET="),
    re.compile(r"TELEGRAM_BOT_TOKEN="),
    re.compile(r"DISCORD_WEBHOOK_URL="),
    re.compile(r"ci-secret"),
    re.compile(r"ci-key"),
)


def should_scan(path: Path, include_env: bool) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIRS:
        return False
    if not include_env and path.name in EXCLUDED_FILES:
        return False
    return path.suffix in TEXT_EXTENSIONS or path.name in {
        "Dockerfile",
        ".dockerignore",
        ".gitignore",
    }


def is_allowlisted(line: str) -> bool:
    return any(pattern.search(line) for pattern in ALLOWLIST_PATTERNS)


def scan(root: Path, include_env: bool = False) -> list[str]:
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or not should_scan(path, include_env):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if is_allowlisted(line):
                continue
            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path}:{line_no}: {name}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan repository text files for secrets"
    )
    parser.add_argument("--include-env", action="store_true", help="Also scan .env")
    args = parser.parse_args()

    findings = scan(Path.cwd(), include_env=args.include_env)
    if findings:
        print("Potential secrets found:")
        for finding in findings:
            print(f"  {finding}")
        return 1
    print("No potential secrets found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
