# Logging setup — configures loguru for console + daily rotating files
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings
from src.security import redact_text


def _redact_record(record: Any) -> None:
    record["message"] = redact_text(record["message"])


# Remove default handler, add coloured console and two file sinks
def setup_logging(level: str | None = None):
    log_level = level or settings.log_level
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.configure(patcher=_redact_record)

    # Console with colourised output
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
            "<cyan>{name}</cyan> | {message}"
        ),
    )

    # File (rotated)
    logger.add(
        log_dir / "trading_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | "
            "{name}:{function}:{line} | {message}"
        ),
    )

    # Error file
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="1 day",
        retention="90 days",
    )
    if settings.json_logs:
        logger.add(
            log_dir / "trading_{time:YYYY-MM-DD}.jsonl",
            level="DEBUG",
            rotation="1 day",
            retention="30 days",
            serialize=True,
        )
    return logger
