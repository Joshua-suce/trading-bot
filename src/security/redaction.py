import re
from collections.abc import Mapping, Sequence
from typing import Any

SENSITIVE_KEYWORDS = (
    "api_key",
    "apikey",
    "api_secret",
    "secret",
    "token",
    "webhook",
    "password",
    "passphrase",
    "private_key",
)

SECRET_PATTERNS = (
    (
        re.compile(r"((?:[?&])?signature=)[A-Za-z0-9_-]+", re.IGNORECASE),
        r"\1<redacted>",
    ),
    (re.compile(r"(https://api\.telegram\.org/bot)[^/\s]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"bot[0-9]{6,}:[A-Za-z0-9_-]{20,}", re.IGNORECASE), "bot***"),
    (
        re.compile(r"[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}"),
        "***",
    ),
    (
        re.compile(
            r"(secret|token|api[_-]?key|webhook)(['\"\s:=]+)[^\s,'\"]{8,}",
            re.IGNORECASE,
        ),
        r"\1\2***",
    ),
)


def is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(keyword in normalized for keyword in SENSITIVE_KEYWORDS)


def redact_text(value: object) -> str:
    text = str(value)
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "***" if is_sensitive_key(key) else redact_mapping(item)
            for key, item in value.items()
        }
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact_mapping(item) for item in value]
    return value
