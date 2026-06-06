import pytest
from pydantic import ValidationError

from src.config import Settings


def test_telegram_command_settings_parse_authorized_users():
    cfg = Settings(
        _env_file=None,
        telegram_bot_token="token",
        telegram_chat_id="-100123",
        telegram_commands_enabled=True,
        telegram_allowed_user_ids="42, 84",
    )

    assert cfg.telegram_commands_enabled
    assert cfg.telegram_allowed_user_ids_list == ["42", "84"]


def test_telegram_commands_require_credentials():
    with pytest.raises(ValidationError, match="TELEGRAM_COMMANDS_ENABLED"):
        Settings(_env_file=None, telegram_commands_enabled=True)


def test_telegram_authorized_users_must_be_numeric():
    with pytest.raises(ValidationError, match="numeric IDs"):
        Settings(
            _env_file=None,
            telegram_allowed_user_ids="operator",
        )


def test_group_telegram_commands_require_user_allowlist():
    with pytest.raises(ValidationError, match="Group Telegram commands"):
        Settings(
            _env_file=None,
            telegram_bot_token="token",
            telegram_chat_id="-100123",
            telegram_commands_enabled=True,
        )
