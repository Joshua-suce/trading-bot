import asyncio
import html
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Optional

import httpx
from loguru import logger

from src.security import redact_text

TelegramCommandHandler = Callable[[str, str, str, int], Awaitable[str]]


class Alerter:
    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook: str = "",
        *,
        telegram_commands_enabled: bool = False,
        telegram_allowed_user_ids: Optional[list[str]] = None,
        telegram_poll_timeout: int = 20,
        queue_size: int = 100,
        delivery_timeout: float = 10.0,
    ):
        self.telegram_token = telegram_token
        self.telegram_chat_id = str(telegram_chat_id)
        self.discord_webhook = discord_webhook
        self.telegram_commands_enabled = telegram_commands_enabled
        self.telegram_allowed_user_ids = {
            str(user_id) for user_id in (telegram_allowed_user_ids or [])
        }
        self.telegram_poll_timeout = telegram_poll_timeout
        self.delivery_timeout = delivery_timeout
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=queue_size)
        self._client: httpx.AsyncClient | None = None
        self._worker_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None
        self._command_handler: TelegramCommandHandler | None = None
        self._running = False
        self._telegram_offset = 0
        self.delivery_successes = 0
        self.delivery_failures = 0
        self.dropped_messages = 0

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)

    @property
    def enabled(self) -> bool:
        return self.telegram_enabled or bool(self.discord_webhook)

    @property
    def pending_messages(self) -> int:
        return self._queue.qsize()

    async def start(
        self, command_handler: TelegramCommandHandler | None = None
    ) -> None:
        if self._running or not self.enabled:
            return
        self._running = True
        self._command_handler = command_handler
        self._client = httpx.AsyncClient(timeout=self.delivery_timeout)
        self._worker_task = asyncio.create_task(
            self._delivery_worker(), name="alert-delivery-worker"
        )
        if (
            self.telegram_enabled
            and self.telegram_commands_enabled
            and command_handler is not None
        ):
            self._poll_task = asyncio.create_task(
                self._poll_telegram_commands(), name="telegram-command-poller"
            )

    async def stop(self, *, flush_timeout: float = 15.0) -> None:
        if not self._running:
            return
        if self._poll_task:
            self._poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
        try:
            await asyncio.wait_for(self._queue.join(), timeout=flush_timeout)
        except TimeoutError:
            logger.error(
                "Alert queue did not drain before shutdown; "
                f"{self._queue.qsize()} message(s) remain"
            )
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(self, message: str, level: str = "info") -> bool:
        logger.info(f"Alert [{level}]: {message}")
        if not self.enabled:
            return False
        if not self._running:
            return await self._deliver(message)
        try:
            self._queue.put_nowait((message, level))
            return True
        except asyncio.QueueFull:
            self.dropped_messages += 1
            logger.error(
                f"Alert queue full; dropped {level} notification "
                f"(total dropped={self.dropped_messages})"
            )
            return False

    async def _delivery_worker(self) -> None:
        while True:
            message, _level = await self._queue.get()
            try:
                await self._deliver(message)
            except Exception as exc:
                self.delivery_failures += 1
                logger.error(
                    "Unexpected alert delivery error: " f"{self._redact(str(exc))}"
                )
            finally:
                self._queue.task_done()

    async def _deliver(self, message: str) -> bool:
        results = []
        if self.telegram_enabled:
            results.append(await self._send_telegram(message))
        if self.discord_webhook:
            results.append(await self._send_discord(message))
        delivered = bool(results) and all(results)
        if delivered:
            self.delivery_successes += 1
        else:
            self.delivery_failures += 1
        return delivered

    async def _send_telegram(self, message: str) -> bool:
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        response = await self._telegram_request("sendMessage", payload)
        return response is not None

    async def _send_discord(self, message: str) -> bool:
        plain_text = html.unescape(re.sub(r"<[^>]+>", "", message))
        return await self._post_with_retries(
            "Discord",
            self.discord_webhook,
            {"content": f"```\n{plain_text}\n```"},
        )

    async def _telegram_request(
        self, method: str, payload: dict[str, Any], attempts: int = 3
    ) -> dict[str, Any] | None:
        url = f"https://api.telegram.org/bot{self.telegram_token}/{method}"
        request_timeout = self.delivery_timeout
        if method == "getUpdates":
            request_timeout = max(
                self.delivery_timeout, self.telegram_poll_timeout + 5.0
            )
        response = await self._request_with_retries(
            "Telegram", url, payload, attempts, timeout=request_timeout
        )
        if response is None:
            return None
        try:
            body = response.json()
        except Exception as exc:
            logger.warning(f"Telegram returned invalid JSON: {self._redact(str(exc))}")
            return None
        if not body.get("ok"):
            logger.warning(
                "Telegram API rejected request: "
                f"{self._redact(str(body.get('description', 'unknown error')))}"
            )
            return None
        return body

    async def _post_with_retries(
        self, channel: str, url: str, payload: dict, attempts: int = 3
    ) -> bool:
        return (
            await self._request_with_retries(channel, url, payload, attempts)
            is not None
        )

    async def _request_with_retries(
        self,
        channel: str,
        url: str,
        payload: dict,
        attempts: int,
        *,
        timeout: float | None = None,
    ) -> httpx.Response | None:
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                if self._client:
                    response = await self._client.post(
                        url, json=payload, timeout=timeout or self.delivery_timeout
                    )
                else:
                    async with httpx.AsyncClient(
                        timeout=self.delivery_timeout
                    ) as client:
                        response = await client.post(
                            url, json=payload, timeout=timeout or self.delivery_timeout
                        )
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = self._redact(str(exc))
                logger.warning(
                    f"{channel} alert failed attempt {attempt}/{attempts}: "
                    f"{last_error}"
                )
                if attempt < attempts:
                    await asyncio.sleep(0.5 * attempt)
        logger.error(f"{channel} alert failed permanently: {last_error}")
        return None

    async def _poll_telegram_commands(self) -> None:
        while self._running:
            payload = {
                "offset": self._telegram_offset,
                "timeout": self.telegram_poll_timeout,
                "allowed_updates": ["message"],
            }
            response = await self._telegram_request("getUpdates", payload, attempts=1)
            if response is None:
                await asyncio.sleep(2)
                continue
            for update in response.get("result", []):
                update_id = int(update.get("update_id", 0))
                self._telegram_offset = max(self._telegram_offset, update_id + 1)
                try:
                    await self.process_telegram_update(update)
                except Exception as exc:
                    logger.error(
                        "Telegram command processing failed: "
                        f"{self._redact(str(exc))}"
                    )

    async def process_telegram_update(self, update: dict[str, Any]) -> bool:
        message = update.get("message") or {}
        chat_id = str((message.get("chat") or {}).get("id", ""))
        user_id = str((message.get("from") or {}).get("id", ""))
        text = str(message.get("text") or "").strip()
        if chat_id != self.telegram_chat_id:
            logger.warning(
                "Ignored Telegram command from unauthorized chat "
                f"{chat_id or 'unknown'}"
            )
            return False
        if (
            self.telegram_allowed_user_ids
            and user_id not in self.telegram_allowed_user_ids
        ):
            logger.warning(
                "Ignored Telegram command from unauthorized user "
                f"{user_id or 'unknown'}"
            )
            return False
        if not text.startswith("/") or self._command_handler is None:
            return False
        command_text, _, arguments = text.partition(" ")
        command = command_text.split("@", 1)[0].lower()
        update_id = int(update.get("update_id", 0))
        response = await self._command_handler(
            command, arguments.strip(), user_id, update_id
        )
        if response:
            await self.send(response, "telegram_command")
        return True

    def _redact(self, text: str) -> str:
        redacted = text
        if self.telegram_token:
            redacted = redacted.replace(self.telegram_token, "***")
        if self.discord_webhook:
            redacted = redacted.replace(self.discord_webhook, "***")
        return redact_text(redacted)

    async def initializing_alert(self, mode: str, environment: str):
        await self.send(
            "<b>Trading Bot Initializing</b>\n"
            f"Mode: {html.escape(mode.upper())}\n"
            f"Environment: {html.escape(environment)}",
            "startup",
        )

    async def startup_alert(self, mode: str, environment: str, symbols: list[str]):
        await self.send(
            "<b>Trading Bot Ready</b>\n"
            f"Mode: {html.escape(mode.upper())}\n"
            f"Environment: {html.escape(environment)}\n"
            f"Symbols: {html.escape(', '.join(symbols))}",
            "startup",
        )

    async def shutdown_alert(self, mode: str, environment: str, reason: str):
        await self.send(
            "<b>Trading Bot Stopped</b>\n"
            f"Mode: {html.escape(mode.upper())}\n"
            f"Environment: {html.escape(environment)}\n"
            f"Reason: {html.escape(reason)}",
            "shutdown",
        )

    async def trade_opened_alert(
        self,
        mode: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: Optional[float],
    ):
        msg = (
            "<b>Trade Placed</b>\n"
            f"Mode: {html.escape(mode.upper())}\n"
            f"Symbol: {html.escape(symbol)}\n"
            f"Side: {html.escape(side.upper())}\n"
            f"Price: {price:.2f}\n"
            f"Qty: {quantity:.6f}\n"
            f"Stop: {stop_loss:.2f}"
        )
        if take_profit is not None:
            msg += f"\nTarget: {take_profit:.2f}"
        await self.send(msg, "trade")

    async def trade_completed_alert(
        self,
        mode: str,
        symbol: str,
        side: str,
        exit_price: float,
        pnl: Optional[float],
        reason: str,
    ):
        pnl_text = "n/a" if pnl is None else f"{pnl:.2f}"
        await self.send(
            "<b>Trade Completed</b>\n"
            f"Mode: {html.escape(mode.upper())}\n"
            f"Symbol: {html.escape(symbol)}\n"
            f"Side: {html.escape(side.upper())}\n"
            f"Exit: {exit_price:.2f}\n"
            f"PnL: {pnl_text}\n"
            f"Reason: {html.escape(reason)}",
            "trade_completed",
        )

    async def trade_failed_alert(self, mode: str, symbol: str, reason: str):
        await self.send(
            "<b>Trade Failed</b>\n"
            f"Mode: {html.escape(mode.upper())}\n"
            f"Symbol: {html.escape(symbol)}\n"
            f"Reason: {html.escape(reason)}",
            "trade_failed",
        )

    async def trade_alert(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
    ):
        msg = (
            f"<b>Trade Alert</b> | {html.escape(symbol)}\n"
            f"Side: {html.escape(side)}\n"
            f"Price: {price:.2f}\n"
            f"Qty: {quantity:.4f}"
        )
        if pnl is not None:
            msg += f"\nPnL: {pnl:.2f}"
        await self.send(msg, "trade")

    async def error_alert(self, error: str):
        await self.send(f"<b>Error</b>: {html.escape(error)}", "error")

    async def daily_summary(self, total_pnl: float, win_rate: float, trades: int):
        await self.send(
            "<b>Daily Summary</b>\n"
            f"PnL: {total_pnl:.2f}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Total Trades: {trades}",
            "daily",
        )
