import asyncio
from typing import Optional

import httpx
from loguru import logger

from src.security import redact_text


class Alerter:
    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook: str = "",
    ):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook

    async def send(self, message: str, level: str = "info"):
        logger.info(f"Alert [{level}]: {message}")
        if self.telegram_token and self.telegram_chat_id:
            await self._send_telegram(message)
        if self.discord_webhook:
            await self._send_discord(message)

    async def _send_telegram(self, message: str):
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        await self._post_with_retries(
            "Telegram",
            url,
            {"chat_id": self.telegram_chat_id, "text": message},
        )

    async def _send_discord(self, message: str):
        await self._post_with_retries(
            "Discord",
            self.discord_webhook,
            {"content": f"```\n{message}\n```"},
        )

    async def _post_with_retries(
        self, channel: str, url: str, payload: dict, attempts: int = 3
    ) -> bool:
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                return True
            except Exception as e:
                last_error = self._redact(str(e))
                logger.warning(
                    f"{channel} alert failed attempt {attempt}/{attempts}: "
                    f"{last_error}"
                )
                if attempt < attempts:
                    await asyncio.sleep(0.5 * attempt)

        logger.error(f"{channel} alert failed permanently: {last_error}")
        return False

    def _redact(self, text: str) -> str:
        redacted = text
        if self.telegram_token:
            redacted = redacted.replace(self.telegram_token, "***")
        if self.discord_webhook:
            redacted = redacted.replace(self.discord_webhook, "***")
        return redact_text(redacted)

    async def startup_alert(self, mode: str, environment: str, symbols: list[str]):
        msg = (
            "*Trading Bot Started*\n"
            f"Mode: {mode.upper()}\n"
            f"Environment: {environment}\n"
            f"Symbols: {', '.join(symbols)}"
        )
        await self.send(msg, "startup")

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
            "*Trade Placed*\n"
            f"Mode: {mode.upper()}\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.upper()}\n"
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
        msg = (
            "*Trade Completed*\n"
            f"Mode: {mode.upper()}\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.upper()}\n"
            f"Exit: {exit_price:.2f}\n"
            f"PnL: {pnl_text}\n"
            f"Reason: {reason}"
        )
        await self.send(msg, "trade_completed")

    async def trade_failed_alert(self, mode: str, symbol: str, reason: str):
        msg = (
            "*Trade Failed*\n"
            f"Mode: {mode.upper()}\n"
            f"Symbol: {symbol}\n"
            f"Reason: {reason}"
        )
        await self.send(msg, "trade_failed")

    async def trade_alert(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
    ):
        msg = (
            f"*Trade Alert* | {symbol}\n"
            f"Side: {side}\n"
            f"Price: {price:.2f}\n"
            f"Qty: {quantity:.4f}"
        )
        if pnl is not None:
            msg += f"\nPnL: {pnl:.2f}"
        await self.send(msg, "trade")

    async def error_alert(self, error: str):
        await self.send(f"*Error*: {error}", "error")

    async def daily_summary(self, total_pnl: float, win_rate: float, trades: int):
        msg = (
            f"*Daily Summary*\n"
            f"PnL: {total_pnl:.2f}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Total Trades: {trades}"
        )
        await self.send(msg, "daily")
