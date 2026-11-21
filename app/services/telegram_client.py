"""Async Telegram Bot API client and a rate-limited background delivery queue.

The bot token is read from ``config.TELEGRAM_BOT_TOKEN`` and is never logged:
only API method names and sanitised error descriptions are emitted.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from app import config
from app.services import db

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot"
COMMAND_TIMEOUT = 12.0
SEND_TIMEOUT = 20.0
MAX_ATTEMPTS = 3


class TelegramError(Exception):
    """Base error for Telegram API failures (message is always sanitised)."""


class TelegramNotConfigured(TelegramError):
    pass


class TelegramUnauthorized(TelegramError):
    pass


class TelegramForbidden(TelegramError):
    pass


class TelegramRateLimited(TelegramError):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited (retry_after={retry_after:.1f}s)")
        self.retry_after = max(0.0, retry_after)


@dataclass
class DeliveryJob:
    chat_id: int
    method: str
    params: dict[str, Any]
    attempts: int = 0
    on_success: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    on_forbidden: Callable[[int], Awaitable[None]] | None = None


class TelegramClient:
    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        self.disabled = False
        self.username = ""

    @property
    def configured(self) -> bool:
        return bool(config.TELEGRAM_BOT_TOKEN) and not self.disabled

    async def start(self) -> None:
        if not config.telegram_enabled():
            logger.info("Telegram bot disabled: TELEGRAM_BOT_TOKEN is not set")
            self.disabled = True
            return
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(SEND_TIMEOUT))
        self.disabled = False
        try:
            me = await self.call("getMe")
            self.username = str(me.get("username") or "")
            logger.info("Telegram bot connected as @%s", self.username or "unknown")
        except TelegramError as exc:
            logger.warning("Telegram getMe failed: %s", exc)

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def call(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None,
    ) -> dict[str, Any]:
        if not config.telegram_enabled():
            raise TelegramNotConfigured("bot token is not configured")
        if self.disabled:
            raise TelegramNotConfigured("bot is disabled")
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(SEND_TIMEOUT))
        url = f"{API_BASE}{config.TELEGRAM_BOT_TOKEN}/{method}"
        try:
            response = await self._http.post(
                url, json=params or {}, timeout=timeout or COMMAND_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise TelegramError(f"{method}: network error ({type(exc).__name__})") from None
        if response.status_code == 401:
            self.disabled = True
            logger.error(
                "Telegram API rejected the bot token (401); disabling bot. "
                "Admin chat IDs: %s", list(config.ADMIN_CHAT_IDS),
            )
            raise TelegramUnauthorized(f"{method}: unauthorized (bot disabled)")
        if response.status_code == 429:
            try:
                retry_after = float(
                    (response.json().get("parameters") or {}).get("retry_after", 1.0)
                )
            except (ValueError, AttributeError):
                retry_after = 1.0
            raise TelegramRateLimited(retry_after)
        try:
            payload = response.json()
        except ValueError:
            raise TelegramError(f"{method}: invalid response") from None
        if not response.is_success or not payload.get("ok"):
            description = str(payload.get("description") or response.status_code)
            lowered = description.lower()
            if response.status_code == 403 or "chat not found" in lowered or "blocked" in lowered:
                raise TelegramForbidden(f"{method}: {description}")
            raise TelegramError(f"{method}: {description[:160]}")
        result = payload.get("result")
        return result if isinstance(result, dict) else {"value": result}

    async def get_me(self) -> dict[str, Any]:
        return await self.call("getMe")

    async def set_my_commands(self, commands: list[dict[str, str]]) -> None:
        await self.call("setMyCommands", {"commands": commands})

    async def set_webhook(
        self, url: str, secret_token: str, allowed_updates: list[str],
    ) -> None:
        await self.call("setWebhook", {
            "url": url,
            "secret_token": secret_token,
            "allowed_updates": allowed_updates,
            "drop_pending_updates": False,
        })

    async def delete_webhook(self) -> None:
        await self.call("deleteWebhook", {"drop_pending_updates": False})

    async def get_webhook_info(self) -> dict[str, Any]:
        return await self.call("getWebhookInfo")

    async def get_updates(
        self, offset: int, timeout: int = 10, allowed_updates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        result = await self.call("getUpdates", {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": allowed_updates or ["message", "callback_query"],
        }, timeout=timeout + 5)
        return list(result.get("value") or [])

    async def send_message(
        self, chat_id: int, text: str, *, reply_to_message_id: int | None = None,
        reply_markup: dict[str, Any] | None = None, parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        if reply_markup:
            params["reply_markup"] = reply_markup
        return await self.call("sendMessage", params)

    async def send_photo(
        self, chat_id: int, photo: str, caption: str = "", *,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "chat_id": chat_id, "photo": photo, "caption": caption,
            "parse_mode": "HTML",
        }
        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id
        return await self.call("sendPhoto", params)

    async def answer_callback_query(
        self, callback_id: str, text: str | None = None,
    ) -> None:
        params: dict[str, Any] = {"callback_query_id": callback_id}
        if text:
            params["text"] = text[:200]
        await self.call("answerCallbackQuery", params)


class DeliveryQueue:
    """Single-worker queue enforcing global and per-chat send rates."""

    def __init__(self, client: TelegramClient) -> None:
        self.client = client
        self._queue: asyncio.Queue[DeliveryJob] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sent_at: deque[float] = deque()
        self._chat_sent_at: dict[int, float] = {}

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is None or (self._loop is not None and self._loop is not loop):
            self._queue = asyncio.Queue()
            self._sent_at.clear()
            self._chat_sent_at.clear()
        self._loop = loop
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker(), name="goldsight-telegram-queue")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._loop = None

    def submit(self, job: DeliveryJob) -> bool:
        if not self.client.configured:
            return False
        self._queue.put_nowait(job)
        return True

    def qsize(self) -> int:
        return self._queue.qsize()

    async def _throttle(self, chat_id: int) -> None:
        now = time.monotonic()
        last_chat = self._chat_sent_at.get(chat_id, 0.0)
        wait_chat = max(0.0, last_chat + config.TELEGRAM_PER_CHAT_SECONDS - now)
        cutoff = now - 1.0
        while self._sent_at and self._sent_at[0] <= cutoff:
            self._sent_at.popleft()
        wait_global = 0.0
        if len(self._sent_at) >= config.TELEGRAM_RATE_PER_SECOND:
            wait_global = max(0.0, self._sent_at[0] + 1.0 - now)
        wait = max(wait_global, wait_chat)
        if wait > 0:
            await asyncio.sleep(wait)
        now = time.monotonic()
        self._sent_at.append(now)
        self._chat_sent_at[chat_id] = now

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bad job must not kill the queue
                logger.exception("Telegram delivery worker error for method %s", job.method)
            finally:
                self._queue.task_done()

    async def _process(self, job: DeliveryJob) -> None:
        while job.attempts < MAX_ATTEMPTS:
            try:
                await self._throttle(job.chat_id)
                result = await self.client.call(job.method, job.params)
            except TelegramRateLimited as exc:
                job.attempts += 1
                if job.attempts >= MAX_ATTEMPTS:
                    logger.warning("Telegram %s still rate limited after %s attempts", job.method, MAX_ATTEMPTS)
                    return
                await asyncio.sleep(min(exc.retry_after, 30.0))
                continue
            except TelegramForbidden as exc:
                logger.info("Telegram %s: %s", job.method, exc)
                db.tg_set_active(job.chat_id, False)
                if job.on_forbidden is not None:
                    await job.on_forbidden(job.chat_id)
                return
            except TelegramNotConfigured:
                return
            except TelegramUnauthorized:
                return
            except TelegramError as exc:
                job.attempts += 1
                if job.attempts >= MAX_ATTEMPTS:
                    logger.warning("Telegram %s failed after %s attempts: %s", job.method, MAX_ATTEMPTS, exc)
                    return
                await asyncio.sleep(min(2.0 ** job.attempts, 8.0))
                continue
            if job.on_success is not None:
                await job.on_success(result)
            return


client = TelegramClient()
queue = DeliveryQueue(client)
