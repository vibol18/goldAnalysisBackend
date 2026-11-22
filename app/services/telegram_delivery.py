"""Deliver signals/results from the existing pipeline to Telegram subscribers."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app import config
from app.services import data, db, price_service
from app.services import telegram_bot as bot
from app.services import telegram_client as tg
from app.services import telegram_text as txt

logger = logging.getLogger(__name__)

GRADE_RANK = {"C": 0, "B": 1, "A": 2, "A+": 3}
BOT_COMMANDS = [
    {"command": "start", "description": "Subscribe and open the menu"},
    {"command": "stop", "description": "Pause alerts"},
    {"command": "help", "description": "List all commands"},
    {"command": "signal", "description": "Latest signal"},
    {"command": "price", "description": "Live XAU/USD price"},
    {"command": "settings", "description": "Grade, timeframes, language"},
    {"command": "history", "description": "Last 10 signals with results"},
    {"command": "stats", "description": "Win rate and profit factor"},
    {"command": "levels", "description": "Key support/resistance"},
    {"command": "lang", "description": "Switch English/Khmer"},
    {"command": "link", "description": "Connect website account with a code"},
]

_polling_task: asyncio.Task[None] | None = None
_last_errors: list[str] = []


def _chart_url(signal_id: Any) -> str:
    base = config.PUBLIC_BASE_URL or "http://localhost:5173"
    return f"{base}/signals?signal={signal_id}"


def _in_quiet_hours(user: dict[str, Any]) -> bool:
    start = str(user.get("quiet_start") or "")
    end = str(user.get("quiet_end") or "")
    if not start or not end:
        return False
    now = datetime.now(timezone.utc).strftime("%H:%M")
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def _eligible(user: dict[str, Any], signal: dict[str, Any]) -> bool:
    if not user.get("is_active"):
        return False
    signal_grade = str(signal.get("grade") or "").strip()
    if signal_grade and GRADE_RANK.get(signal_grade, 0) < GRADE_RANK.get(
        str(user.get("min_grade") or "C"), 0,
    ):
        return False
    timeframes = {value for value in str(user.get("timeframes") or "").split(",") if value}
    if timeframes and str(signal.get("timeframe")) not in timeframes:
        return False
    symbols = {value for value in str(user.get("symbols") or "").split(",") if value}
    if symbols and str(signal.get("symbol") or "XAU/USD") not in symbols:
        return False
    if _in_quiet_hours(user):
        return False
    try:
        cap = int(user.get("daily_cap") or config.TELEGRAM_DAILY_CAP)
    except (TypeError, ValueError):
        cap = config.TELEGRAM_DAILY_CAP
    if db.tg_delivery_count_today(int(user["chat_id"])) >= cap:
        return False
    return True


def _message_id_handler(chat_id: int, signal_id: int, kind: str) -> Any:
    async def handler(result: dict[str, Any]) -> None:
        message_id = result.get("message_id")
        if message_id is not None:
            db.tg_set_message_id(chat_id, signal_id, int(message_id), kind)
    return handler


def dispatch_signal(signal: dict[str, Any]) -> int:
    """Fan out a newly generated BUY/SELL signal. Returns accepted users."""
    if not tg.client.configured:
        return 0
    action = str(signal.get("action") or "").upper()
    if action not in {"BUY", "SELL"}:
        return 0
    try:
        signal_id = int(signal.get("id") or 0)
    except (TypeError, ValueError):
        return 0
    if not signal_id:
        logger.warning("Telegram dispatch skipped: signal has no id")
        return 0
    sent = 0
    for user in db.tg_list_active_users():
        chat_id = int(user["chat_id"])
        if not _eligible(user, signal):
            continue
        if not db.tg_record_delivery(chat_id, signal_id, "signal"):
            continue
        language = txt.lang_of(user)
        params = {
            "chat_id": chat_id,
            "text": txt.format_signal(signal, language),
            "reply_markup": txt.chart_markup(_chart_url(signal_id), language),
        }
        job = tg.DeliveryJob(
            chat_id=chat_id, method="sendMessage", params=params,
            on_success=_message_id_handler(chat_id, signal_id, "signal"),
        )
        if tg.queue.submit(job):
            sent += 1
    return sent


def dispatch_result(
    kind: str, signal: dict[str, Any], *, hit: str | None = None,
) -> int:
    """Reply to the original signal message with a TP/SL/invalidation update."""
    if not tg.client.configured:
        return 0
    try:
        signal_id = int(signal.get("id") or 0)
    except (TypeError, ValueError):
        return 0
    sent = 0
    for user in db.tg_list_active_users():
        chat_id = int(user["chat_id"])
        original = db.tg_get_delivery(chat_id, signal_id, "signal")
        if original is None:
            continue
        if not db.tg_record_delivery(chat_id, signal_id, kind):
            continue
        language = txt.lang_of(user)
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": txt.format_result(kind, signal, language, hit=hit),
        }
        if original.get("message_id"):
            params["reply_to_message_id"] = int(original["message_id"])
        if tg.queue.submit(tg.DeliveryJob(chat_id=chat_id, method="sendMessage", params=params)):
            sent += 1
    return sent


def dispatch_catchup() -> int:
    """Send missed-but-still-valid signals after a Render wake-up."""
    if not config.TELEGRAM_CATCHUP_ENABLED or not tg.client.configured:
        return 0
    try:
        entry = price_service.get_series("1h")
        price = float(entry.df.iloc[-1]["close"])
    except Exception:  # noqa: BLE001
        return 0
    sent = 0
    for signal in db.list_signals(limit=50):
        if signal.get("status") == "closed" or signal.get("action") not in {"BUY", "SELL"}:
            continue
        entry_price = float(signal.get("entry") or 0)
        stop = float(signal.get("stop_loss") or 0)
        if not entry_price or not stop:
            continue
        low_band, high_band = sorted((entry_price, stop))
        if not (low_band * 0.995 <= price <= high_band * 1.005):
            continue
        signal_id = int(signal["id"])
        for user in db.tg_list_active_users():
            chat_id = int(user["chat_id"])
            if not db.tg_record_delivery(chat_id, signal_id, "signal"):
                continue
            language = txt.lang_of(user)
            params = {
                "chat_id": chat_id,
                "text": txt.format_signal(signal, language, delayed=True),
                "reply_markup": txt.chart_markup(_chart_url(signal_id), language),
            }
            if tg.queue.submit(tg.DeliveryJob(
                chat_id=chat_id, method="sendMessage", params=params,
                on_success=_message_id_handler(chat_id, signal_id, "signal"),
            )):
                sent += 1
    return sent


async def configure_transport() -> None:
    """Start polling locally or register the verified webhook on Render."""
    if not tg.client.configured:
        return
    global _polling_task
    try:
        await tg.client.set_my_commands(BOT_COMMANDS)
        if config.PRODUCTION:
            url = f"{config.PUBLIC_BASE_URL}/api/telegram/webhook/{config.TELEGRAM_WEBHOOK_SECRET}"
            await tg.client.set_webhook(
                url, config.TELEGRAM_WEBHOOK_SECRET, ["message", "callback_query"],
            )
            logger.info("Telegram webhook registered")
        else:
            await tg.client.delete_webhook()
            await tg.queue.start()
            dispatch_catchup()
            _polling_task = asyncio.create_task(_polling_loop(), name="goldsight-telegram-poller")
            logger.info("Telegram long polling started")
    except tg.TelegramError as exc:
        logger.warning("Telegram transport setup failed: %s", exc)


async def _polling_loop() -> None:
    offset_raw = db.get_app_setting("telegram_update_offset")
    offset = int(offset_raw or 0)
    while True:
        try:
            updates = await tg.client.get_updates(
                offset, timeout=10, allowed_updates=["message", "callback_query"],
            )
            for update in updates:
                update_id = int(update.get("update_id") or 0)
                offset = max(offset, update_id + 1)
                bot.handle_update(update)
            if updates:
                db.set_app_setting("telegram_update_offset", str(offset))
        except asyncio.CancelledError:
            raise
        except tg.TelegramNotConfigured:
            await asyncio.sleep(30)
        except tg.TelegramError as exc:
            _record_error(str(exc))
            await asyncio.sleep(5)


def _record_error(message: str) -> None:
    _last_errors.append(message[:160])
    del _last_errors[:-5]


def recent_errors() -> list[str]:
    return list(_last_errors)


def notify_admin(message: str) -> int:
    """Send an operational alert to configured admin chats (no-op if unset)."""
    sent = 0
    for chat_id in config.ADMIN_CHAT_IDS:
        if bot.send(int(chat_id), message[:900]):
            sent += 1
    return sent


async def stop() -> None:
    global _polling_task
    if _polling_task is not None:
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass
        _polling_task = None
    await tg.queue.stop()
