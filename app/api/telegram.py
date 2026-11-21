"""Telegram webhook, connection, and admin/status endpoints."""
from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app import config
from app.services import auth, db
from app.services import telegram_client as tg
from app.services import telegram_delivery, telegram_text as txt
from app.services.telegram_bot import _test_signal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telegram", tags=["telegram"])

ALLOWED_UPDATES = {"message", "callback_query"}


def _verify_webhook(secret: str, request: Request) -> None:
    header = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    expected = config.TELEGRAM_WEBHOOK_SECRET
    if not expected or not secrets.compare_digest(secret, expected):
        raise HTTPException(403, "Invalid webhook secret")
    if not header or not secrets.compare_digest(header, expected):
        raise HTTPException(403, "Invalid webhook secret token")


@router.post("/webhook/{secret}")
async def webhook(secret: str, request: Request) -> dict[str, bool]:
    """Receive Telegram updates. Telegrams are authenticated by path + header secret."""
    _verify_webhook(secret, request)
    try:
        update = await request.json()
    except ValueError:
        raise HTTPException(400, "Invalid JSON") from None
    if not isinstance(update, dict):
        raise HTTPException(400, "Invalid update")
    if any(key in update for key in ALLOWED_UPDATES):
        from app.services import telegram_bot as bot
        bot.handle_update(update)
    return {"ok": True}


@router.get("/status")
def status(user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    """Admin-only bot diagnostics (username/webhook state/count/last delivery)."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Administrator access required")
    return {
        "enabled": tg.client.configured,
        "username": tg.client.username,
        "webhook": config.PRODUCTION,
        "subscribers": db.tg_count_users(active_only=True),
        "last_delivery_at": db.tg_last_delivery_at(),
        "queue_size": tg.queue.qsize(),
        "errors": telegram_delivery.recent_errors(),
        "disclaimer": txt.DISCLAIMER,
    }


@router.post("/link-code")
def link_code(user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, str]:
    """Create a one-time code that links a Telegram chat to this website account."""
    if not tg.client.configured:
        raise HTTPException(503, "Telegram bot is not configured")
    code = secrets.token_urlsafe(12)
    db.tg_create_link_code(code, int(user["id"]), config.TELEGRAM_LINK_CODE_TTL_SECONDS)
    username = tg.client.username or "GoldSightBot"
    return {
        "code": code,
        "deep_link": f"https://t.me/{username}?start={code}",
        "expires_in": config.TELEGRAM_LINK_CODE_TTL_SECONDS,
    }


@router.get("/connection")
def connection(user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, Any]:
    linked = db.tg_get_user_by_account(int(user["id"]))
    return {
        "connected": linked is not None,
        "chat_id": linked["chat_id"] if linked else None,
        "language": (linked or {}).get("language") or "en",
        "is_active": bool((linked or {}).get("is_active")),
    }


@router.post("/settings")
def update_settings(
    body: dict[str, Any], user: dict[str, Any] = Depends(auth.current_user),
) -> dict[str, str]:
    """Update Telegram preferences for the signed-in user's linked chat."""
    linked = db.tg_get_user_by_account(int(user["id"]))
    if linked is None:
        raise HTTPException(400, "Connect Telegram first")
    fields: dict[str, Any] = {}
    if body.get("min_grade") in {"A+", "A", "B", "C"}:
        fields["min_grade"] = body["min_grade"]
    if body.get("language") in {"en", "km"}:
        fields["language"] = body["language"]
    if isinstance(body.get("timeframes"), list):
        valid = [tf for tf in body["timeframes"] if tf in {"15m", "1h", "4h", "1D"}]
        fields["timeframes"] = ",".join(valid) or "15m"
    if isinstance(body.get("is_active"), bool):
        fields["is_active"] = int(body["is_active"])
    if fields:
        db.tg_update_user(int(linked["chat_id"]), **fields)
    return {"status": "ok"}


@router.post("/test")
def send_test(user: dict[str, Any] = Depends(auth.current_user)) -> dict[str, bool]:
    """Send a sample signal to the signed-in user's linked Telegram chat."""
    linked = db.tg_get_user_by_account(int(user["id"]))
    if linked is None:
        raise HTTPException(400, "Connect Telegram first")
    _test_signal(int(linked["chat_id"]))
    return {"sent": True}
