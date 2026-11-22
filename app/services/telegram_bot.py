"""Telegram command/callback handlers built on the existing Meas Pro services."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

from app import config
from app.services import chart_annotations, data, db, price_service, signal_engine
from app.services import telegram_client as tg
from app.services import telegram_text as txt

logger = logging.getLogger(__name__)

COMMAND_LIMIT = 10
COMMAND_WINDOW = 60.0
_command_hits: dict[int, deque[float]] = defaultdict(deque)

GRADES = ("A+", "A", "B", "C")
TIMEFRAMES = ("15m", "1h", "4h", "1D")


def _rate_limited(chat_id: int) -> bool:
    hits = _command_hits[chat_id]
    now = time.monotonic()
    while hits and hits[0] <= now - COMMAND_WINDOW:
        hits.popleft()
    if len(hits) >= COMMAND_LIMIT:
        return True
    hits.append(now)
    return False


def _is_admin(chat_id: int) -> bool:
    return chat_id in config.ADMIN_CHAT_IDS


def _user(chat_id: int) -> dict[str, Any] | None:
    return db.tg_get_user(chat_id)


def send(
    chat_id: int, message: str, *, markup: dict[str, Any] | None = None,
    reply_to: int | None = None,
    on_success: Any = None,
) -> Any:
    params: dict[str, Any] = {"chat_id": chat_id, "text": message}
    if markup:
        params["reply_markup"] = markup
    if reply_to:
        params["reply_to_message_id"] = reply_to
    job = tg.DeliveryJob(
        chat_id=chat_id, method="sendMessage", params=params, on_success=on_success,
    )
    return tg.queue.submit(job)


def answer_callback(callback_id: str, message: str | None = None) -> Any:
    return tg.queue.submit(tg.DeliveryJob(
        chat_id=0, method="answerCallbackQuery",
        params={"callback_query_id": callback_id, **({"text": message[:200]} if message else {})},
    ))


def _language(chat_id: int) -> str:
    return txt.lang_of(_user(chat_id))


def _reply(chat_id: int, key: str, **kwargs: Any) -> None:
    language = _language(chat_id)
    send(chat_id, txt.with_disclaimer(txt.tr(language, key, **kwargs), language))


def _latest_signal() -> dict[str, Any] | None:
    rows = db.list_signals(limit=1)
    return rows[0] if rows else None


def _live_signal() -> dict[str, Any] | None:
    try:
        entry = price_service.get_series("1h")
        frame = entry.df.iloc[:-1] if len(entry.df) > 2 else entry.df
        signal = signal_engine.generate_signal(
            frame, "1h", train_model=False, use_concepts=False,
        )
        return signal.to_dict()
    except Exception:  # noqa: BLE001 - fall back to stored data
        logger.exception("Could not build live signal for Telegram /signal")
        return _latest_signal()


def _price_text(chat_id: int) -> str:
    language = _language(chat_id)
    try:
        entry = price_service.get_series("1h")
        price = float(entry.df.iloc[-1]["close"])
        return txt.with_disclaimer(
            txt.tr(language, "price", price=txt.money(price)), language,
        )
    except Exception:  # noqa: BLE001
        return txt.with_disclaimer(txt.tr(language, "price_unavailable"), language)


def _settings_text(chat_id: int, user: dict[str, Any]) -> str:
    language = txt.lang_of(user)
    return txt.with_disclaimer(txt.tr(
        language, "settings_title",
        grade=txt.esc(user.get("min_grade") or "A"),
        timeframes=txt.esc(user.get("timeframes") or ""),
        language="English" if language == "en" else "ខ្មែរ",
        status="active" if user.get("is_active") else "paused",
    ), language)


def _send_settings(chat_id: int) -> None:
    user = _user(chat_id)
    if user is None:
        db.tg_upsert_user(chat_id, None, None)
        user = _user(chat_id) or {}
    send(
        chat_id, _settings_text(chat_id, user),
        markup=txt.settings_markup(user, txt.lang_of(user)),
    )


def _history(chat_id: int) -> None:
    language = _language(chat_id)
    rows = db.list_signals(limit=10)
    if not rows:
        _reply(chat_id, "history_empty")
        return
    lines = [txt.tr(language, "history", count=len(rows))]
    for row in rows:
        result = str(row.get("hit") or row.get("status") or "open")
        lines.append(
            f"{txt.esc(row.get('timeframe'))} {txt.esc(row.get('action'))} "
            f"({txt.esc(row.get('grade'))}) @ {txt.money(row.get('entry'))} - {txt.esc(result)}"
        )
    send(chat_id, txt.with_disclaimer("\n".join(lines), language))


def _stats(chat_id: int) -> None:
    language = _language(chat_id)
    stats = db.signal_stats()
    rows = db.list_signals(limit=2000)
    gains = sum(
        float(row.get("result_pct") or 0) for row in rows
        if float(row.get("result_pct") or 0) > 0
    )
    losses = abs(sum(
        float(row.get("result_pct") or 0) for row in rows
        if float(row.get("result_pct") or 0) < 0
    ))
    factor = gains / losses if losses else 0.0
    send(chat_id, txt.with_disclaimer(txt.tr(
        language, "stats",
        total=stats["total"], closed=stats["closed"],
        win_rate=stats["win_rate"], profit_factor=txt.money(factor),
    ), language))


def _levels(chat_id: int) -> None:
    language = _language(chat_id)
    timeframe = str((_user(chat_id) or {}).get("timeframes") or "15m,1h,4h").split(",")[0]
    try:
        entry = price_service.get_series(timeframe)
        payload = chart_annotations.build_payload(entry, timeframe)
        annotations = payload.get("annotations") or []
        lines = []
        for item in annotations[:10]:
            label = str(item.get("label") or item.get("type") or "level")
            price = item.get("price") or item.get("top") or item.get("bottom")
            lines.append(f"• {txt.esc(label)}: {txt.money(price)}")
        body = txt.tr(language, "levels", tf=txt.esc(timeframe),
                      levels="\n".join(lines) if lines else txt.tr(language, "levels_empty"))
    except Exception:  # noqa: BLE001
        body = txt.tr(language, "levels_empty")
    send(chat_id, txt.with_disclaimer(body, language))


def _admin_status(chat_id: int) -> None:
    language = _language(chat_id)
    webhook = "configured" if config.PRODUCTION else "disabled (polling)"
    last = _latest_signal() or {}
    send(chat_id, txt.with_disclaimer(txt.tr(
        language, "status",
        username=txt.esc(tg.client.username or "unknown"),
        webhook=txt.esc(webhook),
        count=db.tg_count_users(active_only=True),
        last_signal=txt.esc(last.get("signal_time") or last.get("created_at") or "none"),
        feed=txt.esc(str(data.data_source_status()) if hasattr(data, "data_source_status") else "ok"),
        errors=0,
    ), language))


def _test_signal(chat_id: int) -> None:
    language = _language(chat_id)
    sample = {
        "symbol": "XAU/USD", "action": "BUY", "grade": "A", "timeframe": "1h",
        "entry": 3521.40, "stop_loss": 3508.90, "take_profit1": 3540.0,
        "take_profit2": 3556.5, "take_profit3": 3575.0,
        "risk_reward": 2.4, "confidence": 68.0, "id": 0,
        "explanation": {
            "agreeing": ["4h bias bullish", "Price in discount zone"],
            "disagreeing": ["DXY rising"], "invalidation": "1h candle closes below 3,508.90",
        },
    }
    send(chat_id, txt.format_signal(sample, language))


def handle_message(update: dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message") or {}
    chat = message.get("chat") or {}
    if chat.get("type") != "private":
        return
    sender = message.get("from") or {}
    chat_id = int(chat.get("id") or sender.get("id") or 0)
    if not chat_id or _rate_limited(chat_id):
        return
    text = str(message.get("text") or "").strip()
    if not text.startswith("/"):
        return
    parts = text.split(maxsplit=1)
    command = parts[0].split("@")[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command == "/start":
        db.tg_upsert_user(chat_id, sender.get("username"), sender.get("first_name"))
        if argument:
            linked = db.tg_consume_link_code(argument, chat_id)
            if linked is None:
                _reply(chat_id, "link_invalid")
                return
            db.tg_update_user(chat_id, user_id=linked)
            _reply(chat_id, "linked")
        language = _language(chat_id)
        send(chat_id, txt.with_disclaimer(txt.tr(
            language, "welcome", name=txt.esc(sender.get("first_name") or "trader"),
        ), language))
    elif command == "/stop":
        db.tg_set_active(chat_id, False)
        _reply(chat_id, "stopped")
    elif command == "/help":
        _reply(chat_id, "help")
    elif command == "/lang":
        user = _user(chat_id) or {}
        language = "en" if user.get("language") == "km" else "km"
        db.tg_upsert_user(chat_id, sender.get("username"), sender.get("first_name"))
        db.tg_update_user(chat_id, language=language)
        _reply(chat_id, "lang_changed")
    elif command == "/settings":
        db.tg_upsert_user(chat_id, sender.get("username"), sender.get("first_name"))
        _send_settings(chat_id)
    elif command == "/link":
        if not argument:
            _reply(chat_id, "link_invalid")
            return
        linked = db.tg_consume_link_code(argument, chat_id)
        if linked is None:
            _reply(chat_id, "link_invalid")
            return
        db.tg_upsert_user(chat_id, sender.get("username"), sender.get("first_name"))
        db.tg_update_user(chat_id, user_id=linked)
        _reply(chat_id, "linked")
    elif command == "/signal":
        signal = _live_signal()
        if signal is None:
            _reply(chat_id, "no_signal")
        else:
            send(chat_id, txt.format_signal(signal, _language(chat_id)))
    elif command == "/price":
        send(chat_id, _price_text(chat_id))
    elif command == "/history":
        _history(chat_id)
    elif command == "/stats":
        _stats(chat_id)
    elif command == "/levels":
        _levels(chat_id)
    elif command == "/broadcast":
        if not _is_admin(chat_id):
            _reply(chat_id, "not_admin")
        elif not argument:
            _reply(chat_id, "broadcast_usage")
        else:
            users = db.tg_list_active_users()
            for user in users:
                send(int(user["chat_id"]), txt.with_disclaimer(
                    txt.esc(argument), txt.lang_of(user),
                ))
            _reply(chat_id, "broadcast_done", count=len(users))
    elif command == "/users":
        if not _is_admin(chat_id):
            _reply(chat_id, "not_admin")
        else:
            _reply(chat_id, "users_count", count=db.tg_count_users(active_only=True))
    elif command == "/status":
        if not _is_admin(chat_id):
            _reply(chat_id, "not_admin")
        else:
            _admin_status(chat_id)
    elif command == "/testsignal":
        if not _is_admin(chat_id):
            _reply(chat_id, "not_admin")
        else:
            _test_signal(chat_id)
    else:
        _reply(chat_id, "unknown")


def handle_callback(update: dict[str, Any]) -> None:
    query = update.get("callback_query") or {}
    callback_id = str(query.get("id") or "")
    sender = query.get("from") or {}
    chat_id = int((query.get("message") or {}).get("chat", {}).get("id") or sender.get("id") or 0)
    if not chat_id or _rate_limited(chat_id):
        if callback_id:
            answer_callback(callback_id)
        return
    user = _user(chat_id)
    if user is None:
        if callback_id:
            answer_callback(callback_id)
        return
    data = str(query.get("data") or "")
    parts = data.split(":")
    language = txt.lang_of(user)
    if len(parts) >= 3 and parts[0] == "tg" and parts[1] == "g" and parts[2] in GRADES:
        db.tg_update_user(chat_id, min_grade=parts[2])
    elif len(parts) >= 3 and parts[0] == "tg" and parts[1] == "t" and parts[2] in TIMEFRAMES:
        current = {value for value in str(user.get("timeframes") or "").split(",") if value}
        if parts[2] in current:
            current.discard(parts[2])
        else:
            current.add(parts[2])
        db.tg_update_user(chat_id, timeframes=",".join(
            tf for tf in TIMEFRAMES if tf in current
        ) or "15m")
    elif data == "tg:lang":
        db.tg_update_user(chat_id, language="km" if language == "en" else "en")
    elif data == "tg:pause":
        db.tg_set_active(chat_id, not bool(user.get("is_active")))
    if callback_id:
        answer_callback(callback_id)
    _send_settings(chat_id)


def handle_update(update: dict[str, Any]) -> None:
    try:
        if not isinstance(update, dict):
            return
        if "callback_query" in update:
            handle_callback(update)
        elif "message" in update or "edited_message" in update:
            handle_message(update)
    except Exception:  # noqa: BLE001 - a bad update must never crash the API
        logger.exception("Telegram update handling failed")
