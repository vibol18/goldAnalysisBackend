"""Unit tests for the Telegram bot with a mocked Telegram API."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config
from app.api import telegram as telegram_api
from app.services import db
from app.services import telegram_bot as bot
from app.services import telegram_client as tg
from app.services import telegram_delivery as delivery
from app.services import telegram_text as txt

SIGNAL: dict[str, Any] = {
    "id": 7,
    "symbol": "XAU/USD",
    "action": "BUY",
    "grade": "A",
    "timeframe": "1h",
    "entry": 3521.4,
    "stop_loss": 3508.9,
    "take_profit1": 3540.0,
    "take_profit2": 3556.5,
    "take_profit3": 3575.0,
    "risk_reward": 2.4,
    "confidence": 68.0,
    "status": "open",
    "explanation": {
        "agreeing": ["4h bias <bullish>"],
        "disagreeing": ["DXY rising"],
        "invalidation": "close below 3508.90",
    },
}


@pytest.fixture()
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    url = f"sqlite:///{tmp_path / 'telegram.db'}"
    monkeypatch.setattr(config, "DATABASE_URL", url)
    monkeypatch.setattr(db, "engine", db.create_database_engine(url))
    monkeypatch.setattr(db, "_initialized", False)
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN")
    monkeypatch.setattr(config, "TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setattr(config, "ADMIN_CHAT_IDS", (99,))
    tg.client.disabled = False
    bot._command_hits.clear()
    db.init_db()
    yield
    db.close_db()


def test_signal_format_escapes_html_and_supports_khmer() -> None:
    english = txt.format_signal(SIGNAL, "en")
    khmer = txt.format_signal(SIGNAL, "km")
    assert "&lt;bullish&gt;" in english
    assert "<bullish>" not in english
    assert "Grade A" in english
    assert txt.DISCLAIMER in english
    assert txt.TEXT["km"]["disclaimer"] in khmer
    assert "ថ្នាក់" in khmer


def test_dispatch_signal_filters_and_is_idempotent(
    isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs: list[tg.DeliveryJob] = []
    monkeypatch.setattr(tg.queue, "submit", lambda job: jobs.append(job) or True)

    db.tg_upsert_user(11, "alice", "Alice")
    db.tg_update_user(11, min_grade="A+")
    assert delivery.dispatch_signal(SIGNAL) == 0

    db.tg_update_user(11, min_grade="A")
    assert delivery.dispatch_signal(SIGNAL) == 1
    assert delivery.dispatch_signal(SIGNAL) == 0
    assert len(jobs) == 1
    assert "Open chart" in jobs[0].params["reply_markup"]["inline_keyboard"][0][0]["text"]


def test_default_subscriber_gets_every_signal_grade(
    isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs: list[tg.DeliveryJob] = []
    monkeypatch.setattr(tg.queue, "submit", lambda job: jobs.append(job) or True)
    db.tg_upsert_user(21, "carol", "Carol")
    assert db.tg_get_user(21)["min_grade"] == "C"

    for index, grade in enumerate([None, "C", "B", "A", "A+"]):
        assert delivery.dispatch_signal(dict(SIGNAL, id=200 + index, grade=grade)) == 1

    assert len(jobs) == 5


def test_dispatch_signal_honours_timeframe_and_pause(
    isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs: list[tg.DeliveryJob] = []
    monkeypatch.setattr(tg.queue, "submit", lambda job: jobs.append(job) or True)
    db.tg_upsert_user(12, "bob", "Bob")
    db.tg_update_user(12, timeframes="15m,4h")
    assert delivery.dispatch_signal(SIGNAL) == 0
    db.tg_update_user(12, timeframes="1h")
    db.tg_set_active(12, False)
    assert delivery.dispatch_signal(SIGNAL) == 0
    db.tg_set_active(12, True)
    assert delivery.dispatch_signal(SIGNAL) == 1


def test_result_replies_to_original_message(
    isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs: list[tg.DeliveryJob] = []
    monkeypatch.setattr(tg.queue, "submit", lambda job: jobs.append(job) or True)
    db.tg_upsert_user(13, "carol", "Carol")
    delivery.dispatch_signal(SIGNAL)
    db.tg_set_message_id(13, 7, 555, "signal")
    assert delivery.dispatch_result("tp", SIGNAL, hit="1") == 1
    assert jobs[-1].params["reply_to_message_id"] == 555
    assert delivery.dispatch_result("tp", SIGNAL, hit="1") == 0


def test_link_code_expires_and_works_once(isolated: Any) -> None:
    db.tg_upsert_user(21, "dana", "Dana")
    db.tg_create_link_code("abc123", 42, 600)
    assert db.tg_consume_link_code("abc123", 21) == 42
    assert db.tg_consume_link_code("abc123", 21) is None
    assert db.tg_get_user(21)["user_id"] == 42


def test_start_stop_and_group_ignore(isolated: Any) -> None:
    bot.handle_message({
        "update_id": 1,
        "message": {
            "chat": {"id": 31, "type": "private"},
            "from": {"id": 31, "username": "eve", "first_name": "Eve"},
            "text": "/start",
        },
    })
    assert db.tg_get_user(31)["is_active"] == 1
    bot.handle_message({
        "message": {
            "chat": {"id": 31, "type": "private"},
            "from": {"id": 31}, "text": "/stop",
        },
    })
    assert db.tg_get_user(31)["is_active"] == 0
    bot.handle_message({
        "message": {
            "chat": {"id": -100, "type": "group"},
            "from": {"id": 31}, "text": "/start",
        },
    })
    assert db.tg_get_user(-100) is None


@pytest.mark.asyncio()
async def test_queue_retries_429_and_deactivates_on_403(
    isolated: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.tg_upsert_user(41, "frank", "Frank")
    attempts = 0

    async def flaky(method: str, params: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise tg.TelegramRateLimited(0.01)
        return {"message_id": 1}

    monkeypatch.setattr(tg.client, "call", flaky)
    queue = tg.DeliveryQueue(tg.client)
    await queue._process(tg.DeliveryJob(41, "sendMessage", {"chat_id": 41}))
    assert attempts == 2

    async def forbidden(method: str, params: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        raise tg.TelegramForbidden("chat not found")

    monkeypatch.setattr(tg.client, "call", forbidden)
    await queue._process(tg.DeliveryJob(41, "sendMessage", {"chat_id": 41}))
    assert db.tg_get_user(41)["is_active"] == 0


@pytest.mark.asyncio()
async def test_webhook_verifies_secret(isolated: Any) -> None:
    app = FastAPI()
    app.include_router(telegram_api.router)
    with TestClient(app) as client:
        update = {"update_id": 1, "message": {"text": "/help"}}
        bad = client.post(
            "/api/telegram/webhook/test-secret", json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )
        assert bad.status_code == 403
        good = client.post(
            "/api/telegram/webhook/test-secret", json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "test-secret"},
        )
        assert good.status_code == 200


def test_command_rate_limit(isolated: Any) -> None:
    update = {
        "message": {
            "chat": {"id": 51, "type": "private"},
            "from": {"id": 51}, "text": "/help",
        },
    }
    for _ in range(10):
        bot.handle_message(update)
    before = len(bot._command_hits[51])
    bot.handle_message(update)
    assert len(bot._command_hits[51]) == before
