"""Tests for alert settings, scanner, SSE pub/sub and notification gating."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import alerts as alerts_api
from app.services import alerts, concept_context, data, db

ENV_VARS = (
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SMTP_HOST", "SMTP_PORT",
    "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_TO",
)


def frame(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 2000.0 * np.cumprod(1 + rng.normal(0.0002, 0.004, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) * 1.003,
        "Low": np.minimum(open_, close) * 0.997,
        "Close": close,
        "Volume": rng.integers(500, 40000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))


class FakeSignal:
    payload = {
        "action": "BUY",
        "grade": "A+",
        "price": 2345.6,
        "entry": 2340.0,
        "entry_zone": [2338.0, 2342.0],
        "stop_loss": 2330.0,
        "take_profit1": 2355.0,
        "take_profit2": 2365.0,
        "take_profit3": 2372.0,
        "explanation": {"narrative": "Bullish structure with FVG support. " * 12},
        "created_at": "2024-01-01T12:00:00+00:00",
    }

    def to_dict(self) -> dict:
        return dict(self.payload)


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "alerts.db")
    monkeypatch.setattr(db, "_conn", None)
    db.init_db()
    temp_conn = db._conn
    alerts._dedupe.clear()
    alerts._subscribers.clear()
    yield
    temp_conn.close()


@pytest.fixture()
def stubbed(monkeypatch, isolated):
    candles = frame()

    def fake_get_ohlcv(ticker, timeframe, force=False):
        return data.CacheEntry(df=candles, fetched_at=0.0, source="test")

    monkeypatch.setattr(data, "get_ohlcv", fake_get_ohlcv)
    monkeypatch.setattr(concept_context, "build_context", lambda tf: {})
    monkeypatch.setattr(alerts, "generate_signal", lambda *a, **kw: FakeSignal())
    return candles


@pytest.fixture()
def client(isolated):
    app = FastAPI()
    app.include_router(alerts_api.router)
    with TestClient(app) as test_client:
        yield test_client


def test_default_settings_shape(client):
    response = client.get("/api/alerts/settings")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["timeframes"] == ["15m", "1h", "4h"]
    assert body["min_grade"] in ("A", "A+")
    assert set(body["channels"]) == {
        "browser", "sound", "toast", "telegram", "email"}
    assert all(isinstance(v, bool) for v in body["channels"].values())
    assert body["telegram_configured"] is False
    assert body["email_configured"] is False
    assert body["disclaimer"] == (
        "Educational analysis only. Not financial advice. Signals can be wrong. "
        "Never risk money you can't afford to lose."
    )


def test_settings_round_trip(client):
    payload = {
        "timeframes": ["1h", "4h"],
        "min_grade": "A+",
        "channels": {
            "browser": False, "sound": True, "toast": False,
            "telegram": True, "email": True,
        },
    }
    saved = client.post("/api/alerts/settings", json=payload)
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["timeframes"] == ["1h", "4h"]
    assert body["min_grade"] == "A+"
    assert body["channels"]["sound"] is True
    assert body["channels"]["telegram"] is True
    fetched = client.get("/api/alerts/settings").json()
    assert fetched["timeframes"] == ["1h", "4h"]
    assert fetched["min_grade"] == "A+"
    assert fetched["channels"] == payload["channels"]
    assert client.get("/api/alerts").json()["count"] == 0


@pytest.mark.asyncio
async def test_scan_once_creates_signal_and_dedupes(stubbed):
    alerts.save_settings({
        "timeframes": ["1h"], "min_grade": "A",
        "channels": {"browser": True, "toast": True},
    })
    first = await alerts.scan_once()
    assert len(first) == 1
    alert = first[0]
    assert alert["type"] == "signal"
    assert alert["timeframe"] == "1h"
    assert alert["action"] == "BUY"
    assert alert["grade"] == "A+"
    for fragment in ("BUY", "A+", "Entry", "SL", "TP1", "TP2", "TP3"):
        assert fragment in alert["message"]
    assert len(alert["message"].split("Bullish structure")[-1]) <= 240

    second = await alerts.scan_once()
    assert second == []
    stored = alerts.list_alerts()
    assert len(stored) == 1
    assert stored[0]["id"] == alert["id"]
    assert set(stored[0]) == {
        "id", "time", "type", "timeframe", "action", "grade",
        "price", "message", "channels",
    }


@pytest.mark.asyncio
async def test_stored_signal_touch_creates_tp_alert(stubbed):
    alerts.save_settings({"timeframes": ["1h"], "min_grade": "A", "channels": {}})
    db.save_signal({
        "timeframe": "1h", "action": "BUY", "confidence": 80,
        "final_score": 0.7, "rule_score": 0.6, "ml_probability": None,
        "entry": 1000.0, "stop_loss": 900.0,
        "take_profit1": 1000.0, "take_profit2": 1100.0,
        "risk_reward": 2.0, "atr": 12.0, "price": 1000.0,
        "explanation": {"summary": "test"},
    })
    created = await alerts.scan_once()
    kinds = [alert["type"] for alert in created]
    assert "signal" in kinds
    assert "tp" in kinds


@pytest.mark.asyncio
async def test_publish_reaches_subscribers(stubbed):
    queue = alerts.subscribe()
    try:
        sent = await alerts.publish({
            "type": "entry", "timeframe": "1h", "action": "BUY", "grade": "A",
            "price": 2340.0, "message": "Entry zone touched", "channels": ["browser"],
        })
        received = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert received == sent
        assert received["id"] > 0
    finally:
        alerts.unsubscribe(queue)


@pytest.mark.asyncio
async def test_sse_stream_emits_alert_event(stubbed):
    class FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    response = await alerts_api.stream(FakeRequest())
    assert response.media_type == "text/event-stream"
    iterator = response.body_iterator

    async def fire() -> None:
        await asyncio.sleep(0.05)
        await alerts.publish({
            "type": "test", "timeframe": "1h", "action": "TEST",
            "grade": "n/a", "price": None,
            "message": "SSE test alert", "channels": ["browser"],
        })

    task = asyncio.create_task(fire())
    try:
        chunks: list[str] = []

        async def collect() -> str:
            async for chunk in iterator:
                chunks.append(chunk)
                if any("event: alert" in part for part in chunks):
                    return "".join(chunks)
            return "".join(chunks)

        raw = await asyncio.wait_for(collect(), timeout=2.0)
        assert "event: alert" in raw
        data_line = next(line for line in raw.splitlines()
                         if line.startswith("data: "))
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["type"] == "test"
        assert payload["id"] > 0
        assert payload["channels"] == ["browser"]
    finally:
        task.cancel()
        await iterator.aclose()


@pytest.mark.asyncio
async def test_notifications_not_called_when_not_configured(stubbed, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("notification backend must not be called")

    monkeypatch.setattr(alerts, "send_telegram", forbidden)
    monkeypatch.setattr(alerts, "send_email", forbidden)
    settings = alerts.get_settings()
    settings["channels"]["telegram"] = True
    settings["channels"]["email"] = True
    alerts.save_settings(settings)
    await alerts.scan_once()
    results = await alerts.notify({"type": "signal", "timeframe": "1h",
                                   "grade": "A+", "message": "BUY"})
    assert "not configured" in results["telegram"]
    assert "not configured" in results["email"]


@pytest.mark.asyncio
async def test_notification_text_contains_signal_details(stubbed, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    settings = alerts.get_settings()
    settings["channels"]["telegram"] = True
    alerts.save_settings(settings)

    captured: dict[str, str] = {}

    def fake_send(text: str) -> str:
        captured["text"] = text
        return "Telegram message sent."

    monkeypatch.setattr(alerts, "send_telegram", fake_send)
    created = await alerts.scan_once()
    results = await alerts.notify(created[0])
    text = captured["text"]
    for fragment in ("BUY", "A+", "Entry", "SL", "TP1", "TP2", "TP3"):
        assert fragment in text
    assert alerts.DISCLAIMER in text
    assert results["telegram"] == "Telegram message sent."
