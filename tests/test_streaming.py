"""Tests for the shared WebSocket stream hub, helpers and client protocol."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import stream as stream_api
from app.services import data, streaming

T0 = 1_700_000_000


def make_frame(last_epoch: int, close: float = 2000.0, n: int = 4) -> pd.DataFrame:
    start = last_epoch - (n - 1) * 900
    index = pd.date_range(datetime.fromtimestamp(start, timezone.utc),
                          periods=n, freq="15min")
    closes = [close - (n - 1 - index_) for index_ in range(n)]
    return pd.DataFrame({
        "Open": [value - 1.0 for value in closes],
        "High": [value + 1.0 for value in closes],
        "Low": [value - 2.0 for value in closes],
        "Close": closes,
        "Volume": [100.0 + i for i in range(n)],
    }, index=index)


def inject(client: TestClient, events: list[dict[str, Any]]) -> None:
    async def publish_all() -> None:
        for payload in events:
            streaming.HUB.publish(payload)

    assert client.portal is not None
    client.portal.call(publish_all)


@pytest.fixture(autouse=True)
def reset_stream_state():
    streaming.HUB._queues.clear()
    streaming._CANDLE_STATES.clear()
    streaming._ANNOTATION_HASHES.clear()
    streaming._SEEN_SIGNALS.clear()
    streaming._SEEN_RESULTS.clear()
    streaming._LAST_STATUS = None
    streaming._last_force_at = 0.0
    streaming._last_overview_at = 0.0
    streaming._last_news_at = 0.0
    streaming._last_annotations_at = 0.0
    streaming._overview_cache = None
    streaming._task = None
    yield
    streaming.HUB._queues.clear()
    streaming._task = None


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(stream_api.router)
    with TestClient(app) as test_client:
        yield test_client


def test_subscribe_ack_resync_and_event_order(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"subscribe": "XAUUSD", "timeframe": "1h"})
        ack = ws.receive_json()
        assert ack["type"] == "subscribed"
        assert ack["timeframe"] == "1h"
        assert ack["symbol"] == "XAU/USD (Gold)"
        assert "ts" in ack
        resync = ws.receive_json()
        assert resync["type"] == "resync"
        assert resync["reason"] == "reconnect"

        states: dict[str, streaming._CandleState] = {}
        first = streaming.candle_events("1h", make_frame(T0, 2000.0), states)
        second = streaming.candle_events("1h", make_frame(T0 + 3600, 2002.0), states)
        assert [item["type"] for item in first] == ["candle_update"]
        assert [item["type"] for item in second] == ["candle_close", "candle_update"]

        injected = [
            streaming.event("tick", price=2000.0, change=1.0,
                            change_pct=0.05, time="2024-01-01T00:00:00+00:00"),
            *first,
            *second,
        ]
        inject(client, injected)
        received = [ws.receive_json() for _ in injected]
        assert [item["type"] for item in received] == [
            "tick", "candle_update", "candle_close", "candle_update"]
        assert received[0]["price"] == 2000.0
        assert received[2]["candle"]["time"] == T0
        assert received[2]["candle"]["close"] == 2000.0
        assert received[3]["candle"]["time"] == T0 + 3600


def test_closed_candle_is_never_modified():
    states: dict[str, streaming._CandleState] = {}
    events: list[dict[str, Any]] = []
    events += streaming.candle_events("15m", make_frame(T0, 2000.0), states)
    events += streaming.candle_events("15m", make_frame(T0, 2005.0), states)
    events += streaming.candle_events("15m", make_frame(T0 + 900, 2006.0), states)
    events += streaming.candle_events("15m", make_frame(T0 + 900, 2010.0), states)
    unchanged = len(events)
    events += streaming.candle_events("15m", make_frame(T0 + 900, 2010.0), states)
    assert len(events) == unchanged

    close_index = next(i for i, item in enumerate(events)
                       if item["type"] == "candle_close")
    assert events[close_index]["candle"]["time"] == T0
    assert events[close_index]["candle"]["close"] == 2005.0
    later = events[close_index + 1:]
    assert all(item["type"] != "candle_update" or item["candle"]["time"] != T0
               for item in later)


def test_one_publish_reaches_two_clients(client):
    with client.websocket_connect("/ws/stream") as ws1:
        with client.websocket_connect("/ws/stream") as ws2:
            assert streaming.HUB.subscribers_count == 2
            for ws in (ws1, ws2):
                ws.send_json({"type": "subscribe", "symbol": "XAUUSD",
                              "timeframe": "4h"})
                ack = ws.receive_json()
                assert ack["type"] == "subscribed"
                assert ack["timeframe"] == "4h"
                assert ws.receive_json()["type"] == "resync"
            payload = streaming.event("tick", price=1234.5, change=0.0,
                                      change_pct=0.0, time="2024-01-01T00:00:00+00:00")
            inject(client, [payload])
            assert ws1.receive_json() == payload
            assert ws2.receive_json() == payload


def test_disconnect_cleanup(client):
    baseline = streaming.HUB.subscribers_count
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"subscribe": "XAUUSD", "timeframe": "1d"})
        assert ws.receive_json()["timeframe"] == "1D"
        assert ws.receive_json()["type"] == "resync"
        assert streaming.HUB.subscribers_count == baseline + 1
    deadline = time.monotonic() + 2.0
    while streaming.HUB.subscribers_count != baseline and time.monotonic() < deadline:
        time.sleep(0.02)
    assert streaming.HUB.subscribers_count == baseline


@pytest.mark.asyncio
async def test_heartbeat_emitted_when_idle():
    queue: asyncio.Queue = asyncio.Queue()
    sent: list[dict[str, Any]] = []

    class FakeWebSocket:
        async def send_json(self, payload: dict[str, Any]) -> None:
            sent.append(payload)

    await asyncio.wait_for(
        stream_api.sender_tick(FakeWebSocket(), queue, interval=0.01), timeout=1.0)
    assert sent[0]["type"] == "heartbeat"
    assert "ts" in sent[0]

    queued = streaming.event("tick", price=2000.0, change=0.0,
                             change_pct=0.0, time="2024-01-01T00:00:00+00:00")
    queue.put_nowait(queued)
    await asyncio.wait_for(
        stream_api.sender_tick(FakeWebSocket(), queue, interval=5.0), timeout=1.0)
    assert sent[1] == queued


@pytest.mark.asyncio
async def test_start_stop_streaming_is_idempotent(monkeypatch):
    async def fake_loop() -> None:
        await asyncio.sleep(3600)

    monkeypatch.setattr(streaming, "stream_loop", fake_loop)
    first = await streaming.start_streaming()
    second = await streaming.start_streaming()
    assert first is second
    assert first is not None
    await streaming.stop_streaming()
    await streaming.stop_streaming()
    assert streaming._task is None
    assert first.cancelled()


@pytest.mark.asyncio
async def test_stream_cycle_broadcasts_once_and_dedupes(monkeypatch):
    frame = make_frame(T0, 2000.0)
    overview = {
        "gold": {"price": 2000.0, "prev_close": 1990.0, "change_pct": 0.5,
                 "as_of": "2024-01-01T00:00:00+00:00"},
        "correlations": {
            "DXY": {"price": 104.1, "change_pct": -0.2},
            "US10Y": {"price": 4.2, "change_pct": 0.1},
            "SPX": {"price": 5000.0, "change_pct": 0.3},
        },
    }
    rows = [{
        "id": 7, "timeframe": "1h", "action": "BUY", "confidence": 90.0,
        "price": 2000.0, "entry": 1998.0, "stop_loss": 1990.0,
        "take_profit1": 2010.0, "take_profit2": 2020.0, "status": "open",
        "created_at": "2024-01-01T00:00:00+00:00",
        "explanation": '{"grade": "A+", "summary": "real data"}',
    }]
    monkeypatch.setattr(data, "get_ohlcv", lambda *a, **kw: data.CacheEntry(
        df=frame, fetched_at=time.time(), source="test"))
    monkeypatch.setattr(data, "get_market_overview", lambda: overview)
    monkeypatch.setattr(streaming.news, "fetch_news", lambda: {
        "items": [{"title": "Gold rises"}], "sentiment": 0.5,
        "sentiment_label": "bullish"})
    monkeypatch.setattr(streaming.db, "list_signals", lambda limit=20: rows)
    monkeypatch.setattr(streaming.chart_annotations, "build_payload",
                        lambda entry, timeframe: {"annotations": [
                            {"id": "line-1", "type": "bos_line"}]})

    queue = await streaming.HUB.subscribe("1h")
    await streaming.stream_cycle()
    first_types = []
    while not queue.empty():
        first_types.append(queue.get_nowait()["type"])
    for expected in ("status", "tick", "candle_update", "macro", "news",
                     "annotations", "signal"):
        assert expected in first_types

    await streaming.stream_cycle()
    second_types = []
    while not queue.empty():
        second_types.append(queue.get_nowait())
    assert not any(item["type"] == "candle_update" for item in second_types)
    assert not any(item["type"] == "annotations" for item in second_types)
    assert not any(item["type"] == "signal" for item in second_types)

    rows[0] = {**rows[0], "status": "closed", "hit": "TP", "result_pct": 0.5}
    await streaming.stream_cycle()
    third_types = []
    while not queue.empty():
        third_types.append(queue.get_nowait())
    result = next(item for item in third_types if item["type"] == "signal_result")
    assert result["id"] == 7
    assert result["status"] == "closed"
    assert result["hit"] == "TP"
    assert result["result_pct"] == 0.5
