"""WebSocket /ws/stream endpoint: per-client sender with idle heartbeats."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import config
from app.services import streaming

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

HEARTBEAT_SECONDS = 20.0
RECEIVE_TIMEOUT_SECONDS = 20.0
SYMBOL = "XAU/USD (Gold)"


def heartbeat_event() -> dict[str, Any]:
    return streaming.event("heartbeat")


def parse_subscribe(message: Any) -> str | None:
    """Accept both subscribe wire formats and return the canonical timeframe."""
    if not isinstance(message, dict):
        return None
    is_subscribe = ("subscribe" in message
                    or str(message.get("type") or "").lower() == "subscribe")
    if not is_subscribe:
        return None
    return streaming.normalize_timeframe(message.get("timeframe"))


async def sender_tick(websocket: Any, queue: asyncio.Queue,
                      interval: float = HEARTBEAT_SECONDS) -> bool:
    """Send the next queued event, or a heartbeat after ``interval`` idle."""
    try:
        payload = await asyncio.wait_for(queue.get(), timeout=interval)
    except asyncio.TimeoutError:
        payload = heartbeat_event()
    await websocket.send_json(payload)
    return True


async def sender_loop(websocket: Any, queue: asyncio.Queue,
                      interval: float = HEARTBEAT_SECONDS) -> None:
    while True:
        try:
            await sender_tick(websocket, queue, interval)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a failed send ends only this client
            logger.debug("stream sender stopped", exc_info=True)
            return


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    if not config.origin_is_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    await websocket.accept()
    queue = await streaming.HUB.subscribe("1h")
    sender_task = asyncio.create_task(
        sender_loop(websocket, queue), name="goldsight-stream-sender")
    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=RECEIVE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                continue
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            timeframe = parse_subscribe(message)
            if timeframe is None:
                continue
            streaming.HUB.set_timeframe(queue, timeframe)
            streaming.HUB.enqueue(
                queue, streaming.event("subscribed", timeframe=timeframe,
                                       symbol=SYMBOL))
            streaming.HUB.enqueue(
                queue, streaming.event("resync", reason="reconnect"))
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a bad client must not affect the others
        logger.debug("stream client error", exc_info=True)
    finally:
        streaming.HUB.unsubscribe(queue)
        sender_task.cancel()
        try:
            await asyncio.gather(sender_task, return_exceptions=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001 - client may already be gone
            pass
