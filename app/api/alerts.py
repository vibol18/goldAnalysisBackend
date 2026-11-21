"""Alert settings, history, SSE stream and delivery-test endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.services import alerts as alert_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return alert_service.get_settings()


@router.post("/settings")
def post_settings(body: dict[str, Any]) -> dict[str, Any]:
    return alert_service.save_settings(body)


@router.get("")
def list_alerts(limit: int = 200) -> dict[str, Any]:
    rows = alert_service.list_alerts(limit)
    return {"alerts": rows, "count": len(rows)}


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        queue = alert_service.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    alert = await asyncio.wait_for(
                        queue.get(), timeout=alert_service.KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield "event: alert\ndata: " + json.dumps(alert, default=str) + "\n\n"
        finally:
            alert_service.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/test")
async def test_delivery() -> dict[str, Any]:
    settings = alert_service.get_settings()
    channels = [name for name, enabled in settings["channels"].items() if enabled]
    alert = await alert_service.publish({
        "type": "test",
        "timeframe": "1h",
        "action": "TEST",
        "grade": "n/a",
        "price": None,
        "message": "This is a GoldSight test alert. " + alert_service.DISCLAIMER,
        "channels": channels,
    })
    results = await alert_service.notify(alert)
    sent = settings["channels"]["browser"] or any(
        result.startswith(("Telegram message sent", "Email sent"))
        for result in results.values()
    )
    return {"sent": bool(sent), "results": results}
