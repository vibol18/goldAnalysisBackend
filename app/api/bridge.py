"""Authenticated receiver for the Windows MT5 bridge.

Bridge requests use ``X-Bridge-Key`` instead of the browser session. They are
exempt from CSRF and production login, but never from this key, rate limiting,
schema validation and a hard body-size cap.
"""
from __future__ import annotations

import hmac
import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from app import config
from app.services import bridge, db, prices, ratelimit
from app.services import auth as auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/bridge", tags=["bridge"])


class CandleIn(BaseModel):
    time: int = Field(..., ge=0, description="MT5 broker-local open time (epoch seconds)")
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(0, ge=0)
    real_volume: float | None = Field(None, ge=0)
    spread: float = Field(0, ge=0)

    @model_validator(mode="after")
    def ohlc_coherent(self) -> "CandleIn":
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is below open/close/low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is above open/close/high")
        return self


class CandlesBody(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    timeframe: str = Field(..., min_length=1, max_length=8)
    broker: str = Field("", max_length=64)
    server: str = Field("", max_length=64)
    digits: int = Field(2, ge=0, le=5)
    point: float = Field(0.01, gt=0)
    contract_size: float = Field(100.0, gt=0)
    timezone_offset_seconds: int = Field(0, ge=-86400, le=86400)
    currency: str = Field("USD", max_length=16)
    candles: list[CandleIn] = Field(..., min_length=1,
                                    max_length=config.BRIDGE_MAX_CANDLES_PER_REQUEST)

    @field_validator("symbol")
    @classmethod
    def clean_symbol(cls, value: str) -> str:
        return value.strip()[:32]


class TickBody(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    time: int = Field(..., ge=0, description="MT5 broker-local tick time (epoch seconds)")
    bid: float = Field(..., gt=0)
    ask: float = Field(..., gt=0)
    spread: float = Field(0, ge=0)
    broker: str = Field("", max_length=64)
    server: str = Field("", max_length=64)
    digits: int | None = Field(None, ge=0, le=5)
    point: float | None = Field(None, gt=0)
    contract_size: float | None = Field(None, gt=0)
    timezone_offset_seconds: int = Field(0, ge=-86400, le=86400)

    @model_validator(mode="after")
    def coherent(self) -> "TickBody":
        if self.ask < self.bid:
            raise ValueError("ask must be >= bid")
        return self


def _authorize(request: Request) -> None:
    if not config.BRIDGE_API_KEY:
        raise HTTPException(503, "MT5 bridge is not configured (BRIDGE_API_KEY is empty)")
    provided = request.headers.get("X-Bridge-Key", "")
    if not provided or not hmac.compare_digest(provided, config.BRIDGE_API_KEY):
        raise HTTPException(401, "invalid bridge key")
    ratelimit.check(
        ratelimit.ip_key("bridge", auth_service.client_ip(request)),
        config.BRIDGE_RATE_LIMIT_PER_MINUTE,
    )


async def _json_body(request: Request) -> dict[str, Any]:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > config.BRIDGE_MAX_BODY_BYTES:
        raise HTTPException(413, "payload too large")
    raw = await request.body()
    if len(raw) > config.BRIDGE_MAX_BODY_BYTES:
        raise HTTPException(413, "payload too large")
    try:
        import json
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "invalid JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(400, "payload must be an object")
    return payload


def _meta_from(body: BaseModel) -> dict[str, Any]:
    return {
        "symbol": body.symbol,
        "broker": getattr(body, "broker", "") or config.MT5_BROKER_NAME,
        "server": getattr(body, "server", ""),
        "digits": getattr(body, "digits", 2),
        "point": getattr(body, "point", 0.01),
        "contract_size": getattr(body, "contract_size", 100.0),
        "timezone_offset_seconds": body.timezone_offset_seconds,
        "currency": getattr(body, "currency", "USD"),
    }


def _run_signal_catchup() -> None:
    """Evaluate all bridge timeframes after a closed candle arrives."""
    try:
        from app.services.signal_runner import run_all_timeframes
        run_all_timeframes()
    except Exception as exc:  # noqa: BLE001
        logger.warning("bridge signal catch-up failed: %s", exc)


@router.post("/candles")
async def post_candles(request: Request, background: BackgroundTasks) -> dict[str, Any]:
    _authorize(request)
    payload = await _json_body(request)
    try:
        body = CandlesBody(**payload)
    except Exception as exc:  # noqa: BLE001 - pydantic errors are client errors
        raise HTTPException(422, f"invalid candle payload: {exc}") from exc
    tf = bridge.normalise_timeframe(body.timeframe)
    if tf is None or tf not in bridge.CANONICAL_TF:
        raise HTTPException(422, f"unsupported timeframe: {body.timeframe}")
    meta = _meta_from(body)
    offset = body.timezone_offset_seconds
    rows = [
        {
            "broker_time": candle.time,
            "time": bridge.normalise_time(candle.time, offset),
            "open": candle.open, "high": candle.high,
            "low": candle.low, "close": candle.close,
            "volume": candle.volume, "spread": candle.spread,
            "real_volume": candle.real_volume,
        }
        for candle in body.candles
    ]
    from datetime import datetime, timezone
    received_at = datetime.now(timezone.utc).isoformat()
    meta.update({"last_candle_at": received_at})
    db.bridge_update_meta(body.symbol, meta)
    new = db.bridge_insert_candles(body.symbol, tf, rows)
    logger.info("bridge candles %s/%s: %d received, %d new", body.symbol, tf, len(rows), new)
    if new:
        background.add_task(_run_signal_catchup)
    return {"status": "ok", "symbol": body.symbol, "timeframe": tf,
            "received": len(rows), "new": new, "duplicate": len(rows) - new}


@router.post("/tick")
async def post_tick(request: Request) -> dict[str, Any]:
    _authorize(request)
    payload = await _json_body(request)
    try:
        body = TickBody(**payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"invalid tick payload: {exc}") from exc
    offset = body.timezone_offset_seconds
    previous = db.bridge_get_meta(body.symbol) or {}
    meta = {
        "symbol": body.symbol,
        "broker": body.broker or previous.get("broker") or config.MT5_BROKER_NAME,
        "server": body.server or previous.get("server") or "",
        "digits": body.digits if body.digits is not None else previous.get("digits", 2),
        "point": body.point if body.point is not None else previous.get("point", 0.01),
        "contract_size": body.contract_size if body.contract_size is not None
        else previous.get("contract_size", 100.0),
        "timezone_offset_seconds": offset,
        "currency": previous.get("currency", "USD"),
    }
    from datetime import datetime, timezone
    received_at = datetime.now(timezone.utc).isoformat()
    meta["last_tick_at"] = received_at
    db.bridge_update_meta(body.symbol, meta)
    db.bridge_insert_tick(body.symbol, {
        "broker_time": body.time,
        "time": bridge.normalise_time(body.time, offset),
        "bid": body.bid, "ask": body.ask, "spread": body.spread,
        "digits": meta["digits"], "point": meta["point"],
        "contract_size": meta["contract_size"],
        "broker": meta["broker"], "server": meta["server"],
    })
    prices.FEED.update("mt5", body.bid, body.ask, broker_time=body.time)
    return {"status": "ok", "symbol": body.symbol,
            "bid": body.bid, "ask": body.ask, "spread": body.spread,
            "price_state": prices.FEED.snapshot()["state"]}


@router.get("/status")
def get_status() -> dict[str, Any]:
    """Non-secret bridge status (safe for the dashboard)."""
    return bridge.status()


@router.get("/candles")
def get_candles(
    timeframe: str = Query("15m"),
    limit: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    """Last N bridge candles for the Compare-with-MT5 page."""
    tf = bridge.normalise_timeframe(timeframe) or "15m"
    meta = db.bridge_get_meta() or {}
    sym = str(meta.get("symbol") or config.MT5_SYMBOL)
    offset = int(meta.get("timezone_offset_seconds") or 0)
    rows = db.bridge_get_candles(sym, tf, limit=limit)
    from datetime import datetime, timezone
    for row in rows:
        row["utc_time"] = datetime.fromtimestamp(row["time"], timezone.utc).isoformat()
        row["broker_time_iso"] = datetime.fromtimestamp(
            row["broker_time"], timezone.utc).isoformat()
    return {
        "symbol": sym, "timeframe": tf, "timezone_offset_seconds": offset,
        "broker": meta.get("broker") or config.MT5_BROKER_NAME,
        "digits": int(meta.get("digits") or 2),
        "point": float(meta.get("point") or 0.01),
        "status": bridge.status(), "candles": rows,
        "disclaimer": "Educational analysis only. Not financial advice.",
    }
