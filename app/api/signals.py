"""Signal endpoints: generate current signal, history, outcomes, stats."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.services import data, db, price_service, prices, telegram_delivery
from app.services.signal_engine import closed_frame, evaluate_outcome, generate_signal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/signals", tags=["signals"])

_last_signal_cache: dict[str, Any] = {"timeframe": None, "signal": None, "at": 0.0}
_outcome_refreshed_at: dict[str, float] = {}


def _refresh_outcomes(timeframe: str) -> None:
    """Re-evaluate open signals against the latest candles (at most every 30s)."""
    now = time.time()
    if now - _outcome_refreshed_at.get(timeframe, 0.0) < 30:
        return
    try:
        df = price_service.get_series(timeframe).df
        for sig in db.list_signals(limit=500):
            if sig["timeframe"] != timeframe or sig["status"] != "open":
                continue
            if sig["action"] == "HOLD":
                continue
            outcome = evaluate_outcome(sig, df)
            if outcome["status"] != sig["status"] or outcome.get("hit") != sig.get("hit"):
                db.update_signal_outcome(
                    sig["id"], outcome["status"], outcome.get("hit"),
                    outcome.get("result_pct"))
                if outcome.get("hit"):
                    telegram_delivery.dispatch_result(
                        "sl" if outcome["hit"] == "SL" else "tp",
                        {**sig, "status": outcome["status"], "hit": outcome["hit"]},
                    )
    except Exception as exc:  # noqa: BLE001
        logger.warning("outcome refresh failed: %s", exc)
    finally:
        _outcome_refreshed_at[timeframe] = now


@router.get("/current")
def current(timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$"),
            save: bool = True) -> dict[str, Any]:
    """Compute (and optionally persist) the current signal."""
    feed = prices.FEED.snapshot()
    if prices.FEED.received_any() and feed["state"] in (prices.STALE, prices.CLOSED):
        reason = feed.get("reason") or "no recent tick"
        return {
            "action": "HOLD", "grade": "C", "timeframe": timeframe,
            "stale": True, "data_stale": True, "data_synthetic": False,
            "error": f"Price feed {feed['state']}: {reason}",
            "confidence": 0.0, "rule_score": 0.0, "final_score": 0.0,
            "ml_probability": None, "entry": 0.0, "stop_loss": 0.0,
            "take_profit1": 0.0, "take_profit2": 0.0, "risk_reward": 0.0,
            "atr": 0.0, "price": feed.get("price") or 0.0,
            "price_state": feed["state"], "duplicate": False,
            "explanation": {
                "summary": "No trade taken: price feed is not live.",
                "reasons": [f"Feed state {feed['state']}: {reason}"],
                "why_not_trading": f"No trade taken: {feed['state']} price feed ({reason}).",
            },
        }

    if config.REQUIRE_BRIDGE_FOR_SIGNALS and prices.FEED.snapshot()["source"] != "mt5":
        return {
            "action": "HOLD", "grade": "C", "timeframe": timeframe,
            "stale": True, "data_stale": True, "data_synthetic": False,
            "confidence": 0.0, "rule_score": 0.0, "final_score": 0.0,
            "ml_probability": None, "entry": 0.0, "stop_loss": 0.0,
            "take_profit1": 0.0, "take_profit2": 0.0, "risk_reward": 0.0,
            "atr": 0.0, "price": prices.FEED.snapshot().get("price") or 0.0,
            "duplicate": False,
            "error": "MT5 bridge offline - BUY/SELL signals paused",
            "explanation": {
                "summary": "No trade taken: MT5 bridge is offline.",
                "reasons": ["MT5 bridge offline - broker feed required"],
                "why_not_trading": "No trade taken: MT5 bridge offline; signals paused until the broker feed returns.",
            },
        }
    entry = price_service.get_series(timeframe)
    if len(entry.df) < 260:
        return {
            "action": None,
            "timeframe": timeframe,
            "stale": True,
            "data_stale": True,
            "data_synthetic": False,
            "error": (entry.error or "Live XAU/USD data is unavailable. "
                      "Add TWELVE_DATA_API_KEY to backend/.env to enable real data."),
        }

    overview = data.get_market_overview()
    dxy_chg = None
    dxy_info = overview.get("correlations", {}).get("DXY", {})
    if isinstance(dxy_info, dict) and "change_pct" in dxy_info:
        dxy_chg = dxy_info["change_pct"]

    try:
        from app.services.concept_context import build_context
        context = build_context(timeframe)
        closed = closed_frame(entry.df)
        sig = generate_signal(closed, timeframe, dxy_change_pct=dxy_chg,
                              concept_context=context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("signal generation failed")
        raise HTTPException(500, f"signal generation failed: {exc}") from exc

    payload = sig.to_dict()
    payload["data_stale"] = entry.stale
    payload["data_synthetic"] = entry.synthetic
    payload["data_source"] = entry.source
    payload["price_side"] = (entry.meta or {}).get("price_side", "bid")
    payload["broker_symbol"] = (entry.meta or {}).get("symbol") or (
        entry.meta or {}
    ).get("symbol_label")
    if entry.source == "mt5":
        tick = db.bridge_get_latest_tick() or {}
        bid = tick.get("bid")
        ask = tick.get("ask")
        spread = tick.get("spread")
        payload["bid"] = bid
        payload["ask"] = ask
        payload["spread"] = spread
        payload["broker_symbol"] = (entry.meta or {}).get("symbol") or tick.get("broker") or "XAUUSD"
        original_entry = float(payload.get("entry") or 0)
        if payload["action"] == "BUY" and ask is not None:
            delta = float(ask) - original_entry
            payload["entry"] = round(float(ask), 2)
            for key in ("stop_loss", "take_profit1", "take_profit2", "take_profit3"):
                if payload.get(key) is not None:
                    payload[key] = round(float(payload[key]) + delta, 2)
        elif payload["action"] == "SELL" and bid is not None:
            delta = float(bid) - original_entry
            payload["entry"] = round(float(bid), 2)
            for key in ("stop_loss", "take_profit1", "take_profit2", "take_profit3"):
                if payload.get(key) is not None:
                    payload[key] = round(float(payload[key]) + delta, 2)
    payload["legacy_data"] = False
    meta = getattr(entry, "meta", None) or {}
    payload["dataset_hash"] = meta.get("dataset_hash")
    payload["series_source"] = meta.get("series_source") or entry.source
    payload["series_broker"] = meta.get("series_broker")
    if isinstance(payload.get("explanation"), dict):
        payload["explanation"]["dataset_hash"] = payload["dataset_hash"]
        payload["explanation"]["series_source"] = payload["series_source"]
        payload["explanation"]["series_broker"] = payload["series_broker"]

    _refresh_outcomes(timeframe)
    payload["duplicate"] = False
    # Never persist signals computed from fallback/synthetic data. A signal is
    # immutable once stored: it is stamped with its closed-candle time and is
    # never recomputed or deleted.
    if save and payload["action"] in ("BUY", "SELL") and not entry.synthetic:
        from datetime import datetime, timezone
        same = [s for s in db.list_signals(limit=300)
                if s.get("timeframe") == timeframe and s.get("action") == payload["action"]]
        active = [s for s in same if s.get("status") not in ("closed",)]
        blocked = bool(active)
        cooldown = config.SIGNAL_COOLDOWN_MINUTES * 60.0
        for row in same[:8]:
            closed_at = row.get("closed_at")
            if not closed_at:
                continue
            try:
                if (time.time() - datetime.fromisoformat(closed_at).timestamp()) < cooldown:
                    blocked = True
                    break
            except (ValueError, TypeError):
                continue
        if blocked:
            payload["duplicate"] = True
        else:
            from app.services import stats as stats_service
            payload = stats_service.annotate_payload(payload)
            payload["source"] = "legacy_data" if payload.get("legacy_data") else (
                entry.source or "live"
            )
            payload["id"] = db.save_signal(payload)
            telegram_delivery.dispatch_signal(payload)
    return payload


@router.get("/history")
def history(limit: int = Query(100, le=500), offset: int = 0,
            timeframe: str | None = None,
            action: str | None = None) -> dict[str, Any]:
    """Stored signal history with live outcome stats."""
    _refresh_outcomes(timeframe or "1h")
    rows = db.list_signals(limit=limit, offset=offset)
    if timeframe:
        rows = [r for r in rows if r["timeframe"] == timeframe]
    if action:
        rows = [r for r in rows if r["action"] == action.upper()]
    for row in rows:
        row["legacy_data"] = row.get("source") == "legacy_data"
    return {"signals": rows, "stats": db.signal_stats()}


@router.get("/stats")
def stats() -> dict[str, Any]:
    """Aggregate accuracy of past signals."""
    return db.signal_stats()
