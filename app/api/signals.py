"""Signal endpoints: generate current signal, history, outcomes, stats."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.services import data, db, telegram_delivery
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
        df = data.get_ohlcv(config.GOLD_TICKER, timeframe).df
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
    entry = data.get_ohlcv(config.GOLD_TICKER, timeframe)
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
    return {"signals": rows, "stats": db.signal_stats()}


@router.get("/stats")
def stats() -> dict[str, Any]:
    """Aggregate accuracy of past signals."""
    return db.signal_stats()
