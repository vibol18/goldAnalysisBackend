"""Concept library, confluence, Learn catalogue and event-calendar endpoints."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.concepts import DETECTORS, DETECTOR_CATEGORIES, detect_all, detect_all_dicts, load_families
from app.concepts.base import CATEGORIES
from app.services import concept_context, data, db
from app.services.confluence import confluence, load_weights, reload_weights
from app.services.explanation import build_explanation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["concepts"])
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL = 60.0

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Signals can be wrong. "
    "Never risk money you can't afford to lose."
)


def _detect(timeframe: str, use_cache: bool = True) -> dict[str, Any]:
    now = time.time()
    key = f"{timeframe}|gold"
    if use_cache and key in _cache and now - _cache[key][0] < _CACHE_TTL:
        return _cache[key][1]
    entry = data.get_ohlcv(config.GOLD_TICKER, timeframe)
    if entry.df.empty:
        raise HTTPException(503, entry.error or "no live XAU/USD data available")
    context = concept_context.build_context(timeframe)
    concepts = detect_all_dicts(entry.df, context)
    by_category: dict[str, int] = {c: 0 for c in CATEGORIES}
    for c in concepts:
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1
    overview = data.get_market_overview()
    dxy = overview.get("correlations", {}).get("DXY", {})
    dxy_chg = dxy.get("change_pct") if isinstance(dxy, dict) else None
    payload: dict[str, Any] = {
        "timeframe": timeframe,
        "symbol": "XAU/USD (Gold)",
        "count": len(concepts),
        "by_category": by_category,
        "concepts": concepts,
        "dxy_change_pct": dxy_chg,
        "stale": entry.stale,
        "error": entry.error,
        "disclaimer": DISCLAIMER,
    }
    if use_cache:
        _cache[key] = (now, payload)
    return payload


@router.get("/concepts")
def concepts_endpoint(
    timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$"),
    refresh: bool = False,
) -> dict[str, Any]:
    """Every concept detected in real XAU/USD candles, with chart geometry."""
    if refresh:
        _cache.clear()
    try:
        return _detect(timeframe, use_cache=not refresh)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("concept detection failed")
        raise HTTPException(500, f"concept detection failed: {exc}") from exc


@router.get("/confluence")
def confluence_endpoint(
    timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$"),
) -> dict[str, Any]:
    """Confluence score, graded signal and analyst explanation."""
    try:
        payload = _detect(timeframe)
        entry = data.get_ohlcv(config.GOLD_TICKER, timeframe)
        from app.concepts import detect_all
        context = concept_context.build_context(timeframe)
        concepts = detect_all(entry.df, context)
        result = confluence(concepts, entry.df)
        explanation = build_explanation(concepts, result, entry.df, timeframe)
        return {
            "timeframe": timeframe,
            "symbol": "XAU/USD (Gold)",
            "confluence": result.to_dict(),
            "explanation": explanation,
            "concept_count": len(concepts),
            "stale": entry.stale,
            "disclaimer": DISCLAIMER,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("confluence failed")
        raise HTTPException(500, f"confluence failed: {exc}") from exc


@router.get("/backtest/concepts")
def concept_backtest_endpoint(
    years: int = Query(config.BACKTEST_YEARS, ge=1, le=10),
    refresh: bool = False,
) -> dict[str, Any]:
    """Per-concept performance plus the full graded-confluence replay."""
    key = f"backtest|{years}|gold"
    now = time.time()
    if not refresh and key in _cache and now - _cache[key][0] < 300.0:
        return _cache[key][1]
    try:
        from app.services.concept_backtest import (
            full_confluence_backtest, per_concept_performance,
        )
        entry = data.get_ohlcv(config.GOLD_TICKER, "1D")
        if entry.df.empty:
            raise HTTPException(503, entry.error or "no live XAU/USD data available")
        context = concept_context.build_context("1D")
        concepts = detect_all(entry.df, context, unlimited=True)
        payload = {
            "years": years,
            "symbol": "XAU/USD (Gold)",
            "per_concept": per_concept_performance(entry.df, concepts),
            "strategy": full_confluence_backtest(entry.df, years=years,
                                                 context=context, concepts=concepts),
            "weights": load_weights()["weights"],
            "stale": entry.stale,
            "disclaimer": DISCLAIMER,
        }
        _cache[key] = (now, payload)
        return payload
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("concept backtest failed")
        raise HTTPException(500, f"concept backtest failed: {exc}") from exc


@router.get("/learn")
def learn() -> dict[str, Any]:
    """Definition of every concept in the library, grouped by family."""
    load_families()
    definitions: list[dict[str, str]] = []
    for name, fn in sorted(DETECTORS.items(), key=lambda kv: (DETECTOR_CATEGORIES[kv[0]], kv[0])):
        doc = (fn.__doc__ or "").strip().splitlines()
        definitions.append({
            "type": name,
            "category": DETECTOR_CATEGORIES.get(name, "classic"),
            "title": name.replace("_", " ").title(),
            "definition": doc[0] if doc else name.replace("_", " ").capitalize(),
            "example": f"Open the chart with '{name.replace('_', ' ').title()}' enabled to see a detected example on real XAU/USD candles.",
        })
    return {
        "categories": list(CATEGORIES),
        "concepts": definitions,
        "total": len(definitions),
        "disclaimer": DISCLAIMER,
    }


@router.get("/weights")
def weights() -> dict[str, Any]:
    """Current tunable concept weights and category caps."""
    return load_weights()


@router.post("/weights/reload")
def weights_reload() -> dict[str, Any]:
    """Reload concept_weights.yaml after editing it."""
    return reload_weights()


@router.get("/events")
def events_get() -> dict[str, Any]:
    return {"events": concept_context.load_events(), "disclaimer": DISCLAIMER}


@router.post("/events")
def events_post(body: dict[str, Any]) -> dict[str, Any]:
    events = body.get("events")
    try:
        cleaned = [concept_context.normalise_event(e) for e in events]
        cleaned.sort(key=lambda e: e["time"])
        saved = concept_context.save_events(cleaned)
    except (ValueError, TypeError, OSError) as exc:
        raise HTTPException(400, f"invalid event calendar: {exc}") from exc
    return {"events": saved, "status": "saved", "disclaimer": DISCLAIMER}


@router.get("/signal-stats")
def signal_stats() -> dict[str, Any]:
    """Live hit rate of stored concept-driven signals."""
    return {"stats": db.signal_stats(), "disclaimer": DISCLAIMER}
