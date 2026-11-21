"""Build the real-data context passed to concept detectors (no synthetic data)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app import config
from app.services import data

logger = logging.getLogger(__name__)

EVENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "events.json"
MACRO_TICKERS = (
    ("dxy_df", config.DXY_TICKER),
    ("us10y_df", config.TNX_TICKER),
    ("spx_df", config.SPX_TICKER),
)
MTF_FRAMES = (
    ("htf_1d", "1D"),
    ("htf_4h", "4h"),
    ("ltf_15m", "15m"),
)


def load_events() -> list[dict[str, Any]]:
    """Read the user-editable economic event calendar."""
    try:
        payload = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
        events = payload.get("events", []) if isinstance(payload, dict) else payload
        return [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []
    except (OSError, ValueError) as exc:
        logger.warning("could not read event calendar: %s", exc)
        return []


def normalise_event(event: dict[str, Any]) -> dict[str, Any]:
    """Validate one user-edited event and give it a stable shape."""
    import pandas as pd

    if not isinstance(event, dict):
        raise ValueError("each event must be an object")
    title = str(event.get("title") or event.get("name") or "").strip()
    if not title:
        raise ValueError("event title is required")
    raw_time = event.get("time") or event.get("datetime") or event.get("date")
    try:
        when = pd.to_datetime(raw_time, utc=True, errors="coerce")
    except (ValueError, TypeError):
        when = pd.NaT
    if pd.isna(when):
        raise ValueError(f"event '{title}' has an invalid time")
    impact = str(event.get("impact", "medium")).lower()
    if impact not in {"high", "medium", "low"}:
        raise ValueError(f"event '{title}' has an invalid impact level")
    out = {
        "title": title,
        "name": str(event.get("name") or title),
        "time": when.isoformat(),
        "impact": impact,
        "country": str(event.get("country") or ""),
        "forecast": str(event.get("forecast") or event.get("expected") or ""),
        "actual": str(event.get("actual") or ""),
        "previous": str(event.get("previous") or ""),
        "source": "user-edited",
    }
    return out


def save_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist the economic event calendar (validated list of dicts)."""
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise ValueError("events must be a list of objects")
    EVENTS_PATH.write_text(
        json.dumps({"events": events}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return events


def build_context(timeframe: str = "1h", include_mtf: bool = True,
                  include_macro: bool = True) -> dict[str, Any]:
    """Fetch real higher-timeframes, macro series, news and events."""
    ctx: dict[str, Any] = {}
    if include_mtf:
        for key, tf in MTF_FRAMES:
            if tf == timeframe:
                continue
            try:
                frame = data.get_ohlcv(config.GOLD_TICKER, tf).df
                if not frame.empty:
                    ctx[key] = frame
            except Exception as exc:  # noqa: BLE001
                logger.debug("no %s context: %s", tf, exc)
    if include_macro:
        for key, ticker in MACRO_TICKERS:
            try:
                frame = data.get_ohlcv(ticker, "1D").df
                if not frame.empty:
                    ctx[key] = frame
            except Exception as exc:  # noqa: BLE001
                logger.debug("no %s context: %s", ticker, exc)
        try:
            from app.services import news as news_service
            payload = news_service.fetch_news()
            ctx["news"] = {
                "sentiment": payload.get("sentiment"),
                "label": payload.get("sentiment_label"),
                "headlines": [i.get("title") for i in payload.get("items", [])[:6]],
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("no news context: %s", exc)
    ctx["events"] = load_events()
    return ctx
