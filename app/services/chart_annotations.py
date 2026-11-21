"""Build every chart drawing (concept geometry + signal plan + stored signals).

The frontend is a pure renderer: this module turns detected concepts, the
current confluence plan and stored signal history into a flat, fully
serialisable list of chart annotations. Detection is strictly causal, so an
annotation whose ``end_time`` predates an edit to later candles never changes
its identity or geometry.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from app.concepts import detect_all
from app.concepts.base import json_safe, ts_to_str
from app.services import concept_context, data, db
from app.services.confluence import confluence

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Signals can be wrong. "
    "Never risk money you can't afford to lose."
)
SYMBOL = "XAU/USD (Gold)"
RELEVANT_LIMIT = 60

ZONE_MAP: dict[str, str] = {
    "fvg": "fvg_box",
    "ifvg": "fvg_box",
    "order_block": "ob_box",
    "mitigation_block": "ob_box",
    "breaker_block": "breaker_box",
    "sr_zone": "sr_zone",
    "supply_demand": "supply_demand_zone",
    "volume_profile": "volume_profile_zone",
    "ichimoku": "ichimoku_zone",
}

LINE_MAP: dict[str, str] = {
    "equal_highs": "eqh_line",
    "equal_lows": "eql_line",
    "pdh": "pdh_line",
    "pdl": "pdl_line",
    "pwh": "pwh_line",
    "pwl": "pwl_line",
    "session_high": "session_high_line",
    "session_low": "session_low_line",
    "pivot_points": "pivot_line",
    "vwap": "vwap_line",
    "anchored_vwap": "vwap_line",
    "daily_open": "daily_open_line",
    "midnight_open": "daily_open_line",
    "htf_level": "htf_level_line",
    "trendline": "trendline",
    "channel": "channel_line",
    "bos": "bos_line",
    "choch": "choch_line",
    "mss": "choch_line",
}

MARKER_MAP: dict[str, str] = {
    "liquidity_sweep": "sweep_marker",
    "displacement": "displacement_marker",
    "killzone": "killzone_shade",
}

CANDLESTICK_TYPES = {
    "engulfing", "pin_bar", "doji", "morning_star", "evening_star",
    "inside_bar", "outside_bar", "three_soldiers", "three_crows",
}
PATTERN_TYPES = {
    "double_bottom", "double_top", "head_shoulders",
    "inverse_head_shoulders", "triangle", "wedge", "flag", "range_breakout",
}

LEVEL_LABELS: dict[str, str] = {
    "eqh_line": "EQH",
    "eql_line": "EQL",
    "pdh_line": "PDH",
    "pdl_line": "PDL",
    "pwh_line": "PWH",
    "pwl_line": "PWL",
    "session_high_line": "SESSION HIGH",
    "session_low_line": "SESSION LOW",
    "equilibrium_line": "EQUILIBRIUM",
    "pivot_line": "PIVOT",
    "vwap_line": "VWAP",
    "daily_open_line": "DAILY OPEN",
    "htf_level_line": "HTF LEVEL",
    "bos_line": "BOS",
    "choch_line": "CHoCH",
}

ZONE_LABELS: dict[str, str] = {
    "fvg_box": "FVG",
    "ob_box": "OB",
    "breaker_box": "BREAKER",
    "premium_zone": "PREMIUM",
    "discount_zone": "DISCOUNT",
    "ote_zone": "OTE",
    "sr_zone": "SR ZONE",
    "supply_demand_zone": "SUPPLY/DEMAND",
    "volume_profile_zone": "VOLUME PROFILE",
    "ichimoku_zone": "ICHIMOKU",
}

PLAN_LINES = (
    ("entry_line", "ENTRY"),
    ("sl_line", "SL"),
    ("tp1_line", "TP1"),
    ("tp2_line", "TP2"),
    ("tp3_line", "TP3"),
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _midpoint(top: float | None, bottom: float | None) -> float | None:
    if top is None:
        return bottom
    if bottom is None:
        return top
    return (float(top) + float(bottom)) / 2.0


def _stable_id(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]


def _annotation(
    annotation_type: str,
    *,
    category: str,
    direction: str,
    status: str,
    label: str,
    top: float | None,
    bottom: float | None,
    price: float | None,
    start_time: str | None,
    end_time: str | None,
    extend_right: bool,
    strength: float,
    notes: str,
    meta: dict[str, Any],
    uid: str,
) -> dict[str, Any]:
    return {
        "id": f"{annotation_type}-{uid}",
        "type": annotation_type,
        "category": str(category or "classic"),
        "direction": direction,
        "status": status,
        "label": label,
        "top": top,
        "bottom": bottom,
        "price": price,
        "start_time": start_time,
        "end_time": end_time,
        "extend_right": bool(extend_right),
        "strength": round(float(_number(strength) or 0.0), 2),
        "notes": str(notes or ""),
        "meta": json_safe(meta or {}),
    }


def _line_label(annotation_type: str, price: float | None, fallback: str) -> str:
    base = LEVEL_LABELS.get(annotation_type, fallback)
    return f"{base} {price:.2f}" if price is not None else base


def _swing_time(meta: dict[str, Any], default: Any) -> Any:
    for key in ("pivot_time", "level_time", "ob_time"):
        if meta.get(key) is not None:
            return meta[key]
    return default


def _zone_geometry(status: str, end_time: Any, last_time: Any) -> tuple[str | None, bool]:
    """Active zones extend to the right edge; resolved zones keep their stamp."""
    if status == "active":
        return ts_to_str(last_time) if last_time is not None else ts_to_str(end_time), True
    return ts_to_str(end_time), False


def _plan_lines(result: Any, df: pd.DataFrame, last_time: Any) -> list[dict[str, Any]]:
    entry = _midpoint(
        float(result.entry_zone[0]) if result.entry_zone else None,
        float(result.entry_zone[1]) if result.entry_zone else None,
    )
    if entry is None and len(df):
        entry = _number(df["Close"].iloc[-1])

    values: dict[str, float | None] = {
        "entry_line": entry,
        "sl_line": _number(result.stop_loss),
        "tp1_line": _number(result.take_profit1),
        "tp2_line": _number(result.take_profit2),
        "tp3_line": _number(result.take_profit3),
    }
    out: list[dict[str, Any]] = []
    for annotation_type, label in PLAN_LINES:
        price = values[annotation_type]
        if price is None:
            continue
        out.append(_annotation(
            annotation_type,
            category="signal",
            direction="bullish" if result.action == "BUY" else
            "bearish" if result.action == "SELL" else "neutral",
            status="active",
            label=label,
            top=price,
            bottom=price,
            price=price,
            start_time=ts_to_str(last_time),
            end_time=ts_to_str(last_time),
            extend_right=True,
            strength=float(result.confidence),
            notes=f"{result.action} {result.grade} plan: {label} at {price:.2f}",
            meta={"action": result.action, "grade": result.grade, "price": price},
            uid=_stable_id(annotation_type, ts_to_str(last_time), f"{price:.6f}"),
        ))
    return out


def build_annotations(
    df: pd.DataFrame,
    concepts: list[Any],
    result: Any,
    last_time: Any,
) -> list[dict[str, Any]]:
    """Convert detected concepts plus the active plan into chart drawings."""
    out: list[dict[str, Any]] = []
    last_high: float | None = None
    last_low: float | None = None

    for concept in concepts:
        ctype = concept.type
        meta = json_safe(concept.meta or {}) if isinstance(concept.meta, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        top = _number(concept.top)
        bottom = _number(concept.bottom)
        direction = concept.direction
        status = concept.status
        start = ts_to_str(concept.start_time)
        end = ts_to_str(concept.end_time)
        strength = float(concept.strength)
        notes = concept.notes
        category = concept.category

        if ctype in ZONE_MAP or ctype in ("premium_discount", "ote"):
            if ctype == "premium_discount":
                annotation_type = "discount_zone" if direction == "bullish" else "premium_zone"
            elif ctype == "ote":
                annotation_type = "ote_zone"
            else:
                annotation_type = ZONE_MAP[ctype]
            end_time, extend_right = _zone_geometry(status, concept.end_time, last_time)
            prefix = {"bullish": "BULL", "bearish": "BEAR"}.get(direction, "")
            base_label = ZONE_LABELS.get(annotation_type, annotation_type.upper())
            if ctype == "ifvg":
                base_label = "IFVG"
            if ctype == "supply_demand" and meta.get("zone"):
                base_label = str(meta["zone"]).upper()
            label = f"{prefix} {base_label}".strip() if prefix and ctype in ("fvg", "ifvg", "order_block") else base_label
            out.append(_annotation(
                annotation_type,
                category=category,
                direction=direction,
                status=status,
                label=label,
                top=top,
                bottom=bottom,
                price=_midpoint(top, bottom),
                start_time=start,
                end_time=end_time,
                extend_right=extend_right,
                strength=strength,
                notes=notes,
                meta=meta,
                uid=_stable_id(annotation_type, start, end, top, bottom, direction, notes),
            ))
            continue

        if ctype == "dealing_range":
            equilibrium = _number(meta.get("equilibrium"))
            if equilibrium is None:
                equilibrium = _midpoint(top, bottom)
            if equilibrium is not None:
                out.append(_annotation(
                    "equilibrium_line",
                    category=category,
                    direction="neutral",
                    status="active",
                    label=_line_label("equilibrium_line", equilibrium, "EQUILIBRIUM"),
                    top=equilibrium,
                    bottom=equilibrium,
                    price=equilibrium,
                    start_time=start,
                    end_time=ts_to_str(last_time),
                    extend_right=True,
                    strength=strength,
                    notes=notes or "Dealing range equilibrium (50%)",
                    meta={**meta, "equilibrium": equilibrium},
                    uid=_stable_id("equilibrium_line", start, f"{equilibrium:.6f}"),
                ))
            continue

        if ctype in LINE_MAP or ctype in ("internal_structure", "external_structure"):
            if ctype in ("internal_structure", "external_structure"):
                annotation_type = "choch_line" if meta.get("kind") == "choch" else "bos_line"
            else:
                annotation_type = LINE_MAP[ctype]
            price = top if top is not None else bottom
            line_start = start
            if annotation_type in ("bos_line", "choch_line"):
                line_start = ts_to_str(_swing_time(meta, concept.start_time))
            out.append(_annotation(
                annotation_type,
                category=category,
                direction=direction,
                status=status,
                label=_line_label(annotation_type, price, annotation_type.upper()),
                top=price,
                bottom=price,
                price=price,
                start_time=line_start,
                end_time=end,
                extend_right=False,
                strength=strength,
                notes=notes,
                meta=meta,
                uid=_stable_id(annotation_type, line_start, end, price, direction, notes),
            ))
            continue

        if ctype in ("swing_high", "swing_low"):
            level = top if top is not None else bottom
            if ctype == "swing_high":
                swing_label = "HH" if last_high is not None and level is not None and level > last_high else "LH"
                if level is not None:
                    last_high = level
            else:
                swing_label = "HL" if last_low is not None and level is not None and level > last_low else "LL"
                if level is not None:
                    last_low = level
            if last_high is None and last_low is None:
                continue
            out.append(_annotation(
                "swing_label",
                category=category,
                direction=direction,
                status=status,
                label=swing_label,
                top=top,
                bottom=bottom,
                price=level,
                start_time=start,
                end_time=end,
                extend_right=False,
                strength=strength,
                notes=f"{swing_label} at {level:.2f}" if level is not None else swing_label,
                meta={**meta, "swing_label": swing_label},
                uid=_stable_id("swing_label", start, end, level, swing_label),
            ))
            continue

        if ctype in MARKER_MAP or ctype in CANDLESTICK_TYPES or ctype in PATTERN_TYPES:
            if ctype in CANDLESTICK_TYPES:
                annotation_type = "candlestick_marker"
            elif ctype in PATTERN_TYPES:
                annotation_type = "chart_pattern_marker"
            else:
                annotation_type = MARKER_MAP[ctype]
            if annotation_type == "killzone_shade":
                zone = str(meta.get("zone") or "session").upper()
                out.append(_annotation(
                    annotation_type,
                    category=category,
                    direction=direction,
                    status=status,
                    label=f"{zone} KZ",
                    top=top,
                    bottom=bottom,
                    price=None,
                    start_time=end,
                    end_time=end,
                    extend_right=False,
                    strength=strength,
                    notes=notes,
                    meta=meta,
                    uid=_stable_id(annotation_type, end, zone),
                ))
                continue
            if annotation_type == "sweep_marker":
                price = _number(meta.get("swept_price"))
                if price is None:
                    price = _midpoint(top, bottom)
                label = "SWEEP"
            elif annotation_type == "displacement_marker":
                price = top if top is not None else bottom
                label = "DISPLACEMENT"
            else:
                price = _midpoint(top, bottom)
                label = ctype.replace("_", " ").upper()
            out.append(_annotation(
                annotation_type,
                category=category,
                direction=direction,
                status=status,
                label=label,
                top=top,
                bottom=bottom,
                price=price,
                start_time=start,
                end_time=end,
                extend_right=False,
                strength=strength,
                notes=notes,
                meta=meta,
                uid=_stable_id(annotation_type, start, end, top, bottom, price, notes),
            ))
            continue

    # Signal plan lines are drawn once by the frontend from the live signal.
    # Emitting them here duplicated every line on the chart.
    return out


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    number = _number(value)
    return default if number is None else number


def build_signal_markers(timeframe: str) -> list[dict[str, Any]]:
    """Stored signals for ``timeframe``, newest first, with parsed explanation."""
    rows: list[dict[str, Any]]
    try:
        rows = db.list_signals(limit=200)
    except Exception as exc:  # noqa: BLE001 - history is optional on the chart
        logger.warning("could not load signal history: %s", exc)
        return []

    markers: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("timeframe", "")) != timeframe:
            continue

        explanation: Any = row.get("explanation")
        if isinstance(explanation, str):
            try:
                explanation = json.loads(explanation)
            except (TypeError, ValueError):
                explanation = {"summary": explanation}
        if not isinstance(explanation, dict):
            explanation = {"summary": str(explanation)} if explanation is not None else None

        action = str(row.get("action") or "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        status = "closed" if str(row.get("status", "open")).lower() == "closed" else "open"
        price = _as_float(row.get("price"), 0.0) or 0.0

        key = (str(row.get("signal_time") or row.get("created_at") or ""),
               action, round(float(_as_float(row.get("entry"), price) or price), 2),
               round(float(_as_float(row.get("stop_loss"), price) or price), 2))
        if key in seen:
            continue
        seen.add(key)
        markers.append({
            "id": int(row.get("id") or 0),
            "time": ts_to_str(row.get("signal_time") or row.get("created_at")),
            "action": action,
            "grade": str(row.get("grade") or ""),
            "price": float(price),
            "status": status,
            "hit": row.get("hit"),
            "entry": _as_float(row.get("entry"), price) or 0.0,
            "stop_loss": _as_float(row.get("stop_loss"), price) or 0.0,
            "take_profit1": _as_float(row.get("take_profit1"), price) or 0.0,
            "take_profit2": _as_float(row.get("take_profit2"), price) or 0.0,
            "take_profit3": _number(row.get("take_profit3")),
            "result_pct": _number(row.get("result_pct")),
            "explanation": json_safe(explanation) if explanation is not None else None,
        })

    markers.sort(key=lambda marker: (marker["time"] or "", marker["id"]), reverse=True)
    return markers[:15]


def build_payload(
    entry: Any,
    timeframe: str,
    *,
    show_all: bool = False,
    types: str | None = None,
    category: str | None = None,
    mode: str = "clean",
) -> dict[str, Any]:
    """Detect everything, then declutter drawings for the selected chart mode."""
    df = entry.df
    context = concept_context.build_context(timeframe) or {}
    concepts = detect_all(df, context)
    try:
        result = confluence(concepts, df, ml_probability=None)
    except Exception as exc:  # noqa: BLE001 - a plan failure must not hide concept drawings
        logger.warning("chart plan computation failed: %s", exc)
        result = None

    last_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
    annotations = build_annotations(df, concepts, result, last_time) if result is not None else []

    from app.services.chart_declutter import declutter

    decluttered = declutter(
        annotations, df, timeframe=timeframe, mode=mode, show_all=show_all,
        detected=len(concepts),
    )
    annotations = decluttered["annotations"]
    if types:
        wanted = {item.strip() for item in types.split(",") if item.strip()}
        if wanted:
            annotations = [a for a in annotations if a["type"] in wanted]
    if category:
        wanted_category = category.strip().lower()
        annotations = [a for a in annotations
                       if str(a["category"]).lower() == wanted_category]

    return {
        "timeframe": timeframe,
        "symbol": SYMBOL,
        "annotations": annotations,
        "signal_markers": build_signal_markers(timeframe),
        "total": decluttered["total"],
        "shown": len(annotations),
        "detected": decluttered["detected"],
        "mode": decluttered["mode"],
        "show_all": show_all,
        "stale": bool(getattr(entry, "stale", False)),
        "error": getattr(entry, "error", None),
        "disclaimer": DISCLAIMER,
    }
