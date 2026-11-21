"""Ranking and decluttering for chart drawings.

Detection is untouched: every concept is still detected by ``detect_all``. This
module only decides which drawings deserve screen space, using a transparent
score (strength x freshness x proximity x timeframe alignment) and hard caps so
the chart can never be flooded. ``show_all`` bypasses the filter completely.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6, "1D": 1}

ZONE_TYPES = {
    "fvg_box", "ob_box", "breaker_box", "sr_zone", "supply_demand_zone",
    "volume_profile_zone", "ichimoku_zone", "ote_zone",
}
LIQUIDITY_TYPES = {
    "eqh_line", "eql_line", "pdh_line", "pdl_line", "pwh_line", "pwl_line",
    "session_high_line", "session_low_line",
}
MAJOR_LEVELS = {
    "pdh_line", "pdl_line", "pwh_line", "pwl_line", "session_high_line",
    "session_low_line", "sr_zone", "htf_level_line", "equilibrium_line",
}
MARKER_TYPES = {
    "sweep_marker", "displacement_marker", "candlestick_marker",
    "chart_pattern_marker", "swing_label",
}
PLAN_TYPES = {"entry_line", "sl_line", "tp1_line", "tp2_line", "tp3_line"}
REFERENCE_TYPES = {
    "eqh_line", "eql_line", "pdh_line", "pdl_line", "pwh_line", "pwl_line",
    "session_high_line", "session_low_line", "vwap_line", "pivot_line",
    "trendline", "channel_line", "htf_level_line", "daily_open_line",
    "equilibrium_line", "bos_line", "choch_line",
}
PREMIUM_TYPES = {"premium_zone", "discount_zone"}

MODE_BLOCKED: dict[str, set[str]] = {
    "smart_money": {
        "sr_zone", "trendline", "channel_line", "vwap_line", "pivot_line",
        "volume_profile_zone", "ichimoku_zone", "candlestick_marker",
        "chart_pattern_marker", "premium_zone", "discount_zone",
        "killzone_shade",
    },
    "classic": (
        (ZONE_TYPES - {"sr_zone", "volume_profile_zone", "ichimoku_zone"})
        | LIQUIDITY_TYPES | MARKER_TYPES
        | (REFERENCE_TYPES - {"trendline", "channel_line", "vwap_line", "pivot_line"})
        | PREMIUM_TYPES | {"killzone_shade"}
    ),
    "full": set(),
}
MODE_BLOCKED["clean"] = {
    "killzone_shade", "premium_zone", "discount_zone", "ichimoku_zone",
    "volume_profile_zone", "candlestick_marker", "chart_pattern_marker",
    "breaker_box", "ote_zone", "trendline", "channel_line", "pivot_line",
}
CLEAN_MIN_ITEMS = 15
CLEAN_MAX_ITEMS = 22

CAPS: dict[str, tuple[int, int]] = {
    "fvg_box": (3, 3),
    "ob_box": (2, 2),
    "breaker_box": (1, 1),
    "sr_zone": (2, 2),
    "supply_demand_zone": (2, 2),
    "volume_profile_zone": (1, 1),
    "ichimoku_zone": (1, 1),
    "ote_zone": (1, 1),
    "bos_line": (2, 2),
    "choch_line": (2, 2),
    "trendline": (1, 1),
    "channel_line": (1, 1),
    "vwap_line": (1, 1),
    "pivot_line": (1, 1),
    "equilibrium_line": (1, 1),
    "sweep_marker": (2, 2),
    "displacement_marker": (2, 2),
}
CLEAN_CAPS: dict[str, tuple[int, int]] = {
    **CAPS,
    "fvg_box": (4, 4), "ob_box": (3, 3), "sr_zone": (3, 3),
    "supply_demand_zone": (2, 2), "eqh_line": (3, 3), "eql_line": (3, 3),
    "pdh_line": (2, 2), "pdl_line": (2, 2), "bos_line": (3, 3),
    "choch_line": (2, 2), "htf_level_line": (2, 2),
    "sweep_marker": (2, 2), "displacement_marker": (2, 2),
}
LIQUIDITY_CAP = 4
SWING_CAP = 6
MARKER_CAP = 6
REFERENCE_CAP = 4
MIN_ZONE_ATR = 0.25
MAX_DISTANCE_ATR = 4.0
MIN_LINE_GAP_ATR = 0.15


def _to_ts(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    except (TypeError, ValueError):
        return None


def _daily_atr(df: pd.DataFrame) -> float:
    """Daily ATR used for distance/size decisions (independent of chart TF)."""
    try:
        daily = df.resample("1D").agg(
            {"High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        if len(daily) > 2:
            tr = pd.concat([
                daily["High"] - daily["Low"],
                (daily["High"] - daily["Close"].shift(1)).abs(),
                (daily["Low"] - daily["Close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            value = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
            if np.isfinite(value) and value > 0:
                return value
    except Exception:  # noqa: BLE001
        pass
    try:
        value = float((df["High"] - df["Low"]).tail(20).mean())
        return value if np.isfinite(value) and value > 0 else max(float(df["Close"].iloc[-1]) * 0.01, 1e-6)
    except Exception:  # noqa: BLE001
        return 1.0


def _midpoint(annotation: dict[str, Any]) -> float | None:
    try:
        top, bottom = annotation.get("top"), annotation.get("bottom")
        if top is not None and bottom is not None:
            return (float(top) + float(bottom)) / 2
        price = annotation.get("price")
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None


def _direction(annotation: dict[str, Any]) -> str:
    direction = str(annotation.get("direction", "neutral"))
    return direction if direction in {"bullish", "bearish"} else "neutral"


def _merge_zones(zones: list[dict[str, Any]], atr: float) -> list[dict[str, Any]]:
    """Merge same-type zones that overlap in price and touch in time."""
    if len(zones) < 2:
        return zones
    kept: list[dict[str, Any]] = []
    for zone in sorted(zones, key=lambda z: (str(z.get("start_time") or ""), str(z.get("end_time") or ""))):
        z_top, z_bottom = zone.get("top"), zone.get("bottom")
        if z_top is None or z_bottom is None:
            kept.append(zone)
            continue
        z_top, z_bottom = float(z_top), float(z_bottom)
        z_start = _to_ts(zone.get("start_time"))
        z_end = _to_ts(zone.get("end_time")) or z_start
        merged = False
        for existing in kept:
            if existing["type"] != zone["type"] or _direction(existing) != _direction(zone):
                continue
            e_top, e_bottom = float(existing["top"]), float(existing["bottom"])
            if min(z_top, e_top) - max(z_bottom, e_bottom) < -0.1 * atr:
                continue
            e_start = _to_ts(existing.get("start_time"))
            e_end = _to_ts(existing.get("end_time")) or e_start
            if z_start is None or e_start is None:
                continue
            if z_start > e_end + pd.Timedelta(days=1) or e_start > (z_end or z_start) + pd.Timedelta(days=1):
                continue
            existing["top"] = max(z_top, e_top)
            existing["bottom"] = min(z_bottom, e_bottom)
            existing["start_time"] = min(z_start, e_start).isoformat()
            existing["end_time"] = max(z_end or z_start, e_end or e_start).isoformat()
            existing["strength"] = max(float(existing.get("strength") or 0), float(zone.get("strength") or 0))
            existing["_score"] = max(float(existing.get("_score") or 0), float(zone.get("_score") or 0))
            existing.setdefault("_merged", 0)
            existing["_merged"] += 1
            merged = True
            break
        if not merged:
            kept.append(dict(zone))
    return kept


def declutter(
    annotations: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    timeframe: str = "1h",
    mode: str = "clean",
    show_all: bool = False,
    detected: int | None = None,
) -> dict[str, Any]:
    """Return a filtered copy of `annotations`; never mutate detector output."""
    raw = list(annotations or [])
    total = len(raw)
    if show_all:
        for item in raw:
            meta = dict(item.get("meta") or {})
            meta["axis_label"] = True
            meta["fade"] = 1.0
            item["meta"] = meta
        return {
            "annotations": sorted(raw, key=lambda a: (str(a.get("end_time") or ""), a.get("id", ""))),
            "shown": len(raw), "total": total, "detected": detected if detected is not None else total,
            "mode": "full" if mode not in MODE_BLOCKED else mode, "show_all": True,
        }

    mode = mode if mode in MODE_BLOCKED else "clean"
    blocked = MODE_BLOCKED[mode]
    caps = CLEAN_CAPS if mode == "clean" else CAPS
    price = float(df["Close"].iloc[-1]) if df is not None and len(df) else 0.0
    atr = _daily_atr(df) if df is not None and len(df) else max(price * 0.01, 1e-6)
    bars_per_day = BARS_PER_DAY.get(timeframe, 24)
    recency_bars = bars_per_day * 2
    max_age = bars_per_day * 5
    bar_seconds = bars_per_day and 86400 / bars_per_day
    index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True).tz_localize(None)) if df is not None and len(df) else None
    last_time = index[-1] if index is not None else None

    candidates: list[dict[str, Any]] = []
    for original in raw:
        item = dict(original)
        annotation_type = str(item.get("type", ""))
        if annotation_type in blocked:
            continue
        status = str(item.get("status", "active"))
        if status == "invalidated":
            continue
        if annotation_type in ZONE_TYPES and status != "active":
            continue

        end_ts = _to_ts(item.get("end_time")) or _to_ts(item.get("start_time"))
        start_ts = _to_ts(item.get("start_time")) or end_ts
        age = 0
        if index is not None and end_ts is not None:
            bar_index = int(index.searchsorted(end_ts, side="right")) - 1
            age = max(0, len(index) - 1 - bar_index)
        midpoint = _midpoint(item)
        dist = abs(price - midpoint) / atr if midpoint is not None and atr > 0 else 0.0
        major = annotation_type in MAJOR_LEVELS or str(item.get("category")) in {"mtf", "macro"}
        freshness = 0.5 ** (age / max(recency_bars, 1.0))
        proximity = 1.0 / (1.0 + dist / 2.0)
        htf = 1.2 if major else 1.0
        score = float(item.get("strength") or 0) / 100.0 * freshness * proximity * htf
        if annotation_type in PLAN_TYPES:
            score = 10.0

        if annotation_type in ZONE_TYPES:
            size = abs(float(item.get("top") or 0) - float(item.get("bottom") or 0)) / max(atr, 1e-6)
            if size < MIN_ZONE_ATR:
                continue
            if dist > MAX_DISTANCE_ATR and not major:
                continue
            if age > max_age and not major:
                continue
            if item.get("extend_right") and index is not None and start_ts is not None and last_time is not None:
                max_bars = int(bars_per_day * 3)
                cap = start_ts + pd.Timedelta(seconds=bar_seconds * max_bars)
                if end_ts is None or end_ts > cap or cap < last_time:
                    item["end_time"] = min(cap, last_time).isoformat()
                    item["extend_right"] = False

        item["_score"] = score
        item["_age"] = age
        item["_dist"] = dist
        meta = dict(item.get("meta") or {})
        meta["fade"] = round(float(np.clip(freshness, 0.35, 1.0)), 3)
        meta["axis_label"] = annotation_type in PLAN_TYPES
        item["meta"] = meta
        candidates.append(item)

    zones = [c for c in candidates if c["type"] in ZONE_TYPES]
    others = [c for c in candidates if c["type"] not in ZONE_TYPES]
    merged = _merge_zones(zones, atr)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in merged:
        position = "neutral"
        if item["type"] == "sr_zone":
            mid = _midpoint(item)
            position = "above" if mid is not None and mid > price else "below"
        key = (item["type"], _direction(item), position)
        groups.setdefault(key, []).append(item)

    kept: list[dict[str, Any]] = []
    for key, items in groups.items():
        cap = caps.get(key[0], (2, 2))
        limit = cap[0] if key[1] == "bullish" else cap[1] if key[1] == "bearish" else 1
        if key[0] == "sr_zone":
            limit = cap[0] if key[2] == "above" else cap[1]
        items.sort(key=lambda a: a["_score"], reverse=True)
        kept.extend(items[:limit])

    liquidity = [c for c in others if c["type"] in LIQUIDITY_TYPES]
    liquidity.sort(key=lambda a: a["_score"], reverse=True)
    kept.extend(liquidity[:6 if mode == "clean" else LIQUIDITY_CAP])
    others = [c for c in others if c["type"] not in LIQUIDITY_TYPES]

    plan = [c for c in others if c["type"] in PLAN_TYPES]
    kept.extend(plan)
    others = [c for c in others if c["type"] not in PLAN_TYPES]

    references = [c for c in others if c["type"] in REFERENCE_TYPES]
    references.sort(key=lambda a: a["_score"], reverse=True)
    selected_refs: list[dict[str, Any]] = []
    last_price: float | None = None
    for ref in references:
        value = _midpoint(ref)
        if value is None:
            continue
        if last_price is not None and abs(value - last_price) < MIN_LINE_GAP_ATR * atr:
            continue
        selected_refs.append(ref)
        last_price = value
        if len(selected_refs) >= (6 if mode == "clean" else REFERENCE_CAP):
            break
    for ref in selected_refs:
        meta = dict(ref.get("meta") or {})
        meta["axis_label"] = len(selected_refs) <= 3
        ref["meta"] = meta
    kept.extend(selected_refs)
    others = [c for c in others if c["type"] not in REFERENCE_TYPES]

    swings = sorted((c for c in others if c["type"] == "swing_label"),
                    key=lambda a: str(a.get("end_time") or ""), reverse=True)[:SWING_CAP]
    kept.extend(swings)
    markers = sorted((c for c in others if c["type"] in MARKER_TYPES),
                     key=lambda a: str(a.get("end_time") or ""), reverse=True)[:MARKER_CAP]
    kept.extend(markers)
    rest = [c for c in others if c["type"] not in {"swing_label"} | MARKER_TYPES]
    for item in rest:
        cap = caps.get(item["type"], (2, 2))
        direction = _direction(item)
        limit = cap[0] if direction == "bullish" else cap[1] if direction == "bearish" else 1
        group = [c for c in kept if c["type"] == item["type"] and _direction(c) == direction]
        if len(group) < limit:
            kept.append(item)

    kept.sort(key=lambda a: (a.get("type") in PLAN_TYPES, a.get("_score", 0)), reverse=True)
    if mode == "clean" and len(kept) > CLEAN_MAX_ITEMS:
        priority = {t: 3 for t in PLAN_TYPES}
        priority.update({
            "htf_level_line": 2, "sr_zone": 2, "vwap_line": 2,
            "eqh_line": 1, "eql_line": 1, "pdh_line": 1, "pdl_line": 1,
        })
        kept.sort(key=lambda a: (priority.get(a["type"], 0), a.get("_score", 0)), reverse=True)
        kept = kept[:CLEAN_MAX_ITEMS]

    labelled = 0
    last_labelled_price: float | None = None
    gap = max(atr * 0.35, price * 0.0005, 1e-6)
    for item in sorted(kept, key=lambda a: (
        a.get("type") not in PLAN_TYPES,
        _midpoint(a) if _midpoint(a) is not None else 0.0,
    )):
        midpoint = _midpoint(item)
        if midpoint is None or labelled >= 8:
            item["meta"]["axis_label"] = False
            continue
        show = last_labelled_price is None or abs(midpoint - last_labelled_price) >= gap
        item["meta"]["axis_label"] = show
        if show:
            last_labelled_price = midpoint
            labelled += 1

    for item in kept:
        item.pop("_score", None)
        item.pop("_age", None)
        item.pop("_dist", None)
        item.pop("_merged", None)
    return {
        "annotations": kept,
        "shown": len(kept),
        "total": total,
        "detected": detected if detected is not None else total,
        "mode": mode,
        "show_all": False,
    }
