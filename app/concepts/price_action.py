"""Causal price-action concept detectors.

Every detector is a pure function ``detect_x(df, ...) -> list[Concept]`` that is
registered under a stable type name in the shared registry. Patterns are stamped
at their confirming bar, so geometry only ever depends on data at or before that
bar. Candlestick signals additionally require a nearby context level (a supplied
zone, or a causal swing point) so isolated candles in empty price are ignored.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.concepts.base import (
    atr,
    body,
    capped,
    candle_range,
    ensure_frame,
    is_bear,
    is_bull,
    lower_wick,
    make_concept,
    overlap,
    pivots_before,
    register,
    signed_body,
    swing_points,
    upper_wick,
)

MAX_CONCEPTS = 12
PIVOT_LOOKBACK = 2


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _all_pivots(df: pd.DataFrame) -> list[dict[str, Any]]:
    return swing_points(df, lookback=PIVOT_LOOKBACK, confirm=1)


def _usable(pivots: list[dict[str, Any]], i: int) -> list[dict[str, Any]]:
    """Pivots whose whole confirmation window sits at or before bar ``i``.

    Requiring ``index + lookback <= i`` (rather than only ``confirm_index <= i``)
    makes the result identical whether the frame ends at ``i`` or continues,
    which keeps detection strictly causal under truncation.
    """
    return [p for p in pivots if p["index"] + PIVOT_LOOKBACK <= i]


def _atr_at(df: pd.DataFrame, a_s: pd.Series, i: int) -> float:
    try:
        f = float(a_s.iloc[i])
        if np.isfinite(f) and f > 0:
            return f
    except (IndexError, TypeError, ValueError):
        pass
    lo = max(0, i - 13)
    try:
        f = float((df["High"] - df["Low"]).iloc[lo:i + 1].mean())
        if np.isfinite(f) and f > 0:
            return f
    except (TypeError, ValueError):
        pass
    return 1.0


def _strength(base: float, magnitude: float, a: float,
              context: bool = False) -> float:
    intensity = 0.0
    if np.isfinite(a) and a > 0 and np.isfinite(magnitude):
        intensity = float(np.clip(magnitude / a, 0.0, 2.5)) / 2.5
    value = base + 18.0 * intensity + (8.0 if context else 0.0)
    return float(np.clip(value, 5.0, 95.0))


def _zone_bounds(zone: Any) -> tuple[float, float] | None:
    if not isinstance(zone, dict):
        return None

    def pick(*keys: str) -> float | None:
        for key in keys:
            if key in zone and zone[key] is not None:
                try:
                    value = float(zone[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
        return None

    top = pick("top", "high", "price_top", "resistance", "price")
    bottom = pick("bottom", "low", "price_bottom", "support", "price")
    if top is None and bottom is None:
        return None
    if top is None:
        top = bottom
    if bottom is None:
        bottom = top
    return (float(top), float(bottom)) if top >= bottom else (float(bottom), float(top))


def _match_context(band_top: float, band_bottom: float, i: int,
                   zones: list[dict] | None,
                   pivots: list[dict[str, Any]], a: float) -> dict[str, Any] | None:
    """Return the context level overlapping a pattern's price band, if any."""
    if zones:
        for zone in zones:
            bounds = _zone_bounds(zone)
            if bounds is None:
                continue
            if overlap(band_top, band_bottom, bounds[0], bounds[1]):
                return {
                    "source": "zone",
                    "kind": str(zone.get("kind", "level")),
                    "top": bounds[0],
                    "bottom": bounds[1],
                }
        return None

    usable = _usable(pivots, i)
    if not usable:
        return None
    mid = (float(band_top) + float(band_bottom)) / 2.0
    tol = max(0.6 * a, 0.004 * abs(mid))
    recent = pivots_before(usable, i)
    candidates = ([recent] if recent is not None else [])
    candidates.extend(p for p in usable if p is not recent)
    for p in candidates:
        level = float(p["price"])
        if not np.isfinite(level):
            continue
        if overlap(band_top, band_bottom, level + tol, level - tol):
            return {
                "source": "swing",
                "kind": str(p["kind"]),
                "top": level + tol,
                "bottom": level - tol,
            }
    return None


def _context_note(ctx: dict[str, Any] | None) -> str:
    if not ctx:
        return ""
    return (f" near {ctx['kind']} ({ctx['source']} level "
            f"{ctx['bottom']:.2f}-{ctx['top']:.2f})")


def _arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "o": df["Open"].to_numpy(float),
        "h": df["High"].to_numpy(float),
        "l": df["Low"].to_numpy(float),
        "c": df["Close"].to_numpy(float),
        "body": body(df).to_numpy(float),
        "sb": signed_body(df).to_numpy(float),
        "uw": upper_wick(df).to_numpy(float),
        "lw": lower_wick(df).to_numpy(float),
        "rng": candle_range(df).to_numpy(float),
        "bull": is_bull(df).to_numpy(bool),
        "bear": is_bear(df).to_numpy(bool),
    }


def _concept(type_: str, direction: str, start: Any, end: Any,
             top: float, bottom: float, strength: float,
             notes: str, meta: dict[str, Any]) -> Any:
    return make_concept(
        type_=type_, direction=direction, start_time=start, end_time=end,
        top=top, bottom=bottom, strength=strength, status="active",
        notes=notes, category="price_action", meta=meta,
    )


# --------------------------------------------------------------------------- #
# candlestick patterns
# --------------------------------------------------------------------------- #
@register("engulfing", "price_action")
def detect_engulfing(df: pd.DataFrame,
                     zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 2:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(1, n):
        if i - last < 2:
            continue
        p, c = i - 1, i
        bull_eng = (arr["bear"][p] and arr["bull"][c]
                    and arr["c"][c] > arr["o"][p] and arr["o"][c] < arr["c"][p])
        bear_eng = (arr["bull"][p] and arr["bear"][c]
                    and arr["o"][c] > arr["c"][p] and arr["c"][c] < arr["o"][p])
        if not (bull_eng or bear_eng):
            continue
        if arr["body"][c] <= arr["body"][p]:
            continue
        top = float(max(arr["h"][p], arr["h"][c]))
        bottom = float(min(arr["l"][p], arr["l"][c]))
        a = _atr_at(df, a_s, c)
        ctx = _match_context(top, bottom, c, zones, pivots, a)
        if ctx is None:
            continue
        direction = "bullish" if bull_eng else "bearish"
        strength = _strength(56, arr["body"][c] - arr["body"][p], a, True)
        meta = {"confirm_index": c, "context": ctx}
        found.append(_concept(
            "engulfing", direction, df.index[p], df.index[c], top, bottom,
            strength, f"{direction} engulfing{_context_note(ctx)}", meta,
        ))
        last = c
    return capped(found, MAX_CONCEPTS)


@register("pin_bar", "price_action")
def detect_pin_bar(df: pd.DataFrame,
                   zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 1:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(n):
        if i - last < 2:
            continue
        bd, rng = arr["body"][i], arr["rng"][i]
        if not (np.isfinite(bd) and bd > 0 and np.isfinite(rng) and rng > 0):
            continue
        hammer = (arr["lw"][i] >= 2 * bd and arr["lw"][i] >= 0.5 * rng
                  and arr["lw"][i] > arr["uw"][i])
        star = (arr["uw"][i] >= 2 * bd and arr["uw"][i] >= 0.5 * rng
                and arr["uw"][i] > arr["lw"][i])
        if not (hammer or star):
            continue
        top, bottom = float(arr["h"][i]), float(arr["l"][i])
        a = _atr_at(df, a_s, i)
        ctx = _match_context(top, bottom, i, zones, pivots, a)
        if ctx is None:
            continue
        direction = "bullish" if hammer else "bearish"
        label = "hammer" if hammer else "shooting star"
        wick = arr["lw"][i] if hammer else arr["uw"][i]
        strength = _strength(54, wick, a, True)
        meta = {"pattern": label, "confirm_index": i, "context": ctx}
        found.append(_concept(
            "pin_bar", direction, df.index[i], df.index[i], top, bottom,
            strength, f"{label} pin bar{_context_note(ctx)}", meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("doji", "price_action")
def detect_doji(df: pd.DataFrame,
                zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 1:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(n):
        if i - last < 2:
            continue
        rng = arr["rng"][i]
        if not (np.isfinite(rng) and rng > 0):
            continue
        if arr["body"][i] > 0.1 * rng:
            continue
        top, bottom = float(arr["h"][i]), float(arr["l"][i])
        a = _atr_at(df, a_s, i)
        ctx = _match_context(top, bottom, i, zones, pivots, a)
        if ctx is None:
            continue
        strength = _strength(38, arr["uw"][i] + arr["lw"][i], a, True)
        meta = {"confirm_index": i, "context": ctx}
        found.append(_concept(
            "doji", "neutral", df.index[i], df.index[i], top, bottom,
            strength, f"doji / indecision{_context_note(ctx)}", meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("morning_star", "price_action")
def detect_morning_star(df: pd.DataFrame,
                        zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 3:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(2, n):
        if i - last < 3:
            continue
        s1, s2, s3 = i - 2, i - 1, i
        if not (arr["bear"][s1] and arr["bull"][s3]):
            continue
        if not (np.isfinite(arr["body"][s1]) and arr["body"][s1] > 0):
            continue
        if arr["body"][s2] > 0.35 * arr["body"][s1]:
            continue
        mid = (arr["o"][s1] + arr["c"][s1]) / 2.0
        if arr["c"][s3] <= mid or arr["body"][s3] < 0.5 * arr["body"][s1]:
            continue
        top = float(max(arr["h"][s1], arr["h"][s2], arr["h"][s3]))
        bottom = float(min(arr["l"][s1], arr["l"][s2], arr["l"][s3]))
        a = _atr_at(df, a_s, s3)
        ctx = _match_context(top, bottom, s3, zones, pivots, a)
        if ctx is None:
            continue
        strength = _strength(64, arr["body"][s3], a, True)
        meta = {"confirm_index": s3, "context": ctx}
        found.append(_concept(
            "morning_star", "bullish", df.index[s1], df.index[s3], top, bottom,
            strength, f"morning star reversal{_context_note(ctx)}", meta,
        ))
        last = s3
    return capped(found, MAX_CONCEPTS)


@register("evening_star", "price_action")
def detect_evening_star(df: pd.DataFrame,
                        zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 3:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(2, n):
        if i - last < 3:
            continue
        s1, s2, s3 = i - 2, i - 1, i
        if not (arr["bull"][s1] and arr["bear"][s3]):
            continue
        if not (np.isfinite(arr["body"][s1]) and arr["body"][s1] > 0):
            continue
        if arr["body"][s2] > 0.35 * arr["body"][s1]:
            continue
        mid = (arr["o"][s1] + arr["c"][s1]) / 2.0
        if arr["c"][s3] >= mid or arr["body"][s3] < 0.5 * arr["body"][s1]:
            continue
        top = float(max(arr["h"][s1], arr["h"][s2], arr["h"][s3]))
        bottom = float(min(arr["l"][s1], arr["l"][s2], arr["l"][s3]))
        a = _atr_at(df, a_s, s3)
        ctx = _match_context(top, bottom, s3, zones, pivots, a)
        if ctx is None:
            continue
        strength = _strength(64, arr["body"][s3], a, True)
        meta = {"confirm_index": s3, "context": ctx}
        found.append(_concept(
            "evening_star", "bearish", df.index[s1], df.index[s3], top, bottom,
            strength, f"evening star reversal{_context_note(ctx)}", meta,
        ))
        last = s3
    return capped(found, MAX_CONCEPTS)


@register("inside_bar", "price_action")
def detect_inside_bar(df: pd.DataFrame,
                      zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 2:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(1, n):
        if i - last < 2:
            continue
        p = i - 1
        if not (arr["h"][i] < arr["h"][p] and arr["l"][i] > arr["l"][p]):
            continue
        top, bottom = float(arr["h"][i]), float(arr["l"][i])
        a = _atr_at(df, a_s, i)
        ctx = _match_context(top, bottom, i, zones, pivots, a)
        if ctx is None:
            continue
        direction = ("bullish" if arr["bull"][p]
                     else "bearish" if arr["bear"][p] else "neutral")
        strength = _strength(42, arr["h"][p] - arr["l"][p], a, True)
        meta = {"confirm_index": i, "mother_index": p, "context": ctx}
        found.append(_concept(
            "inside_bar", direction, df.index[p], df.index[i], top, bottom,
            strength, f"inside bar ({direction} mother){_context_note(ctx)}", meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("outside_bar", "price_action")
def detect_outside_bar(df: pd.DataFrame,
                       zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 2:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(1, n):
        if i - last < 2:
            continue
        p = i - 1
        if not (arr["h"][i] > arr["h"][p] and arr["l"][i] < arr["l"][p]):
            continue
        top, bottom = float(arr["h"][i]), float(arr["l"][i])
        a = _atr_at(df, a_s, i)
        ctx = _match_context(top, bottom, i, zones, pivots, a)
        if ctx is None:
            continue
        direction = ("bullish" if arr["bull"][i]
                     else "bearish" if arr["bear"][i] else "neutral")
        strength = _strength(48, arr["rng"][i], a, True)
        meta = {"confirm_index": i, "context": ctx}
        found.append(_concept(
            "outside_bar", direction, df.index[p], df.index[i], top, bottom,
            strength, f"outside / engulfing bar{_context_note(ctx)}", meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("three_soldiers", "price_action")
def detect_three_soldiers(df: pd.DataFrame,
                          zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 3:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(2, n):
        if i - last < 4:
            continue
        a0, b0, c0 = i - 2, i - 1, i
        if not (arr["bull"][a0] and arr["bull"][b0] and arr["bull"][c0]):
            continue
        if not (arr["c"][c0] > arr["c"][b0] > arr["c"][a0]):
            continue
        if arr["o"][b0] > arr["c"][a0] or arr["o"][c0] > arr["c"][b0]:
            continue
        bodies = [arr["body"][a0], arr["body"][b0], arr["body"][c0]]
        if not all(np.isfinite(v) and v >= 0.4 * arr["rng"][k]
                   for v, k in zip(bodies, (a0, b0, c0))):
            continue
        top = float(max(arr["h"][a0], arr["h"][b0], arr["h"][c0]))
        bottom = float(min(arr["l"][a0], arr["l"][b0], arr["l"][c0]))
        a = _atr_at(df, a_s, c0)
        ctx = _match_context(top, bottom, c0, zones, pivots, a)
        if ctx is None:
            continue
        strength = _strength(66, arr["body"][c0], a, True)
        meta = {"confirm_index": c0, "context": ctx}
        found.append(_concept(
            "three_soldiers", "bullish", df.index[a0], df.index[c0], top, bottom,
            strength, f"three white soldiers{_context_note(ctx)}", meta,
        ))
        last = c0
    return capped(found, MAX_CONCEPTS)


@register("three_crows", "price_action")
def detect_three_crows(df: pd.DataFrame,
                       zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 3:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(2, n):
        if i - last < 4:
            continue
        a0, b0, c0 = i - 2, i - 1, i
        if not (arr["bear"][a0] and arr["bear"][b0] and arr["bear"][c0]):
            continue
        if not (arr["c"][c0] < arr["c"][b0] < arr["c"][a0]):
            continue
        if arr["o"][b0] < arr["c"][a0] or arr["o"][c0] < arr["c"][b0]:
            continue
        bodies = [arr["body"][a0], arr["body"][b0], arr["body"][c0]]
        if not all(np.isfinite(v) and v >= 0.4 * arr["rng"][k]
                   for v, k in zip(bodies, (a0, b0, c0))):
            continue
        top = float(max(arr["h"][a0], arr["h"][b0], arr["h"][c0]))
        bottom = float(min(arr["l"][a0], arr["l"][b0], arr["l"][c0]))
        a = _atr_at(df, a_s, c0)
        ctx = _match_context(top, bottom, c0, zones, pivots, a)
        if ctx is None:
            continue
        strength = _strength(66, arr["body"][c0], a, True)
        meta = {"confirm_index": c0, "context": ctx}
        found.append(_concept(
            "three_crows", "bearish", df.index[a0], df.index[c0], top, bottom,
            strength, f"three black crows{_context_note(ctx)}", meta,
        ))
        last = c0
    return capped(found, MAX_CONCEPTS)


# --------------------------------------------------------------------------- #
# chart patterns
# --------------------------------------------------------------------------- #
@register("double_bottom", "price_action")
def detect_double_bottom(df: pd.DataFrame,
                         zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 8:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(6, n):
        if i - last < 4:
            continue
        usable = _usable(pivots, i)
        lows_p = [p for p in usable if p["kind"] == "low"]
        highs_p = [p for p in usable if p["kind"] == "high"]
        found_pair = None
        for ai in range(len(lows_p) - 1, -1, -1):
            l2 = lows_p[ai]
            for bi in range(ai - 1, -1, -1):
                l1 = lows_p[bi]
                if abs(l1["price"] - l2["price"]) > 0.02 * l2["price"]:
                    continue
                between = [h for h in highs_p
                           if l1["index"] < h["index"] < l2["index"]]
                if not between:
                    continue
                neck = max(h["price"] for h in between)
                bottom = min(l1["price"], l2["price"])
                if neck <= bottom:
                    continue
                found_pair = (l1, l2, neck, bottom)
                break
            if found_pair:
                break
        if not found_pair:
            continue
        l1, l2, neck, bottom = found_pair
        if i <= l2["index"] or i - l2["index"] > 30:
            continue
        if not (arr["c"][i] > neck and arr["c"][i - 1] <= neck):
            continue
        height = neck - bottom
        target = neck + height
        a = _atr_at(df, a_s, i)
        strength = _strength(60, height, a)
        meta = {
            "target": float(target), "neckline": float(neck),
            "breakout": float(arr["c"][i]), "height": float(height),
            "pivots": [l1["index"], l2["index"]], "confirm_index": i,
        }
        notes = (f"double bottom breakout above {neck:.2f}; measured target "
                 f"{target:.2f} (neckline + range height)")
        found.append(_concept(
            "double_bottom", "bullish", df.index[l1["index"]], df.index[i],
            float(neck), float(bottom), strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("double_top", "price_action")
def detect_double_top(df: pd.DataFrame,
                      zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 8:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(6, n):
        if i - last < 4:
            continue
        usable = _usable(pivots, i)
        highs_p = [p for p in usable if p["kind"] == "high"]
        lows_p = [p for p in usable if p["kind"] == "low"]
        found_pair = None
        for ai in range(len(highs_p) - 1, -1, -1):
            h2 = highs_p[ai]
            for bi in range(ai - 1, -1, -1):
                h1 = highs_p[bi]
                if abs(h1["price"] - h2["price"]) > 0.02 * h2["price"]:
                    continue
                between = [l for l in lows_p
                           if h1["index"] < l["index"] < h2["index"]]
                if not between:
                    continue
                neck = min(l["price"] for l in between)
                top = max(h1["price"], h2["price"])
                if neck >= top:
                    continue
                found_pair = (h1, h2, neck, top)
                break
            if found_pair:
                break
        if not found_pair:
            continue
        h1, h2, neck, top = found_pair
        if i <= h2["index"] or i - h2["index"] > 30:
            continue
        if not (arr["c"][i] < neck and arr["c"][i - 1] >= neck):
            continue
        height = top - neck
        target = neck - height
        a = _atr_at(df, a_s, i)
        strength = _strength(60, height, a)
        meta = {
            "target": float(target), "neckline": float(neck),
            "breakout": float(arr["c"][i]), "height": float(height),
            "pivots": [h1["index"], h2["index"]], "confirm_index": i,
        }
        notes = (f"double top breakdown below {neck:.2f}; measured target "
                 f"{target:.2f} (neckline - range height)")
        found.append(_concept(
            "double_top", "bearish", df.index[h1["index"]], df.index[i],
            float(top), float(neck), strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("head_shoulders", "price_action")
def detect_head_shoulders(df: pd.DataFrame,
                          zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 10:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(8, n):
        if i - last < 5:
            continue
        usable = _usable(pivots, i)
        highs_p = [p for p in usable if p["kind"] == "high"]
        lows_p = [p for p in usable if p["kind"] == "low"]
        formation = None
        for ci in range(len(highs_p) - 1, 1, -1):
            h3 = highs_p[ci]
            for bi in range(ci - 1, 0, -1):
                h2 = highs_p[bi]
                if h2["price"] <= h3["price"]:
                    continue
                for ai in range(bi - 1, -1, -1):
                    h1 = highs_p[ai]
                    if abs(h1["price"] - h3["price"]) > 0.05 * h3["price"]:
                        continue
                    v1 = [p for p in lows_p
                          if h1["index"] < p["index"] < h2["index"]]
                    v2 = [p for p in lows_p
                          if h2["index"] < p["index"] < h3["index"]]
                    if not v1 or not v2:
                        continue
                    valley1 = min(v1, key=lambda p: p["price"])
                    valley2 = min(v2, key=lambda p: p["price"])
                    neck = min(valley1["price"], valley2["price"])
                    if h2["price"] <= neck:
                        continue
                    formation = (h1, h2, h3, neck)
                    break
                if formation:
                    break
            if formation:
                break
        if not formation:
            continue
        h1, h2, h3, neck = formation
        if i <= h3["index"] or i - h3["index"] > 30:
            continue
        if not (arr["c"][i] < neck and arr["c"][i - 1] >= neck):
            continue
        height = h2["price"] - neck
        target = neck - height
        a = _atr_at(df, a_s, i)
        strength = _strength(64, height, a)
        meta = {
            "target": float(target), "neckline": float(neck),
            "head": float(h2["price"]), "breakout": float(arr["c"][i]),
            "height": float(height), "confirm_index": i,
        }
        notes = (f"head and shoulders breakdown below {neck:.2f}; measured "
                 f"target {target:.2f} (neckline - head height)")
        found.append(_concept(
            "head_shoulders", "bearish", df.index[h1["index"]], df.index[i],
            float(h2["price"]), float(neck), strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("inverse_head_shoulders", "price_action")
def detect_inverse_head_shoulders(df: pd.DataFrame,
                                  zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 10:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(8, n):
        if i - last < 5:
            continue
        usable = _usable(pivots, i)
        lows_p = [p for p in usable if p["kind"] == "low"]
        highs_p = [p for p in usable if p["kind"] == "high"]
        formation = None
        for ci in range(len(lows_p) - 1, 1, -1):
            l3 = lows_p[ci]
            for bi in range(ci - 1, 0, -1):
                l2 = lows_p[bi]
                if l2["price"] >= l3["price"]:
                    continue
                for ai in range(bi - 1, -1, -1):
                    l1 = lows_p[ai]
                    if abs(l1["price"] - l3["price"]) > 0.05 * l3["price"]:
                        continue
                    v1 = [p for p in highs_p
                          if l1["index"] < p["index"] < l2["index"]]
                    v2 = [p for p in highs_p
                          if l2["index"] < p["index"] < l3["index"]]
                    if not v1 or not v2:
                        continue
                    peak1 = max(v1, key=lambda p: p["price"])
                    peak2 = max(v2, key=lambda p: p["price"])
                    neck = max(peak1["price"], peak2["price"])
                    if l2["price"] >= neck:
                        continue
                    formation = (l1, l2, l3, neck)
                    break
                if formation:
                    break
            if formation:
                break
        if not formation:
            continue
        l1, l2, l3, neck = formation
        if i <= l3["index"] or i - l3["index"] > 30:
            continue
        if not (arr["c"][i] > neck and arr["c"][i - 1] <= neck):
            continue
        height = neck - l2["price"]
        target = neck + height
        a = _atr_at(df, a_s, i)
        strength = _strength(64, height, a)
        meta = {
            "target": float(target), "neckline": float(neck),
            "head": float(l2["price"]), "breakout": float(arr["c"][i]),
            "height": float(height), "confirm_index": i,
        }
        notes = (f"inverse head and shoulders breakout above {neck:.2f}; "
                 f"measured target {target:.2f} (neckline + head height)")
        found.append(_concept(
            "inverse_head_shoulders", "bullish", df.index[l1["index"]],
            df.index[i], float(neck), float(l2["price"]), strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


def _converging_pivots(usable: list[dict[str, Any]], i: int, flat: float):
    """Return (kind, high_ref, low_ref, height, start_index) for a converging
    two-high/two-low formation confirmed at or before ``i``."""
    highs_p = [p for p in usable if p["kind"] == "high"][-4:]
    lows_p = [p for p in usable if p["kind"] == "low"][-4:]
    if len(highs_p) < 2 or len(lows_p) < 2:
        return None
    h1, h2 = highs_p[-2], highs_p[-1]
    l1, l2 = lows_p[-2], lows_p[-1]
    if h2["index"] == h1["index"] or l2["index"] == l1["index"]:
        return None
    hs = (h2["price"] - h1["price"]) / (h2["index"] - h1["index"])
    ls = (l2["price"] - l1["price"]) / (l2["index"] - l1["index"])
    high_ref = max(h1["price"], h2["price"])
    low_ref = min(l1["price"], l2["price"])
    if high_ref <= low_ref:
        return None
    height = high_ref - low_ref
    start = min(h1["index"], l1["index"])
    return hs, ls, high_ref, low_ref, height, start


@register("triangle", "price_action")
def detect_triangle(df: pd.DataFrame,
                    zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 12:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(10, n):
        if i - last < 5:
            continue
        a = _atr_at(df, a_s, i)
        flat = 0.0015 * arr["c"][i]
        info = _converging_pivots(_usable(pivots, i), i, flat)
        if info is None:
            continue
        hs, ls, high_ref, low_ref, height, start = info
        if abs(hs) <= flat and ls > flat:
            variant = "ascending"
        elif abs(ls) <= flat and hs < -flat:
            variant = "descending"
        elif hs < -flat and ls > flat:
            variant = "symmetrical"
        else:
            continue
        bull = arr["c"][i] > high_ref and arr["c"][i - 1] <= high_ref
        bear = arr["c"][i] < low_ref and arr["c"][i - 1] >= low_ref
        if not (bull or bear):
            continue
        direction = "bullish" if bull else "bearish"
        target = float(arr["c"][i] + height if bull else arr["c"][i] - height)
        strength = _strength(58, height, a)
        meta = {
            "target": target, "variant": variant,
            "breakout": float(arr["c"][i]), "height": float(height),
            "confirm_index": i,
        }
        notes = (f"{variant} triangle {direction} breakout; measured target "
                 f"{target:.2f} (breakout +/− range height)")
        found.append(_concept(
            "triangle", direction, df.index[start], df.index[i],
            float(high_ref), float(low_ref), strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("wedge", "price_action")
def detect_wedge(df: pd.DataFrame,
                 zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 12:
        return []
    arr, pivots, a_s = _arrays(df), _all_pivots(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(10, n):
        if i - last < 5:
            continue
        a = _atr_at(df, a_s, i)
        flat = 0.0015 * arr["c"][i]
        info = _converging_pivots(_usable(pivots, i), i, flat)
        if info is None:
            continue
        hs, ls, high_ref, low_ref, height, start = info
        if hs > flat and ls > flat:
            variant, direction = "rising", "bearish"
            trigger = arr["c"][i] < low_ref and arr["c"][i - 1] >= low_ref
        elif hs < -flat and ls < -flat:
            variant, direction = "falling", "bullish"
            trigger = arr["c"][i] > high_ref and arr["c"][i - 1] <= high_ref
        else:
            continue
        if not trigger:
            continue
        target = float(arr["c"][i] - height if direction == "bearish"
                       else arr["c"][i] + height)
        strength = _strength(58, height, a)
        meta = {
            "target": target, "variant": variant,
            "breakout": float(arr["c"][i]), "height": float(height),
            "confirm_index": i,
        }
        notes = (f"{variant} wedge {direction} breakout; measured target "
                 f"{target:.2f} (breakout +/− range height)")
        found.append(_concept(
            "wedge", direction, df.index[start], df.index[i],
            float(high_ref), float(low_ref), strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("flag", "price_action")
def detect_flag(df: pd.DataFrame,
                zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    if n < 8:
        return []
    arr, a_s = _arrays(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(6, n):
        if i - last < 5:
            continue
        a = _atr_at(df, a_s, i)
        lo_j = max(1, i - 24)
        chosen = None
        for j in range(i - 3, lo_j - 1, -1):
            impulse = arr["sb"][j]
            if arr["bull"][j] and impulse > 1.5 * a:
                chosen = (j, "bullish")
            elif arr["bear"][j] and -impulse > 1.5 * a:
                chosen = (j, "bearish")
            if chosen:
                break
        if not chosen:
            continue
        j, direction = chosen
        if i - j < 3:
            continue
        cons_high = float(np.max(arr["h"][j + 1:i]))
        cons_low = float(np.min(arr["l"][j + 1:i]))
        cons_range = cons_high - cons_low
        impulse_range = float(arr["rng"][j])
        if not (np.isfinite(cons_range) and cons_range <= 0.8 * impulse_range):
            continue
        if direction == "bullish":
            if not (arr["c"][i] > cons_high and arr["c"][i - 1] <= cons_high):
                continue
            height = max(impulse_range, float(arr["sb"][j]))
            target = float(arr["c"][i] + height)
        else:
            if not (arr["c"][i] < cons_low and arr["c"][i - 1] >= cons_low):
                continue
            height = max(impulse_range, float(-arr["sb"][j]))
            target = float(arr["c"][i] - height)
        top = float(max(cons_high, arr["h"][j]))
        bottom = float(min(cons_low, arr["l"][j]))
        strength = _strength(60, height, a)
        meta = {
            "target": target, "variant": direction,
            "impulse_index": j, "breakout": float(arr["c"][i]),
            "height": float(height), "confirm_index": i,
        }
        notes = (f"{direction} flag continuation; measured target "
                 f"{target:.2f} (breakout +/− impulse height)")
        found.append(_concept(
            "flag", direction, df.index[j], df.index[i], top, bottom,
            strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


@register("range_breakout", "price_action")
def detect_range_breakout(df: pd.DataFrame,
                          zones: list[dict] | None = None) -> list[Any]:
    df = ensure_frame(df)
    n = len(df)
    window = 20
    if n < window + 1:
        return []
    arr, a_s = _arrays(df), atr(df)
    found: list[Any] = []
    last = -99
    for i in range(window, n):
        if i - last < 4:
            continue
        resistance = float(np.max(arr["h"][i - window:i]))
        support = float(np.min(arr["l"][i - window:i]))
        height = resistance - support
        if not (np.isfinite(height) and height > 0):
            continue
        a = _atr_at(df, a_s, i)
        bull = arr["c"][i] > resistance and arr["c"][i - 1] <= resistance
        bear = arr["c"][i] < support and arr["c"][i - 1] >= support
        if not (bull or bear):
            continue
        if arr["body"][i] < 0.25 * height:
            continue
        direction = "bullish" if bull else "bearish"
        target = float(arr["c"][i] + height if bull else arr["c"][i] - height)
        strength = _strength(62, height, a)
        meta = {
            "target": target, "resistance": resistance, "support": support,
            "breakout": float(arr["c"][i]), "height": float(height),
            "confirm_index": i,
        }
        notes = (f"range {direction} breakout "
                 f"({support:.2f}-{resistance:.2f}); measured target "
                 f"{target:.2f} (breakout +/− range height)")
        found.append(_concept(
            "range_breakout", direction, df.index[i - window], df.index[i],
            resistance, support, strength, notes, meta,
        ))
        last = i
    return capped(found, MAX_CONCEPTS)


__all__ = [
    "detect_engulfing", "detect_pin_bar", "detect_doji", "detect_morning_star",
    "detect_evening_star", "detect_inside_bar", "detect_outside_bar",
    "detect_three_soldiers", "detect_three_crows", "detect_double_top",
    "detect_double_bottom", "detect_head_shoulders",
    "detect_inverse_head_shoulders", "detect_triangle", "detect_wedge",
    "detect_flag", "detect_range_breakout",
]
