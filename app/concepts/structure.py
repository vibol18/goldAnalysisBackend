"""Causal market-structure concept detectors.

Swings are detected as confirmed fractals via ``swing_points``. Break concepts
(BOS / CHoCH / MSS) are stamped at the breaking bar and only ever look at
confirmed pivots whose confirmation bar is at or before that bar. Status is
determined at the same bar, so a concept's geometry and status are fully fixed
once its ``end_time`` has printed — editing later candles cannot rewrite it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.concepts.base import (
    Concept,
    atr,
    capped,
    candle_range,
    displacement_mask,
    make_concept,
    pivots_before,
    register,
    swing_points,
    ts_to_str,
)

CATEGORY = "structure"

MAX_SWINGS = 40
MAX_RECENT = 10
SWING_LOOKBACK = 2
TREND_LOOKBACK = 2
BREAK_LOOKBACK = 3
INTERNAL_LOOKBACK = 1
EXTERNAL_LOOKBACK = 4


def _strength(clearance: float, atr_value: float, base: float = 45.0) -> float:
    """Score a break from how far it cleared the level, in ATR units."""
    if not np.isfinite(atr_value) or atr_value <= 0:
        return float(np.clip(base, 0.0, 100.0))
    if not np.isfinite(clearance):
        return float(np.clip(base, 0.0, 100.0))
    return float(np.clip(base + 55.0 * (clearance / atr_value), 0.0, 100.0))


def _dedupe(pivots: list[dict[str, Any]], lookback: int) -> list[dict[str, Any]]:
    """Drop plateau duplicates (same kind/price only `lookback` bars apart)."""
    out: list[dict[str, Any]] = []
    last: dict[str, dict[str, Any] | None] = {"high": None, "low": None}
    for p in pivots:
        prev = last[p["kind"]]
        if (prev is not None
                and p["index"] - prev["index"] <= lookback
                and abs(p["price"] - prev["price"]) <= 1e-9):
            continue
        last[p["kind"]] = p
        out.append(p)
    return out


def _pivot_strength(df: pd.DataFrame, p: dict[str, Any], a: pd.Series) -> float:
    i = int(p["index"])
    av = float(a.iloc[i]) if i < len(a) else np.nan
    rng = float(candle_range(df).iloc[i]) if i < len(df) else np.nan
    if not np.isfinite(av) or av <= 0 or not np.isfinite(rng):
        return 55.0
    return float(np.clip(50.0 + 25.0 * (rng / av), 0.0, 100.0))


def _trend_from_pivots(highs: list[dict[str, Any]], lows: list[dict[str, Any]],
                       i: int) -> tuple[str, str]:
    """HH/HL -> bullish, LH/LL -> bearish, using swings confirmed by bar i."""
    hs = [p for p in highs if p["confirm_index"] <= i]
    ls = [p for p in lows if p["confirm_index"] <= i]
    if len(hs) < 2 or len(ls) < 2:
        return "neutral", "awaiting two confirmed highs and lows"
    ph, ch = hs[-2], hs[-1]
    pl, cl = ls[-2], ls[-1]
    hh, lh = ch["price"] > ph["price"], ch["price"] < ph["price"]
    hl, ll = cl["price"] > pl["price"], cl["price"] < pl["price"]
    if hh and hl:
        return "bullish", "higher highs and higher lows across the last swings"
    if lh and ll:
        return "bearish", "lower highs and lower lows across the last swings"

    def label(up: bool, down: bool, up_name: str, down_name: str) -> str:
        return up_name if up else down_name if down else "equal levels"

    seq = (
        label(hh, lh, "higher highs", "lower highs")
        + " / "
        + label(hl, ll, "higher lows", "lower lows")
    )
    return "neutral", f"mixed sequence: {seq} (range)"


def _break_events(df: pd.DataFrame, lookback: int) -> list[dict[str, Any]]:
    """All causal break events for a given swing size.

    ``confirm=lookback`` aligns a pivot's visibility with the bar at which its
    full fractal window exists, which keeps results identical under truncation.
    """
    highs = _dedupe(swing_points(df, lookback=lookback, confirm=lookback, kind="high"), lookback)
    lows = _dedupe(swing_points(df, lookback=lookback, confirm=lookback, kind="low"), lookback)
    if not highs and not lows:
        return []

    a = atr(df)
    disp = displacement_mask(df, atr_s=a)
    closes = df["Close"].to_numpy(float)
    n = len(df)
    events: list[dict[str, Any]] = []
    broken: set[tuple[str, int]] = set()

    for i in range(1, n):
        close = closes[i]
        if not np.isfinite(close):
            continue
        previous = closes[i - 1]
        trend, seq = _trend_from_pivots(highs, lows, i)
        if trend == "neutral":
            continue
        av = float(a.iloc[i]) if i < len(a) else np.nan
        is_disp = bool(disp.iloc[i]) if i < len(disp) else False
        body_i = float(close - previous) if np.isfinite(previous) else 0.0
        candle = "bullish" if body_i > 0 else "bearish" if body_i < 0 else "doji"

        def emit(pivot: dict[str, Any], broke_up: bool) -> None:
            key = (pivot["kind"], int(pivot["index"]))
            if key in broken:
                return
            level = float(pivot["price"])
            if (broke_up and close <= level) or (not broke_up and close >= level):
                return
            broken.add(key)
            continuation = (trend == "bullish" and broke_up) or (
                trend == "bearish" and not broke_up
            )
            events.append({
                "bar": i,
                "kind": "bos" if continuation else "choch",
                "direction": "bullish" if broke_up else "bearish",
                "pivot": pivot,
                "level": level,
                "clearance": abs(close - level),
                "atr_value": av,
                "close": float(close),
                "trend": trend,
                "seq": seq,
                "displacement": is_disp,
                "candle": candle,
            })

        h = pivots_before(highs, i, "high")
        l = pivots_before(lows, i, "low")
        if h is not None and close > h["price"]:
            emit(h, True)
        if l is not None and close < l["price"]:
            emit(l, False)

    return events


def _event_concepts(df: pd.DataFrame, events: list[dict[str, Any]], type_: str,
                    kinds: tuple[str, ...] | None = None,
                    displacement_only: bool = False,
                    max_n: int = MAX_RECENT,
                    label: str = "") -> list[Concept]:
    out: list[Concept] = []
    for e in events:
        if kinds and e["kind"] not in kinds:
            continue
        if displacement_only and not e["displacement"]:
            continue
        p = e["pivot"]
        level = e["level"]
        prefix = f"{label} " if label else ""
        name = e["kind"].upper()
        verb = "above" if e["direction"] == "bullish" else "below"
        pivot_name = "swing high" if p["kind"] == "high" else "swing low"
        notes = (
            f"{prefix}{name} {e['direction']}: close {e['close']:.2f} broke "
            f"{verb} {pivot_name} at {level:.2f} ({e['seq']}); "
            f"{e['candle']} break candle"
            + (" with displacement" if e["displacement"] else "")
        )
        av = e["atr_value"]
        clearance_atr = (e["clearance"] / av) if np.isfinite(av) and av > 0 else None
        meta = {
            "kind": e["kind"],
            "broken_level": round(float(level), 6),
            "pivot_kind": p["kind"],
            "pivot_index": int(p["index"]),
            "pivot_time": ts_to_str(p["pivot_time"]),
            "bar_index": int(e["bar"]),
            "trend": e["trend"],
            "displacement": bool(e["displacement"]),
            "clearance_atr": round(float(clearance_atr), 6) if clearance_atr is not None else None,
        }
        out.append(make_concept(
            type_,
            e["direction"],
            p["confirm_time"],
            df.index[e["bar"]],
            top=level,
            bottom=level,
            strength=_strength(e["clearance"], av),
            status="active",
            notes=notes,
            category=CATEGORY,
            meta=meta,
        ))
    return capped(out, max_n)


@register("swing_high", CATEGORY)
def detect_swing_high(df: pd.DataFrame) -> list[Concept]:
    if len(df) < 2 * SWING_LOOKBACK + 1:
        return []
    pivots = _dedupe(
        swing_points(df, lookback=SWING_LOOKBACK, confirm=SWING_LOOKBACK, kind="high"),
        SWING_LOOKBACK,
    )
    a = atr(df)
    out: list[Concept] = []
    for p in capped(pivots, MAX_SWINGS):
        meta = {
            "kind": "high",
            "pivot_index": int(p["index"]),
            "pivot_time": ts_to_str(p["pivot_time"]),
            "lookback": SWING_LOOKBACK,
        }
        out.append(make_concept(
            "swing_high",
            "bearish",
            p["confirm_time"],
            p["confirm_time"],
            top=float(p["price"]),
            bottom=float(p["price"]),
            strength=_pivot_strength(df, p, a),
            status="active",
            notes=f"Confirmed fractal swing high at {float(p['price']):.2f}",
            category=CATEGORY,
            meta=meta,
        ))
    return out


@register("swing_low", CATEGORY)
def detect_swing_low(df: pd.DataFrame) -> list[Concept]:
    if len(df) < 2 * SWING_LOOKBACK + 1:
        return []
    pivots = _dedupe(
        swing_points(df, lookback=SWING_LOOKBACK, confirm=SWING_LOOKBACK, kind="low"),
        SWING_LOOKBACK,
    )
    a = atr(df)
    out: list[Concept] = []
    for p in capped(pivots, MAX_SWINGS):
        meta = {
            "kind": "low",
            "pivot_index": int(p["index"]),
            "pivot_time": ts_to_str(p["pivot_time"]),
            "lookback": SWING_LOOKBACK,
        }
        out.append(make_concept(
            "swing_low",
            "bullish",
            p["confirm_time"],
            p["confirm_time"],
            top=float(p["price"]),
            bottom=float(p["price"]),
            strength=_pivot_strength(df, p, a),
            status="active",
            notes=f"Confirmed fractal swing low at {float(p['price']):.2f}",
            category=CATEGORY,
            meta=meta,
        ))
    return out


@register("trend", CATEGORY)
def detect_trend(df: pd.DataFrame) -> list[Concept]:
    if len(df) < 2 * TREND_LOOKBACK + 1:
        return []
    pivots = _dedupe(
        swing_points(df, lookback=TREND_LOOKBACK, confirm=TREND_LOOKBACK, kind="both"),
        TREND_LOOKBACK,
    )
    highs = [p for p in pivots if p["kind"] == "high"]
    lows = [p for p in pivots if p["kind"] == "low"]

    out: list[Concept] = []
    last_direction: str | None = None
    window: list[tuple[str, float]] = []
    for p in pivots:
        window.append((p["kind"], float(p["price"])))
        if len(window) > 6:
            window.pop(0)
        seq_highs = [v for k, v in window if k == "high"]
        seq_lows = [v for k, v in window if k == "low"]
        if len(seq_highs) < 2 or len(seq_lows) < 2:
            continue
        direction, seq = _trend_from_pivots(highs, lows, p["confirm_index"])
        if direction == last_direction:
            continue
        last_direction = direction
        out.append(make_concept(
            "trend",
            direction,
            p["confirm_time"],
            p["confirm_time"],
            top=float(seq_highs[-1]),
            bottom=float(seq_lows[-1]),
            strength=70.0 if direction != "neutral" else 40.0,
            status="active",
            notes=f"Trend {direction}: {seq}.",
            category=CATEGORY,
            meta={
                "sequence": seq,
                "highs": [round(v, 6) for v in seq_highs[-2:]],
                "lows": [round(v, 6) for v in seq_lows[-2:]],
                "confirm_bar": int(p["confirm_index"]),
            },
        ))
    return capped(out, MAX_RECENT)


@register("bos", CATEGORY)
def detect_bos(df: pd.DataFrame) -> list[Concept]:
    return _event_concepts(df, _break_events(df, BREAK_LOOKBACK), "bos", kinds=("bos",))


@register("choch", CATEGORY)
def detect_choch(df: pd.DataFrame) -> list[Concept]:
    return _event_concepts(df, _break_events(df, BREAK_LOOKBACK), "choch", kinds=("choch",))


@register("mss", CATEGORY)
def detect_mss(df: pd.DataFrame) -> list[Concept]:
    return _event_concepts(
        df, _break_events(df, BREAK_LOOKBACK), "mss",
        kinds=("choch",), displacement_only=True,
    )


@register("internal_structure", CATEGORY)
def detect_internal_structure(df: pd.DataFrame) -> list[Concept]:
    events = _break_events(df, INTERNAL_LOOKBACK)
    return _event_concepts(
        df, events, "internal_structure", max_n=MAX_RECENT, label="Internal",
    )


@register("external_structure", CATEGORY)
def detect_external_structure(df: pd.DataFrame) -> list[Concept]:
    events = _break_events(df, EXTERNAL_LOOKBACK)
    return _event_concepts(
        df, events, "external_structure", max_n=MAX_RECENT, label="External",
    )


__all__ = [
    "detect_swing_high", "detect_swing_low", "detect_trend", "detect_bos",
    "detect_choch", "detect_mss", "detect_internal_structure",
    "detect_external_structure",
]
