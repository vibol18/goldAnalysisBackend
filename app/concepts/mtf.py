"""Multi-timeframe concept detectors.

Every detector trims any extra timeframe frame to the current bar's timestamp
before use, so a future candle on another timeframe can never leak into a
concept that is visible now.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.concepts.base import (
    atr,
    displacement_mask,
    make_concept,
    register,
    signed_body,
    swing_points,
)

TF_WEIGHTS = {"1d": 4.0, "4h": 2.5, "1h": 1.0, "15m": 0.5}


def _normalise(frame: Any) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    out = frame.copy()
    if "Close" not in out.columns:
        lower = {str(c).lower(): c for c in out.columns}
        close_col = lower.get("close")
        if close_col is None:
            return None
        out = out.rename(columns={close_col: "Close"})
    for col in ("Open", "High", "Low"):
        if col not in out.columns:
            out[col] = out["Close"]
    for col in ("Open", "High", "Low", "Close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[out["Close"].notna()]
    if out.empty:
        return None
    try:
        out = out.sort_index()
    except TypeError:
        pass
    return out


def _truncate(frame: Any, ts: Any) -> pd.DataFrame | None:
    out = _normalise(frame)
    if out is None:
        return None
    if isinstance(out.index, pd.DatetimeIndex) and ts is not None:
        try:
            target = pd.Timestamp(ts)
            if pd.isna(target):
                return None
            out = out[out.index <= target]
        except (TypeError, ValueError):
            pass
    if out.empty:
        return None
    return out


def _last_ts(frame: pd.DataFrame | None) -> Any:
    if frame is not None and isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
        return frame.index[-1]
    return None


def _bar_time(frame: pd.DataFrame) -> Any:
    return frame.index[-1] if len(frame.index) else None


def _atr_value(frame: pd.DataFrame) -> float:
    try:
        value = float(atr(frame).iloc[-1])
    except (ValueError, TypeError, IndexError):
        value = float("nan")
    if np.isfinite(value) and value > 0:
        return value
    return max(abs(float(frame["Close"].iloc[-1])) * 0.001, 1e-9)


def _structure_dir(frame: pd.DataFrame) -> str:
    pivots = swing_points(frame, lookback=2, confirm=1)
    lows = [p["price"] for p in pivots if p["kind"] == "low"][-2:]
    highs = [p["price"] for p in pivots if p["kind"] == "high"][-2:]
    if len(lows) == 2 and lows[-1] > lows[-2]:
        return "bullish"
    if len(highs) == 2 and highs[-1] < highs[-2]:
        return "bearish"
    return "neutral"


def _trend(frame: pd.DataFrame) -> tuple[str, float, float | None, float | None]:
    close = frame["Close"]
    if len(close) < 5:
        return "neutral", 0.0, None, None
    fast_span = min(20, max(3, len(close) // 2))
    slow_span = min(50, max(5, len(close)))
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    c, f, s = float(close.iloc[-1]), float(fast.iloc[-1]), float(slow.iloc[-1])
    if not all(np.isfinite(v) for v in (c, f, s)):
        return "neutral", 0.0, None, None
    if c > f > s:
        direction = "bullish"
    elif c < f < s:
        direction = "bearish"
    else:
        direction = "neutral"
    separation = abs(f - s) / _atr_value(frame)
    strength = float(np.clip(35 + separation * 18, 0, 100)) if direction != "neutral" else 8.0
    structure = _structure_dir(frame)
    if direction != "neutral" and structure == direction:
        strength = float(np.clip(strength + 10, 0, 100))
    elif direction != "neutral" and structure not in ("neutral", direction):
        strength *= 0.7
    return direction, strength, f, s


def _location(frame: pd.DataFrame) -> tuple[str, float]:
    close = frame["Close"]
    window = min(50, len(close))
    high = float(frame["High"].iloc[-window:].max())
    low = float(frame["Low"].iloc[-window:].min())
    current = float(close.iloc[-1])
    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        return "neutral", 0.5
    position = float(np.clip((current - low) / (high - low), 0.0, 1.0))
    if position >= 0.62:
        return "premium", position
    if position <= 0.38:
        return "discount", position
    return "equilibrium", position


def _nearest_level(frame: pd.DataFrame, price: float) -> dict[str, Any] | None:
    pivots = swing_points(frame, lookback=2, confirm=1)
    supports = [p for p in pivots if p["kind"] == "low" and p["price"] < price]
    resistances = [p for p in pivots if p["kind"] == "high" and p["price"] > price]
    support = min(supports, key=lambda p: abs(p["price"] - price)) if supports else None
    resistance = min(resistances, key=lambda p: abs(p["price"] - price)) if resistances else None
    if support is not None or resistance is not None:
        chosen = min(
            [p for p in (support, resistance) if p is not None],
            key=lambda p: abs(p["price"] - price),
        )
        return {
            "kind": "support" if chosen is support else "resistance",
            "price": float(chosen["price"]),
            "time": chosen.get("confirm_time"),
        }
    window = min(20, len(frame))
    high = float(frame["High"].iloc[-window:].max())
    low = float(frame["Low"].iloc[-window:].min())
    if price >= high:
        return {"kind": "resistance", "price": high, "time": _bar_time(frame)}
    if price <= low:
        return {"kind": "support", "price": low, "time": _bar_time(frame)}
    resistance_nearer = (high - price) <= (price - low)
    return {
        "kind": "resistance" if resistance_nearer else "support",
        "price": high if resistance_nearer else low,
        "time": _bar_time(frame),
    }


def _recent_displacement(frame: pd.DataFrame, direction: str, window: int = 3) -> bool:
    if len(frame) < 3:
        return False
    a = atr(frame)
    mask = displacement_mask(frame, atr_s=a)
    signed = signed_body(frame)
    tail = slice(-window, None)
    values = mask.iloc[tail].fillna(False).astype(bool)
    bodies = signed.iloc[tail]
    if direction == "bullish":
        return bool((values & (bodies > 0)).any())
    return bool((values & (bodies < 0)).any())


@register("mtf_bias", "mtf")
def detect_mtf_bias(df: pd.DataFrame, htf_1d: pd.DataFrame | None = None,
                    htf_4h: pd.DataFrame | None = None,
                    ltf_15m: pd.DataFrame | None = None) -> list[Any]:
    """Top-down gold bias from the 1D/4h context and the current frame."""
    frame = _normalise(df)
    if frame is None:
        return []
    ts = _last_ts(frame)
    per_tf: dict[str, dict[str, Any]] = {}
    bias_votes: dict[str, str] = {}
    score = 0.0
    strengths: list[float] = []

    base_dir, base_strength, _, _ = _trend(frame)
    base_loc, base_pos = _location(frame)
    per_tf["1h"] = {
        "direction": base_dir, "strength": round(base_strength, 2),
        "location": base_loc, "position": round(base_pos, 4),
    }
    bias_votes["1h"] = base_dir
    score += (1 if base_dir == "bullish" else -1 if base_dir == "bearish" else 0) * TF_WEIGHTS["1h"]
    strengths.append(base_strength)

    for label, raw in (("1d", htf_1d), ("4h", htf_4h), ("15m", ltf_15m)):
        other = _truncate(raw, ts)
        if other is None:
            continue
        direction, strength, _, _ = _trend(other)
        location, position = _location(other)
        per_tf[label] = {
            "direction": direction, "strength": round(strength, 2),
            "location": location, "position": round(position, 4),
        }
        bias_votes[label] = direction
        score += (1 if direction == "bullish" else -1 if direction == "bearish" else 0) * TF_WEIGHTS[label]
        strengths.append(strength)

    if not strengths:
        return []
    direction = "bullish" if score > 0.25 else "bearish" if score < -0.25 else "neutral"
    strength = float(np.clip(float(np.mean(strengths)) + min(abs(score), 3) * 8, 0, 100))
    levels: dict[str, dict[str, Any]] = {}
    for label, raw in (("1d", htf_1d), ("4h", htf_4h)):
        other = _truncate(raw, ts)
        if other is None:
            continue
        level = _nearest_level(other, float(frame["Close"].iloc[-1]))
        if level is not None:
            levels[label] = level

    notes = (
        f"Top-down bias: {direction}; "
        + ", ".join(f"{k}={v['direction']}" for k, v in per_tf.items())
        + f". Price is at {base_loc} on the current frame."
    )
    meta = {
        "timeframes": per_tf,
        "biases": bias_votes,
        "bias": direction,
        "htf_bias": per_tf.get("1d", per_tf.get("4h", {})).get("direction", "neutral"),
        "levels": levels,
        "score": round(score, 3),
    }
    return [make_concept(
        "mtf_bias", direction, _bar_time(frame), _bar_time(frame),
        float(frame["High"].iloc[-1]), float(frame["Low"].iloc[-1]),
        strength, "active", notes, category="mtf", meta=meta,
    )]


@register("htf_veto", "mtf")
def detect_htf_veto(df: pd.DataFrame, htf_1d: pd.DataFrame | None = None,
                    htf_4h: pd.DataFrame | None = None,
                    ltf_15m: pd.DataFrame | None = None) -> list[Any]:
    """Flag a lower-timeframe setup that conflicts with the HTF direction."""
    frame = _normalise(df)
    if frame is None:
        return []
    ts = _last_ts(frame)
    htf_dir, htf_strength, htf_label = "neutral", 0.0, "HTF"
    for label, raw in (("1d", htf_1d), ("4h", htf_4h)):
        other = _truncate(raw, ts)
        if other is None:
            continue
        direction, strength, _, _ = _trend(other)
        if direction != "neutral" and strength > htf_strength:
            htf_dir, htf_strength, htf_label = direction, strength, label
    if htf_dir == "neutral":
        return []

    lower: list[tuple[str, str, float]] = []
    base_dir, base_strength, _, _ = _trend(frame)
    lower.append(("1h", base_dir, base_strength))
    ltf = _truncate(ltf_15m, ts)
    if ltf is not None:
        direction, strength, _, _ = _trend(ltf)
        lower.append(("15m", direction, strength))
    opposing = [
        (label, direction, strength)
        for label, direction, strength in lower
        if direction not in (htf_dir, "neutral") and strength >= 20
    ]
    if not opposing:
        return []
    label, setup_dir, setup_strength = max(opposing, key=lambda item: item[2])
    confirmed = _recent_displacement(frame, setup_dir) or (
        ltf is not None and _recent_displacement(ltf, setup_dir)
    )
    strength = float(np.clip(
        55 + 0.25 * htf_strength + 0.15 * setup_strength + (8 if confirmed else 0),
        40, 95,
    ))
    notes = (
        f"{htf_label} {htf_dir} bias vetoes the {setup_dir} {label} setup"
        + (" (LTF displacement confirms setup)." if confirmed else ".")
    )
    meta = {
        "htf_timeframe": htf_label,
        "htf_direction": htf_dir,
        "htf_strength": round(htf_strength, 2),
        "setup_timeframe": label,
        "vetoed_direction": setup_dir,
        "setup_strength": round(setup_strength, 2),
        "displacement_confirmed": confirmed,
    }
    return [make_concept(
        "htf_veto", htf_dir, _bar_time(frame), _bar_time(frame),
        float(frame["High"].iloc[-1]), float(frame["Low"].iloc[-1]),
        strength, "active", notes, category="mtf", meta=meta,
    )]


@register("htf_level", "mtf")
def detect_htf_level(df: pd.DataFrame, htf_1d: pd.DataFrame | None = None,
                     htf_4h: pd.DataFrame | None = None,
                     ltf_15m: pd.DataFrame | None = None) -> list[Any]:
    """Nearest HTF support/resistance projected onto the current timeframe."""
    frame = _normalise(df)
    if frame is None:
        return []
    ts = _last_ts(frame)
    price = float(frame["Close"].iloc[-1])
    a = _atr_value(frame)
    candidates: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []
    for label, raw in (("1d", htf_1d), ("4h", htf_4h)):
        other = _truncate(raw, ts)
        if other is None:
            continue
        level = _nearest_level(other, price)
        if level is not None:
            candidates.append((label, other, level))
    if not candidates:
        return []
    label, other, level = min(candidates, key=lambda item: abs(item[2]["price"] - price))
    kind = level["kind"]
    direction = "bullish" if kind == "support" else "bearish"
    distance_atr = abs(level["price"] - price) / a
    strength = float(np.clip(85 - distance_atr * 18, 20, 95))
    half_width = max(a * 0.35, abs(level["price"]) * 0.0005)
    notes = (
        f"Nearest {label} {kind} at {level['price']:.2f}, "
        f"{distance_atr:.2f} ATR from price; projected as a reaction zone."
    )
    meta = {
        "timeframe": label,
        "kind": kind,
        "level": round(level["price"], 4),
        "distance_atr": round(distance_atr, 4),
    }
    return [make_concept(
        "htf_level", direction, level.get("time") or _bar_time(other), _bar_time(frame),
        level["price"] + half_width, level["price"] - half_width,
        strength, "active", notes, category="mtf", meta=meta,
    )]


def _sweep_confirmation(frame: pd.DataFrame, direction: str, look: int) -> tuple[int, str, float] | None:
    n = len(frame)
    for i in range(n - 1, max(-1, n - 1 - look), -1):
        prior = frame.iloc[max(0, i - 6):i]
        if prior.empty:
            continue
        if direction == "bullish":
            swept = float(frame["Low"].iloc[i]) < float(prior["Low"].min())
            reversed_bar = float(frame["Close"].iloc[i]) > float(frame["Open"].iloc[i])
        else:
            swept = float(frame["High"].iloc[i]) > float(prior["High"].max())
            reversed_bar = float(frame["Close"].iloc[i]) < float(frame["Open"].iloc[i])
        if swept and reversed_bar:
            magnitude = float(frame["Close"].iloc[i]) - float(frame["Open"].iloc[i])
            return i, "sweep", abs(magnitude) / _atr_value(frame)
    return None


@register("ltf_entry_confirmation", "mtf")
def detect_ltf_entry_confirmation(df: pd.DataFrame, ltf_15m: pd.DataFrame | None = None,
                                  htf_1d: pd.DataFrame | None = None,
                                  htf_4h: pd.DataFrame | None = None) -> list[Any]:
    """15m displacement or liquidity sweep confirming a 1h setup direction."""
    frame = _normalise(df)
    ltf = _truncate(ltf_15m, _last_ts(frame))
    if frame is None or ltf is None or len(ltf) < 5:
        return []
    setup_dir, setup_strength, _, _ = _trend(frame)
    if setup_dir == "neutral":
        return []

    a = atr(ltf)
    mask = displacement_mask(ltf, atr_s=a)
    signed = signed_body(ltf)
    look = min(8, len(ltf))
    found: tuple[int, str, float] | None = None
    for i in range(len(ltf) - 1, max(-1, len(ltf) - 1 - look), -1):
        if not bool(mask.iloc[i]):
            continue
        body = float(signed.iloc[i])
        if setup_dir == "bullish" and body > 0:
            found = (i, "displacement", body / max(float(a.iloc[i]) if np.isfinite(a.iloc[i]) else 0.0, 1e-9))
            break
        if setup_dir == "bearish" and body < 0:
            found = (i, "displacement", abs(body) / max(float(a.iloc[i]) if np.isfinite(a.iloc[i]) else 0.0, 1e-9))
            break
    if found is None:
        found = _sweep_confirmation(ltf, setup_dir, look)
    if found is None:
        return []

    index, kind, magnitude = found
    strength = float(np.clip(45 + min(magnitude, 5) * 8 + setup_strength * 0.15, 30, 95))
    notes = (
        f"15m {kind} confirms the {setup_dir} 1h setup "
        f"({magnitude:.2f} ATR body move)."
    )
    meta = {
        "setup_direction": setup_dir,
        "setup_timeframe": "1h",
        "confirm_type": kind,
        "confirm_timeframe": "15m",
        "atr_multiple": round(magnitude, 4),
    }
    return [make_concept(
        "ltf_entry_confirmation", setup_dir, ltf.index[index], _bar_time(frame),
        float(ltf["High"].iloc[index]), float(ltf["Low"].iloc[index]),
        strength, "active", notes, category="mtf", meta=meta,
    )]
