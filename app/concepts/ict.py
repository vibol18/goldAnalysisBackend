"""ICT / time-and-price concept detectors (strictly causal).

Every detector only reads candles at or before the last (closed) bar and stamps
its output at that bar. Geometry is computed from completed bars, so editing a
future candle can never rewrite a past detection.
"""
from __future__ import annotations

from datetime import time as dtime
from typing import Any

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from app.concepts.base import (
    Concept,
    atr,
    body,
    candle_range,
    ensure_frame,
    make_concept,
    register,
    swing_points,
    upper_wick,
    lower_wick,
)

NY_TZ = ZoneInfo("America/New_York")
SILVER_BULLET_WINDOWS = ((dtime(3, 0), dtime(4, 0)), (dtime(10, 0), dtime(11, 0)),
                         (dtime(14, 0), dtime(15, 0)))


def _prep(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    try:
        return ensure_frame(df)
    except Exception:
        return None


def _stamp(df: pd.DataFrame) -> Any:
    return df.index[-1]


def _emit(type_: str, direction: str, df: pd.DataFrame, top: float | None,
          bottom: float | None, strength: float, notes: str,
          *, meta: dict[str, Any] | None = None, start: Any = None,
          category: str = "smc") -> list[Concept]:
    if top is None or bottom is None or not np.isfinite(top) or not np.isfinite(bottom):
        return []
    return [make_concept(type_, direction, start if start is not None else _stamp(df),
                         _stamp(df), float(top), float(bottom), float(strength),
                         "active", notes, category, meta or {})]


def _rolling_extremes(df: pd.DataFrame, window: int) -> tuple[float, float]:
    n = min(window, len(df))
    return float(df["High"].iloc[-n:].max()), float(df["Low"].iloc[-n:].min())


@register("ipda_ranges", "smc")
def detect_ipda_ranges(df: pd.DataFrame, ipda_lookbacks: tuple[int, ...] = (20, 40, 60)) -> list[Concept]:
    """IPDA 20/40/60-day highs and lows as liquidity targets."""
    frame = _prep(df)
    if frame is None or len(frame) < 20:
        return []
    price = float(frame["Close"].iloc[-1])
    out: list[Concept] = []
    for lookback in ipda_lookbacks:
        high, low = _rolling_extremes(frame, lookback)
        if high <= low:
            continue
        position = (price - low) / (high - low)
        direction = "bullish" if position < 0.5 else "bearish"
        target = low if direction == "bullish" else high
        out.extend(_emit("ipda_ranges", direction, frame, high, low,
                         40 + 30 * abs(position - 0.5) * 2,
                         f"IPDA {lookback}: draw on liquidity at {target:.2f}",
                         meta={"lookback": lookback, "target": target, "position": position}))
    return out


@register("draw_on_liquidity", "smc")
def detect_draw_on_liquidity(df: pd.DataFrame) -> list[Concept]:
    """Nearest equal-high/equal-low pool inside the current dealing range."""
    frame = _prep(df)
    if frame is None or len(frame) < 30:
        return []
    pivots = swing_points(frame, lookback=2, confirm=1)
    highs = [p["price"] for p in pivots if p["kind"] == "high"][-8:]
    lows = [p["price"] for p in pivots if p["kind"] == "low"][-8:]
    price = float(frame["Close"].iloc[-1])
    a = float(atr(frame).iloc[-1]) or max(price * 0.01, 1e-6)
    best: tuple[float, str, float] | None = None
    for level in highs + lows:
        distance = abs(level - price) / a
        if best is None or distance < best[2]:
            best = (level, "sell_side" if level >= price else "buy_side", distance)
    if best is None:
        return []
    level, side, distance = best
    direction = "bullish" if side == "sell_side" else "bearish"
    return _emit("draw_on_liquidity", direction, frame, level + 0.15 * a,
                 level - 0.15 * a, max(20.0, 80.0 - distance * 10),
                 f"Draw on {side.replace('_', ' ')} liquidity at {level:.2f} "
                 f"({distance:.1f} ATR away)",
                 meta={"level": level, "pool": side, "distance_atr": distance})


@register("nwog_ndog", "smc")
def detect_nwog_ndog(df: pd.DataFrame) -> list[Concept]:
    """New week/day opening gaps, valid only while still unfilled."""
    frame = _prep(df)
    if frame is None or not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 30:
        return []
    price = float(frame["Close"].iloc[-1])
    out: list[Concept] = []
    for rule, label in (("D", "ndog"), ("W", "nwog")):
        grouped = frame.resample(rule)
        opens = grouped["Open"].first().dropna()
        closes = grouped["Close"].last().dropna()
        common = opens.index.intersection(closes.index)
        if len(common) < 2:
            continue
        open_v, prev_close = opens.loc[common], closes.loc[common].shift(1)
        gaps = pd.DataFrame({"open": open_v, "prev": prev_close}).dropna()
        if gaps.empty:
            continue
        last = gaps.iloc[-1]
        gap = float(last["open"] - last["prev"])
        if abs(gap) < 1e-9:
            continue
        high, low = max(float(last["open"]), float(last["prev"])), min(float(last["open"]), float(last["prev"]))
        filled = low <= price <= high
        if filled:
            continue
        direction = "bullish" if gap > 0 else "bearish"
        out.extend(_emit(label, direction, frame, high, low,
                         40 + min(abs(gap), 20.0), f"{label.upper()} {'up' if gap > 0 else 'down'} gap {gap:+.2f}",
                         meta={"gap": gap, "filled": False, "period": str(common[-1])}))
    return out


@register("silver_bullet", "smc")
def detect_silver_bullet(df: pd.DataFrame, windows: Any = SILVER_BULLET_WINDOWS) -> list[Concept]:
    """ICT Silver Bullet time windows, DST-safe (America/New_York)."""
    frame = _prep(df)
    if frame is None or not isinstance(frame.index, pd.DatetimeIndex):
        return []
    ts = frame.index[-1]
    try:
        local = pd.Timestamp(ts)
        if local.tzinfo is None:
            local = local.tz_localize("UTC")
        local = local.tz_convert(NY_TZ)
    except Exception:
        return []
    for start, end in windows:
        if start <= local.time() < end:
            direction = "bullish" if float(frame["Close"].iloc[-1]) > float(frame["Open"].iloc[-1]) else "bearish"
            top, bottom = float(frame["High"].iloc[-1]), float(frame["Low"].iloc[-1])
            return _emit("silver_bullet", direction, frame, top, bottom, 55,
                         f"Silver Bullet window {start:%H:%M}-{end:%H:%M} ET",
                         meta={"window": f"{start:%H:%M}-{end:%H:%M}", "timezone": "America/New_York"})
    return []


@register("asian_range", "smc")
def detect_asian_range(df: pd.DataFrame) -> list[Concept]:
    """Asian session high/low from completed bars, with break direction."""
    frame = _prep(df)
    if frame is None or not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 8:
        return []
    utc = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
    mask = (utc.hour >= 0) & (utc.hour < 7)
    if mask.sum() < 4:
        return []
    asian = frame.loc[mask]
    high, low = float(asian["High"].max()), float(asian["Low"].min())
    price = float(frame["Close"].iloc[-1])
    if high <= low:
        return []
    direction = "bullish" if price > high else "bearish" if price < low else "neutral"
    return _emit("asian_range", direction, frame, high, low, 50,
                 f"Asian range {low:.2f}-{high:.2f}" + (" broken" if direction != "neutral" else ""),
                 meta={"session": "asian", "broken": direction != "neutral"})


@register("london_reversal", "smc")
def detect_london_reversal(df: pd.DataFrame) -> list[Concept]:
    """First-hour London range rejection/reversal on a closed candle."""
    frame = _prep(df)
    if frame is None or not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 6:
        return []
    utc = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
    mask = (utc.hour >= 7) & (utc.hour < 8)
    if mask.sum() < 1:
        return []
    first = frame.loc[mask]
    high, low = float(first["High"].max()), float(first["Low"].min())
    price, close = float(frame["Close"].iloc[-1]), float(frame["Close"].iloc[-1])
    open_ = float(frame["Open"].iloc[-1])
    if high <= low or not (price < low or price > high):
        return []
    direction = "bearish" if price > high and close < open_ else "bullish" if price < low and close > open_ else "neutral"
    if direction == "neutral":
        return []
    return _emit("london_reversal", direction, frame, high, low, 55,
                 "London opening range failed break (reversal)", meta={"session": "london"})


@register("ny_reversal", "smc")
def detect_ny_reversal(df: pd.DataFrame) -> list[Concept]:
    """New York opening-range reversal (12:00-13:00 UTC)."""
    frame = _prep(df)
    if frame is None or not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 6:
        return []
    utc = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
    mask = (utc.hour >= 12) & (utc.hour < 13)
    if mask.sum() < 1:
        return []
    first = frame.loc[mask]
    high, low = float(first["High"].max()), float(first["Low"].min())
    price = float(frame["Close"].iloc[-1])
    if high <= low or not (price < low or price > high):
        return []
    direction = "bullish" if price < low else "bearish"
    return _emit("ny_reversal", direction, frame, high, low, 55,
                 "New York opening range failed break (reversal)", meta={"session": "new_york"})


@register("bpr", "smc")
def detect_bpr(df: pd.DataFrame) -> list[Concept]:
    """Balanced Price Range: overlap of a bullish and a bearish FVG."""
    frame = _prep(df)
    if frame is None or len(frame) < 8:
        return []
    o, h, l, c = (frame[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    bulls, bears = [], []
    for i in range(2, len(frame)):
        if l[i] > h[i - 2]:
            bulls.append((float(l[i]), float(h[i - 2])))
        if h[i] < l[i - 2]:
            bears.append((float(l[i]), float(h[i - 2])))
    for bottom, top in bulls[-5:]:
        for b_bottom, b_top in bears[-5:]:
            low, high = max(bottom, b_bottom), min(top, b_top)
            if high > low:
                return _emit("bpr", "neutral", frame, high, low, 60,
                             "Balanced Price Range (overlapping FVGs)",
                             meta={"type": "bpr"})
    return []


@register("volume_imbalance", "smc")
def detect_volume_imbalance(df: pd.DataFrame) -> list[Concept]:
    """Candle body that completely covers the previous candle body."""
    frame = _prep(df)
    if frame is None or len(frame) < 5:
        return []
    o, c = frame["Open"].to_numpy(float), frame["Close"].to_numpy(float)
    for i in range(len(frame) - 1, 0, -1):
        up = o[i] >= max(o[i - 1], c[i - 1]) and c[i] <= min(o[i - 1], c[i - 1])
        down = c[i] >= max(o[i - 1], c[i - 1]) and o[i] <= min(o[i - 1], c[i - 1])
        if up or down:
            top, bottom = max(o[i - 1], c[i - 1], o[i], c[i]), min(o[i - 1], c[i - 1], o[i], c[i])
            return _emit("volume_imbalance", "bearish" if up else "bullish", frame,
                         top, bottom, 50, "Volume imbalance (full body engulf)",
                         meta={"candle": str(frame.index[i])})
    return []


@register("liquidity_void", "smc")
def detect_liquidity_void(df: pd.DataFrame) -> list[Concept]:
    """Price gap where no candle traded between two ranges."""
    frame = _prep(df)
    if frame is None or len(frame) < 5:
        return []
    h, l = frame["High"].to_numpy(float), frame["Low"].to_numpy(float)
    for i in range(2, len(frame)):
        if l[i] > h[i - 1]:
            return _emit("liquidity_void", "bearish", frame, float(l[i]), float(h[i - 1]), 55,
                         "Upward liquidity void (unfilled gap)", meta={"gap": float(l[i] - h[i - 1])})
        if h[i] < l[i - 1]:
            return _emit("liquidity_void", "bullish", frame, float(l[i - 1]), float(h[i]), 55,
                         "Downward liquidity void (unfilled gap)", meta={"gap": float(l[i - 1] - h[i])})
    return []


@register("rejection_block", "smc")
def detect_rejection_block(df: pd.DataFrame) -> list[Concept]:
    """Long-wick rejection followed by displacement in the opposite direction."""
    frame = _prep(df)
    if frame is None or len(frame) < 5:
        return []
    h, l = frame["High"].to_numpy(float), frame["Low"].to_numpy(float)
    c, o = frame["Close"].to_numpy(float), frame["Open"].to_numpy(float)
    uw, lw = upper_wick(frame).to_numpy(float), lower_wick(frame).to_numpy(float)
    rng = candle_range(frame).to_numpy(float) + 1e-9
    for i in range(len(frame) - 2, 1, -1):
        if uw[i] / rng[i] > 0.5 and c[i + 1] < o[i + 1]:
            return _emit("rejection_block", "bearish", frame, h[i], l[i], 55,
                         "Rejection block: upper wick then bearish displacement", meta={"bar": str(frame.index[i])})
        if lw[i] / rng[i] > 0.5 and c[i + 1] > o[i + 1]:
            return _emit("rejection_block", "bullish", frame, h[i], l[i], 55,
                         "Rejection block: lower wick then bullish displacement", meta={"bar": str(frame.index[i])})
    return []


@register("turtle_soup", "smc")
def detect_turtle_soup(df: pd.DataFrame, period: int = 20) -> list[Concept]:
    """Failed break of a 20-bar high/low (Turtle Soup)."""
    frame = _prep(df)
    if frame is None or len(frame) < period + 2:
        return []
    high = float(frame["High"].iloc[-period - 1:-1].max())
    low = float(frame["Low"].iloc[-period - 1:-1].min())
    last_h, last_l = float(frame["High"].iloc[-1]), float(frame["Low"].iloc[-1])
    close = float(frame["Close"].iloc[-1])
    if last_h > high and close < high:
        return _emit("turtle_soup", "bearish", frame, last_h, high, 65,
                     "Turtle Soup: failed break above 20-bar high", meta={"level": high})
    if last_l < low and close > low:
        return _emit("turtle_soup", "bullish", frame, low, last_l, 65,
                     "Turtle Soup: failed break below 20-bar low", meta={"level": low})
    return []


@register("relative_equal_levels", "smc")
def detect_relative_equal_levels(df: pd.DataFrame, tolerance_atr: float = 0.15) -> list[Concept]:
    """Pivot clusters within a small ATR tolerance (relative equal H/L)."""
    frame = _prep(df)
    if frame is None or len(frame) < 30:
        return []
    a = float(atr(frame).iloc[-1]) or max(float(frame["Close"].iloc[-1]) * 0.01, 1e-6)
    pivots = swing_points(frame, lookback=2, confirm=1)
    highs = sorted(p["price"] for p in pivots if p["kind"] == "high")
    lows = sorted(p["price"] for p in pivots if p["kind"] == "low")
    for values, direction, label in ((highs, "bearish", "relative equal highs"),
                                     (lows, "bullish", "relative equal lows")):
        for x, y in zip(values, values[1:]):
            if abs(x - y) <= tolerance_atr * a:
                return _emit("relative_equal_levels", direction, frame, y + 0.1 * a,
                             x - 0.1 * a, 60, f"{label} at {y:.2f}", meta={"level": y})
    return []


@register("stop_run_quality", "smc")
def detect_stop_run_quality(df: pd.DataFrame) -> list[Concept]:
    """Quality score for a recent sweep: wick size, volume and speed of return."""
    frame = _prep(df)
    if frame is None or len(frame) < 6:
        return []
    i = len(frame) - 1
    h, l = float(frame["High"].iloc[i]), float(frame["Low"].iloc[i])
    prev_h, prev_l = float(frame["High"].iloc[i - 1]), float(frame["Low"].iloc[i - 1])
    close = float(frame["Close"].iloc[i])
    rng = candle_range(frame).to_numpy(float) + 1e-9
    vol = frame["Volume"].to_numpy(float) if "Volume" in frame else np.zeros(len(frame))
    avg_vol = float(np.mean(vol[max(0, i - 20):i])) if i > 0 else 0.0
    vol_ratio = float(vol[i] / avg_vol) if avg_vol > 0 else 0.0
    if h > prev_h and close < prev_h:
        wick = (h - max(float(frame["Open"].iloc[i]), close)) / rng[i]
        quality = min(100.0, 30 + wick * 60 + min(vol_ratio, 3) * 10)
        return _emit("stop_run_quality", "bearish", frame, h, prev_h, quality,
                     f"Stop run of prior high, quality {quality:.0f}",
                     meta={"wick_ratio": wick, "volume_ratio": vol_ratio})
    if l < prev_l and close > prev_l:
        wick = (min(float(frame["Open"].iloc[i]), close) - l) / rng[i]
        quality = min(100.0, 30 + wick * 60 + min(vol_ratio, 3) * 10)
        return _emit("stop_run_quality", "bullish", frame, prev_l, l, quality,
                     f"Stop run of prior low, quality {quality:.0f}",
                     meta={"wick_ratio": wick, "volume_ratio": vol_ratio})
    return []


@register("big_figure_levels", "classic")
def detect_big_figure_levels(df: pd.DataFrame) -> list[Concept]:
    """Institutional round numbers xx00 / xx25 / xx50 / xx75 near price."""
    frame = _prep(df)
    if frame is None or len(frame) < 3:
        return []
    price = float(frame["Close"].iloc[-1])
    a = float(atr(frame).iloc[-1]) or max(price * 0.01, 1e-6)
    step = 25.0 if price > 500 else 5.0
    base = round(price / step) * step
    out: list[Concept] = []
    for level in (base - step, base, base + step):
        if abs(level - price) > 2 * a:
            continue
        direction = "bullish" if level < price else "bearish"
        out.extend(_emit("big_figure_levels", direction, frame, level + 0.05 * a,
                         level - 0.05 * a, 40, f"Big figure {level:.0f}",
                         meta={"level": level}, category="classic"))
    return out


@register("quarterly_theory", "mtf")
def detect_quarterly_theory(df: pd.DataFrame) -> list[Concept]:
    """Low-weight Q1-Q4 time context (ICT quarterly theory)."""
    frame = _prep(df)
    if frame is None or not isinstance(frame.index, pd.DatetimeIndex):
        return []
    ts = pd.Timestamp(frame.index[-1])
    quarter = (ts.month - 1) // 3 + 1
    return _emit("quarterly_theory", "neutral", frame, float(frame["High"].iloc[-1]),
                 float(frame["Low"].iloc[-1]), 15,
                 f"Quarter {quarter} time context", meta={"quarter": quarter},
                 category="mtf")


@register("ict_narrative", "mtf")
def detect_ict_narrative(df: pd.DataFrame) -> list[Concept]:
    """ICT narrative state machine: bias -> target -> sweep -> displacement -> retest."""
    frame = _prep(df)
    if frame is None or len(frame) < 40:
        return []
    price = float(frame["Close"].iloc[-1])
    high, low = _rolling_extremes(frame, 20)
    swept_high = float(frame["High"].iloc[-1]) >= high
    swept_low = float(frame["Low"].iloc[-1]) <= low
    body_now = abs(float(frame["Close"].iloc[-1]) - float(frame["Open"].iloc[-1]))
    a = float(atr(frame).iloc[-1]) or max(price * 0.01, 1e-6)
    displaced = body_now > a
    recent_break = bool(swept_high or swept_low)
    if not recent_break:
        step, waiting = "liquidity_target", f"waiting for sweep of {high:.2f} / {low:.2f}"
    elif not displaced:
        step, waiting = "displacement", "waiting for displacement/MSS after the sweep"
    else:
        step, waiting = "pd_array_retest", "waiting for the retest of the PD array"
    direction = "bearish" if swept_high else "bullish" if swept_low else "neutral"
    return _emit("ict_narrative", direction, frame, high, low, 50,
                 f"ICT narrative: {step}. {waiting}.",
                 meta={"step": step, "waiting_for": waiting}, category="mtf")
