"""Advanced causal concepts: Wyckoff, Elliott, volume profile, order flow,
harmonics, zone quality, breakout quality and session statistics.

Every detector is pure and causal: a concept stamped at bar i is derived only
from bars <= i. Elliott and harmonic outputs are deliberately low strength and
labelled as *possible counts* - they are never certainties.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from app.concepts.base import (
    Concept,
    atr as atr_series,
    capped,
    make_concept,
    register,
    session_of,
    swing_points,
)


ADVANCED_TYPES = (
    "wyckoff_phase", "wyckoff_spring", "wyckoff_utad", "wyckoff_sos",
    "wyckoff_breakdown", "elliott_count", "market_profile", "naked_poc",
    "value_acceptance", "initial_balance", "order_flow_delta", "volume_spike",
    "absorption", "exhaustion", "harmonic_pattern", "zone_quality",
    "false_breakout", "breakout_retest", "volatility_squeeze", "nr7",
    "inside_bar_cluster", "session_statistics",
)


def _arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "o": df["Open"].to_numpy(float),
        "h": df["High"].to_numpy(float),
        "l": df["Low"].to_numpy(float),
        "c": df["Close"].to_numpy(float),
        "v": df["Volume"].to_numpy(float) if "Volume" in df else np.zeros(len(df)),
    }


def _volume_mean(vol: np.ndarray, i: int, window: int = 20) -> float:
    if i <= 0:
        return float(vol[0]) if len(vol) else 0.0
    return float(np.mean(vol[max(0, i - window):i]))


def _recent(concepts: list[Concept], type_: str, i: int, index: pd.Index, bars: int = 10) -> bool:
    """True when a concept of this type was stamped within the last `bars` bars."""
    for c in reversed(concepts):
        if c.type != type_:
            continue
        try:
            at = index.get_loc(pd.Timestamp(c.end_time))
        except (KeyError, TypeError, ValueError):
            continue
        return i - int(at) < bars
    return False


# ---------------------------------------------------------------------------
# 1. Wyckoff accumulation / distribution
# ---------------------------------------------------------------------------

@register("wyckoff_phase", "advanced")
def detect_wyckoff_phase(df: pd.DataFrame, window: int = 40) -> list[Concept]:
    """PS/SC/AR/ST/Spring/SOS/UTAD/LPS events with volume + volatility confirmation."""
    a = _arrays(df)
    n = len(df)
    if n < window + 10:
        return []
    atr = atr_series(df, 14).to_numpy(float)
    out: list[Concept] = []
    c, h, l, v = a["c"], a["h"], a["l"], a["v"]
    for i in range(window, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        lo, hi = l[i - window:i].min(), h[i - window:i].max()
        width = hi - lo
        if width <= 0 or width / atr[i] > 8:
            continue
        vol_ref = _volume_mean(v, i, window)
        if vol_ref <= 0:
            continue
        rng = h[i] - l[i]
        body = abs(c[i] - a["o"][i])

        # Selling climax / buying climax: volume spike + wide range near extreme.
        climactic = v[i] > 1.8 * vol_ref and rng > 1.5 * atr[i]
        near_low = l[i] <= lo + 0.15 * width
        near_high = h[i] >= hi - 0.15 * width
        if climactic and near_low and not _recent(out, "wyckoff_phase", i, df.index, 12):
            strength = min(85, 45 + 25 * min(2.0, v[i] / vol_ref))
            out.append(make_concept(
                "wyckoff_phase", "bullish", df.index[i], df.index[i], hi, lo,
                strength, "active",
                "Wyckoff selling climax (SC) after preliminary support; possible accumulation.",
                category="advanced",
                meta={"event": "SC", "phase": "accumulation", "range_high": float(hi),
                      "range_low": float(lo), "volume_ratio": round(float(v[i] / vol_ref), 2)},
            ))
        elif climactic and near_high and not _recent(out, "wyckoff_phase", i, df.index, 12):
            strength = min(85, 45 + 25 * min(2.0, v[i] / vol_ref))
            out.append(make_concept(
                "wyckoff_phase", "bearish", df.index[i], df.index[i], hi, lo,
                strength, "active",
                "Wyckoff buying climax (BC) at range highs; possible distribution.",
                category="advanced",
                meta={"event": "BC", "phase": "distribution", "range_high": float(hi),
                      "range_low": float(lo), "volume_ratio": round(float(v[i] / vol_ref), 2)},
            ))

        # Spring / upthrust: wick beyond the range, close back inside.
        if l[i] < lo and c[i] > lo and rng > 0.6 * atr[i] and v[i] > 0.9 * vol_ref:
            if not _recent(out, "wyckoff_spring", i, df.index, 12):
                out.append(make_concept(
                    "wyckoff_spring", "bullish", df.index[i], df.index[i], hi, lo,
                    min(80, 50 + 20 * min(2.0, v[i] / vol_ref)), "active",
                    "Wyckoff spring: low pierces range support then closes back inside (ST/Spring).",
                    category="advanced",
                    meta={"event": "Spring", "phase": "accumulation", "range_low": float(lo)},
                ))
        if h[i] > hi and c[i] < hi and rng > 0.6 * atr[i] and v[i] > 0.9 * vol_ref:
            if not _recent(out, "wyckoff_utad", i, df.index, 12):
                out.append(make_concept(
                    "wyckoff_utad", "bearish", df.index[i], df.index[i], hi, lo,
                    min(80, 50 + 20 * min(2.0, v[i] / vol_ref)), "active",
                    "Wyckoff UTAD: upthrust above range resistance rejected back inside.",
                    category="advanced",
                    meta={"event": "UTAD", "phase": "distribution", "range_high": float(hi)},
                ))

        # SOS / LPS: strong close beyond range on expanding volume.
        if c[i] > hi and body > 0.6 * atr[i] and v[i] > 1.3 * vol_ref:
            if not _recent(out, "wyckoff_sos", i, df.index, 12):
                out.append(make_concept(
                    "wyckoff_sos", "bullish", df.index[i], df.index[i], c[i], lo,
                    min(88, 55 + 20 * min(2.0, v[i] / vol_ref)), "active",
                    "Wyckoff sign of strength: decisive close above the trading range (SOS/LPS).",
                    category="advanced",
                    meta={"event": "SOS", "phase": "accumulation", "break_level": float(hi)},
                ))
        if c[i] < lo and body > 0.6 * atr[i] and v[i] > 1.3 * vol_ref:
            if not _recent(out, "wyckoff_utad", i, df.index, 12):
                out.append(make_concept(
                    "wyckoff_breakdown", "bearish", df.index[i], df.index[i], hi, c[i],
                    min(88, 55 + 20 * min(2.0, v[i] / vol_ref)), "active",
                    "Wyckoff breakdown below support on expanding volume (distribution mark-down).",
                    category="advanced",
                    meta={"event": "Markdown", "phase": "distribution", "break_level": float(lo)},
                ))
    return capped(out, 40)


# ---------------------------------------------------------------------------
# 2. Elliott wave (rule-based, low weight)
# ---------------------------------------------------------------------------

def _valid_impulse(pivots: list[dict[str, Any]], up: bool) -> bool:
    if len(pivots) < 6:
        return False
    p = [x["price"] for x in pivots[-6:]]
    kinds = [x["kind"] for x in pivots[-6:]]
    expected = ["low", "high", "low", "high", "low", "high"] if up else \
               ["high", "low", "high", "low", "high", "low"]
    if kinds != expected:
        return False
    w1 = abs(p[1] - p[0])
    w2 = abs(p[2] - p[1])
    w3 = abs(p[3] - p[2])
    w4 = abs(p[4] - p[3])
    if w1 <= 0 or w3 <= 0:
        return False
    if w2 >= w1:                      # wave 2 never retraces all of wave 1
        return False
    if w3 <= min(w1, w4):             # wave 3 is never the shortest
        return False
    if up and p[4] <= p[1]:           # wave 4 does not overlap wave 1
        return False
    if not up and p[4] >= p[1]:
        return False
    return True


def _valid_abc(pivots: list[dict[str, Any]], up: bool) -> bool:
    if len(pivots) < 4:
        return False
    p = [x["price"] for x in pivots[-4:]]
    kinds = [x["kind"] for x in pivots[-4:]]
    expected = ["high", "low", "high", "low"] if up else ["low", "high", "low", "high"]
    if kinds != expected:
        return False
    a, b = abs(p[1] - p[0]), abs(p[2] - p[1])
    return a > 0 and 0 < b < a


@register("elliott_count", "advanced")
def detect_elliott_count(df: pd.DataFrame) -> list[Concept]:
    """Possible impulse 1-5 / ABC counts. Low strength, never a certainty."""
    if len(df) < 60:
        return []
    pivots = swing_points(df, lookback=2, confirm=1)
    out: list[Concept] = []
    for i in range(6, len(pivots) + 1):
        window = pivots[:i]
        if _valid_impulse(window, up=True):
            direction, label = "bullish", "impulse 1-5 (up)"
        elif _valid_impulse(window, up=False):
            direction, label = "bearish", "impulse 1-5 (down)"
        elif _valid_abc(window, up=True):
            direction, label = "bearish", "ABC correction (down)"
        elif _valid_abc(window, up=False):
            direction, label = "bullish", "ABC correction (up)"
        else:
            continue
        last = window[-1]
        if last["confirm_index"] > len(df) - 1:
            continue
        out.append(make_concept(
            "elliott_count", direction, df.index[last["confirm_index"]],
            df.index[last["confirm_index"]],
            float(max(x["price"] for x in window[-6:])),
            float(min(x["price"] for x in window[-6:])),
            30, "active",
            f"Possible Elliott {label} - rule-checked count only, NOT a certainty.",
            category="advanced",
            meta={"count": label, "confidence": "low", "pivots": len(window[-6:])},
        ))
    return capped(out, 12)


# ---------------------------------------------------------------------------
# 3. Market Profile / Volume Profile
# ---------------------------------------------------------------------------

def _volume_profile(df: pd.DataFrame, end: int, window: int, bins: int = 32) -> dict[str, float]:
    start = max(0, end - window)
    typical = ((df["High"].to_numpy(float) + df["Low"].to_numpy(float) +
                df["Close"].to_numpy(float)) / 3.0)[start:end]
    volume = df["Volume"].to_numpy(float)[start:end] if "Volume" in df else np.zeros(len(df))[start:end]
    finite = np.isfinite(typical) & np.isfinite(volume)
    typical, volume = typical[finite], volume[finite]
    if len(typical) < 5:
        return {}
    lo, hi = float(typical.min()), float(typical.max())
    if hi <= lo:
        return {}
    hist, edges = np.histogram(typical, bins=bins, range=(lo, hi), weights=volume)
    if hist.sum() <= 0:
        return {}
    poc_idx = int(np.argmax(hist))
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
    total = float(hist.sum())
    target, lo_i, hi_i = total * 0.70, poc_idx, poc_idx
    while hist[lo_i] + hist[hi_i] < target and (lo_i > 0 or hi_i < bins - 1):
        left = hist[lo_i - 1] if lo_i > 0 else -1
        right = hist[hi_i + 1] if hi_i < bins - 1 else -1
        if right >= left:
            hi_i += 1
        else:
            lo_i -= 1
    hvn = int(np.argmax(np.convolve(hist, np.ones(3), mode="same")))
    lvn = int(np.argmin(np.where(hist > 0, hist, np.inf)))
    return {
        "poc": poc,
        "vah": float(edges[min(bins - 1, hi_i + 1)]),
        "val": float(edges[max(0, lo_i)]),
        "hvn": float((edges[hvn] + edges[hvn + 1]) / 2),
        "lvn": float((edges[lvn] + edges[lvn + 1]) / 2),
        "total_volume": total,
    }


@register("market_profile", "advanced")
def detect_market_profile(df: pd.DataFrame, window: int = 100) -> list[Concept]:
    """Rolling POC, value area high/low and high/low volume nodes."""
    if len(df) < 30:
        return []
    out: list[Concept] = []
    for i in range(min(window, len(df) - 1), len(df), max(5, window // 10)):
        profile = _volume_profile(df, i + 1, min(window, i + 1))
        if not profile:
            continue
        price = float(df["Close"].iloc[i])
        direction = "bullish" if price > profile["vah"] else "bearish" if price < profile["val"] else "neutral"
        out.append(make_concept(
            "market_profile", direction, df.index[i], df.index[i],
            profile["vah"], profile["val"], 45, "active",
            f"Volume profile: POC {profile['poc']:.2f}, VA {profile['val']:.2f}-{profile['vah']:.2f}.",
            category="advanced",
            meta={**profile, "position": "above_value" if price > profile["vah"] else
                  "below_value" if price < profile["val"] else "in_value"},
        ))
    return capped(out, 30)


@register("naked_poc", "advanced")
def detect_naked_poc(df: pd.DataFrame, window: int = 60) -> list[Concept]:
    """Prior POC that has not yet been revisited (unclosed volume node)."""
    if len(df) < window + 20:
        return []
    prior = _volume_profile(df, len(df) - window, window)
    if not prior:
        return []
    poc = prior["poc"]
    since = df.iloc[-window:]
    if ((since["Low"] <= poc) & (poc <= since["High"])).any():
        return []
    direction = "bearish" if float(df["Close"].iloc[-1]) > poc else "bullish"
    return [make_concept(
        "naked_poc", direction, df.index[-window], df.index[-1],
        poc, poc, 38, "active",
        f"Naked POC at {poc:.2f} has not been revisited - possible magnet.",
        category="advanced", meta={"poc": poc, "bars_untested": window},
    )]


@register("value_acceptance", "advanced")
def detect_value_acceptance(df: pd.DataFrame, window: int = 100) -> list[Concept]:
    """Acceptance (2 closes beyond value) vs rejection (wick + close back inside)."""
    if len(df) < window + 3:
        return []
    profile = _volume_profile(df, len(df) - 1, window)
    if not profile:
        return []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["Close"])
    direction = "neutral"
    event = "in_value"
    if close > profile["vah"] and float(prev["Close"]) > profile["vah"]:
        direction, event = "bullish", "acceptance_above"
    elif close < profile["val"] and float(prev["Close"]) < profile["val"]:
        direction, event = "bearish", "acceptance_below"
    elif float(last["High"]) > profile["vah"] and close < profile["vah"]:
        direction, event = "bearish", "rejection_above"
    elif float(last["Low"]) < profile["val"] and close > profile["val"]:
        direction, event = "bullish", "rejection_below"
    if event == "in_value":
        return []
    return [make_concept(
        "value_acceptance", direction, df.index[-1], df.index[-1],
        profile["vah"], profile["val"], 42 if "acceptance" in event else 36, "active",
        f"Value area {event.replace('_', ' ')} at {profile['val']:.2f}-{profile['vah']:.2f}.",
        category="advanced", meta={"event": event, **profile},
    )]


@register("initial_balance", "advanced")
def detect_initial_balance(df: pd.DataFrame, ib_bars: int = 4) -> list[Concept]:
    """First-hour (4 x 15m) balance range per session; N/A on daily frames."""
    if len(df) < ib_bars + 5:
        return []
    ts = pd.to_datetime(df.index, utc=True)
    minutes = pd.Series(ts.hour * 60 + ts.minute, index=df.index)
    if (minutes.diff().dropna() < 30).mean() < 0.5:  # daily or slow frame
        return []
    out: list[Concept] = []
    for date, group in df.groupby(ts.date):
        if len(group) < ib_bars + 4:
            continue
        ib = group.iloc[:ib_bars]
        high, low = float(ib["High"].max()), float(ib["Low"].min())
        rest = group.iloc[ib_bars:]
        if len(rest) == 0 or high <= low:
            continue
        direction = "bullish" if float(rest["Close"].iloc[-1]) > high else \
                    "bearish" if float(rest["Close"].iloc[-1]) < low else "neutral"
        out.append(make_concept(
            "initial_balance", direction, group.index[0], group.index[-1], high, low,
            40, "active", f"Initial balance {low:.2f}-{high:.2f} ({session_of(group.index[0])}).",
            category="advanced", meta={"high": high, "low": low, "bars": ib_bars},
        ))
    return capped(out, 20)


# ---------------------------------------------------------------------------
# 4. Order flow proxies from OHLCV
# ---------------------------------------------------------------------------

def _delta(a: dict[str, np.ndarray], i: int) -> float:
    rng = a["h"][i] - a["l"][i]
    if rng <= 0:
        return 0.0
    vol = a["v"][i]
    if a["c"][i] >= a["o"][i]:
        return float(vol * (a["c"][i] - a["l"][i]) / rng)
    return -float(vol * (a["h"][i] - a["c"][i]) / rng)


@register("order_flow_delta", "advanced")
def detect_order_flow_delta(df: pd.DataFrame, window: int = 20) -> list[Concept]:
    """OHLCV-estimated bar delta and cumulative delta (proxy, not true tape)."""
    if len(df) < window + 5:
        return []
    a = _arrays(df)
    deltas = np.array([_delta(a, i) for i in range(len(df))])
    cum = pd.Series(deltas).rolling(window, min_periods=5).sum().to_numpy()
    out: list[Concept] = []
    for i in range(window, len(df), max(3, window // 4)):
        if not np.isfinite(cum[i]) or cum[i] == 0:
            continue
        ref = np.nanmean(np.abs(cum[max(0, i - 60):i])) or 1.0
        direction = "bullish" if cum[i] > 0 else "bearish"
        strength = min(80, 40 + 25 * min(2.0, abs(cum[i]) / ref))
        out.append(make_concept(
            "order_flow_delta", direction, df.index[i], df.index[i],
            float(a["h"][i]), float(a["l"][i]), strength, "active",
            f"Cumulative delta proxy {cum[i]:.0f} over {window} bars.",
            category="advanced",
            meta={"delta": float(deltas[i]), "cumulative": float(cum[i]),
                  "note": "OHLCV proxy, not exchange order flow"},
        ))
    return capped(out, 20)


@register("volume_spike", "advanced")
def detect_volume_spike(df: pd.DataFrame, mult: float = 2.0) -> list[Concept]:
    if len(df) < 25 or "Volume" not in df:
        return []
    a = _arrays(df)
    out: list[Concept] = []
    for i in range(20, len(df)):
        ref = _volume_mean(a["v"], i, 20)
        if ref <= 0 or a["v"][i] < mult * ref:
            continue
        direction = "bullish" if a["c"][i] > a["o"][i] else "bearish"
        out.append(make_concept(
            "volume_spike", direction, df.index[i], df.index[i],
            float(a["h"][i]), float(a["l"][i]),
            min(85, 45 + 20 * min(2.5, a["v"][i] / ref)), "active",
            f"Volume {a['v'][i] / ref:.1f}x the 20-bar average.",
            category="advanced", meta={"ratio": round(float(a["v"][i] / ref), 2)},
        ))
    return capped(out, 25)


@register("absorption", "advanced")
def detect_absorption(df: pd.DataFrame) -> list[Concept]:
    """High volume, small body and a long rejection wick: effort without result."""
    if len(df) < 25 or "Volume" not in df:
        return []
    a = _arrays(df)
    out: list[Concept] = []
    for i in range(20, len(df)):
        ref = _volume_mean(a["v"], i, 20)
        rng = a["h"][i] - a["l"][i]
        body = abs(a["c"][i] - a["o"][i])
        if ref <= 0 or rng <= 0 or a["v"][i] < 1.5 * ref or body > 0.35 * rng:
            continue
        upper = a["h"][i] - max(a["c"][i], a["o"][i])
        lower = min(a["c"][i], a["o"][i]) - a["l"][i]
        if max(upper, lower) < 0.5 * rng:
            continue
        direction = "bearish" if upper > lower else "bullish"
        out.append(make_concept(
            "absorption", direction, df.index[i], df.index[i],
            float(a["h"][i]), float(a["l"][i]), 55, "active",
            "Absorption: high volume with little net progress and a long rejection wick.",
            category="advanced", meta={"volume_ratio": round(float(a["v"][i] / ref), 2)},
        ))
    return capped(out, 20)


@register("exhaustion", "advanced")
def detect_exhaustion(df: pd.DataFrame, window: int = 5) -> list[Concept]:
    """Trend making new extremes on declining volume/delta (exhaustion)."""
    if len(df) < window + 25 or "Volume" not in df:
        return []
    a = _arrays(df)
    out: list[Concept] = []
    for i in range(window + 20, len(df)):
        recent = a["c"][i - window:i + 1]
        prior = a["v"][i - window - 20:i - window]
        now_vol = a["v"][i - window:i + 1]
        if len(prior) < 5 or np.mean(prior) <= 0:
            continue
        if np.mean(now_vol) >= 0.85 * np.mean(prior):
            continue
        if a["c"][i] >= max(recent):
            direction, event = "bearish", "buying_exhaustion"
        elif a["c"][i] <= min(recent):
            direction, event = "bullish", "selling_exhaustion"
        else:
            continue
        if not _recent(out, "exhaustion", i, df.index, 10):
            out.append(make_concept(
                "exhaustion", direction, df.index[i], df.index[i],
                float(a["h"][i]), float(a["l"][i]), 48, "active",
                f"{event.replace('_', ' ').title()}: new extreme on falling volume.",
                category="advanced", meta={"event": event,
                    "volume_ratio": round(float(np.mean(now_vol) / np.mean(prior)), 2)},
            ))
    return capped(out, 15)


# ---------------------------------------------------------------------------
# 5. Harmonic patterns (low weight)
# ---------------------------------------------------------------------------

_HARMONICS = {
    "gartley": ((0.382, 0.618), (0.382, 0.886), (0.75, 0.85), (1.27, 1.68)),
    "bat": ((0.382, 0.50), (0.382, 0.886), (0.85, 0.95), (1.618, 2.618)),
    "butterfly": ((0.75, 0.85), (0.382, 0.886), (1.20, 1.40), (1.618, 2.24)),
    "crab": ((0.382, 0.618), (0.382, 0.886), (1.55, 1.70), (2.618, 3.14)),
}


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


@register("harmonic_pattern", "advanced")
def detect_harmonic_pattern(df: pd.DataFrame) -> list[Concept]:
    """Gartley/Bat/Butterfly/Crab from 5 pivots with Fib tolerances (low weight)."""
    if len(df) < 80:
        return []
    pivots = swing_points(df, lookback=3, confirm=1)
    out: list[Concept] = []
    for end in range(5, len(pivots) + 1):
        pts = pivots[end - 5:end]
        kinds = [p["kind"] for p in pts]
        if kinds not in (["low", "high", "low", "high", "low"],
                         ["high", "low", "high", "low", "high"]):
            continue
        X, A, B, C, D = [p["price"] for p in pts]
        xa = abs(A - X)
        if xa <= 0:
            continue
        xab = abs(B - X) / xa
        ab = abs(B - A)
        if ab <= 0:
            continue
        bc_ab = abs(C - B) / ab
        xad = abs(D - X) / xa
        cd_bc = abs(D - C) / abs(C - B) if abs(C - B) > 0 else 0
        bullish = kinds[0] == "low"
        for name, (b_range, c_range, d_range, cd_range) in _HARMONICS.items():
            if (_in_range(xab, b_range) and _in_range(bc_ab, c_range) and
                    _in_range(xad, d_range) and _in_range(cd_bc, cd_range)):
                last = pts[-1]
                out.append(make_concept(
                    "harmonic_pattern", "bullish" if bullish else "bearish",
                    df.index[last["confirm_index"]], df.index[last["confirm_index"]],
                    max(X, A, B, C, D), min(X, A, B, C, D), 25, "active",
                    f"Possible {name.title()} harmonic completion - possible count, NOT a certainty.",
                    category="advanced",
                    meta={"pattern": name, "xab": round(xab, 3), "bc_ab": round(bc_ab, 3),
                          "xad": round(xad, 3), "cd_bc": round(cd_bc, 3), "confidence": "low"},
                ))
                break
    return capped(out, 10)


# ---------------------------------------------------------------------------
# 6. Supply/demand zone quality
# ---------------------------------------------------------------------------

@register("zone_quality", "advanced")
def detect_zone_quality(df: pd.DataFrame, window: int = 120) -> list[Concept]:
    """Freshness, departure strength, time-at-zone and test count for S/D zones."""
    if len(df) < 60:
        return []
    a = _arrays(df)
    atr = atr_series(df, 14).to_numpy(float)
    body_all = np.abs(a["c"] - a["o"])
    out: list[Concept] = []
    step = max(5, window // 6)
    for end in range(window, len(df) + 1, step):
        start = end - window
        av = np.nanmean(body_all[start:end])
        if not np.isfinite(av) or av <= 0:
            continue
        for i in range(start + 2, min(end - 2, len(df))):
            if body_all[i] < 2.0 * av or body_all[i] <= 0:
                continue
            bullish = a["c"][i] > a["o"][i]
            ztop, zbottom = (max(a["o"][i], a["c"][i]), min(a["o"][i], a["c"][i]))
            after = a["l"][i + 1:end] if bullish else a["h"][i + 1:end]
            touched = int(((after <= ztop) & (after >= zbottom)).sum()) if bullish else \
                      int(((after <= ztop) & (after >= zbottom)).sum())
            departure = float(body_all[i] / (atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else av))
            age = end - 1 - i
            score = 30 + min(35, departure * 10) + (25 if touched == 0 else max(0, 15 - 5 * touched))
            score -= min(20, age / 5)
            if not _recent(out, "zone_quality", end - 1, df.index, 8):
                out.append(make_concept(
                    "zone_quality", "bullish" if bullish else "bearish",
                    df.index[i], df.index[end - 1], ztop, zbottom,
                    float(np.clip(score, 10, 95)), "active",
                    f"{'Demand' if bullish else 'Supply'} zone: {touched} tests, "
                    f"{'fresh' if touched == 0 else 'tested'}, departure {departure:.1f} ATR.",
                    category="advanced",
                    meta={"tests": touched, "fresh": touched == 0, "age_bars": age,
                          "departure_atr": round(departure, 2)},
                ))
            break
    return capped(out, 20)


# ---------------------------------------------------------------------------
# 7. Range / breakout quality
# ---------------------------------------------------------------------------

@register("false_breakout", "advanced")
def detect_false_breakout(df: pd.DataFrame, window: int = 20, lookahead: int = 3) -> list[Concept]:
    """Break beyond a range that is reclaimed within `lookahead` bars (fakeout)."""
    if len(df) < window + lookahead + 5:
        return []
    a = _arrays(df)
    out: list[Concept] = []
    for i in range(window, len(df) - lookahead):
        hi = float(np.max(a["h"][i - window:i]))
        lo = float(np.min(a["l"][i - window:i]))
        if hi <= lo:
            continue
        direction: str | None = None
        event = ""
        if a["c"][i] > hi:
            next_low = float(np.min(a["l"][i + 1:i + 1 + lookahead]))
            if next_low < hi:
                direction, event = "bearish", "bull_trap"
        elif a["c"][i] < lo:
            next_high = float(np.max(a["h"][i + 1:i + 1 + lookahead]))
            if next_high > lo:
                direction, event = "bullish", "bear_trap"
        if direction is not None and not _recent(out, "false_breakout", i, df.index, 8):
            out.append(make_concept(
                "false_breakout", direction, df.index[i], df.index[i + lookahead],
                hi, lo, 62, "active",
                f"False breakout ({event.replace('_', ' ')}): level reclaimed within {lookahead} bars.",
                category="advanced", meta={"event": event, "level": hi if direction == "bearish" else lo},
            ))
    return capped(out, 20)


@register("breakout_retest", "advanced")
def detect_breakout_retest(df: pd.DataFrame, window: int = 20) -> list[Concept]:
    """Broken range level retested and held (quality continuation setup)."""
    if len(df) < window + 8:
        return []
    a = _arrays(df)
    out: list[Concept] = []
    for i in range(window + 3, len(df)):
        hi = float(np.max(a["h"][i - window - 3:i - 3]))
        lo = float(np.min(a["l"][i - window - 3:i - 3]))
        if hi <= lo:
            continue
        broke_up = a["c"][i - 3] > hi
        broke_down = a["c"][i - 3] < lo
        if broke_up and lo <= a["l"][i] <= hi and a["c"][i] > hi:
            direction, level = "bullish", hi
        elif broke_down and lo <= a["h"][i] <= hi and a["c"][i] < lo:
            direction, level = "bearish", lo
        else:
            continue
        if not _recent(out, "breakout_retest", i, df.index, 10):
            out.append(make_concept(
                "breakout_retest", direction, df.index[i], df.index[i],
                hi, lo, 60, "active",
                f"Breakout retest of {level:.2f} held - continuation quality filter.",
                category="advanced", meta={"level": float(level)},
            ))
    return capped(out, 20)


@register("volatility_squeeze", "advanced")
def detect_volatility_squeeze(df: pd.DataFrame, window: int = 100) -> list[Concept]:
    """ATR/BB compression (squeeze) - potential pre-breakout energy."""
    if len(df) < window + 5:
        return []
    a = _arrays(df)
    rng = a["h"] - a["l"]
    series = pd.Series(rng)
    median = series.rolling(window, min_periods=40).median().to_numpy()
    q20 = series.rolling(window, min_periods=40).quantile(0.2).to_numpy()
    out: list[Concept] = []
    for i in range(window, len(df)):
        if not np.isfinite(median[i]) or not np.isfinite(q20[i]):
            continue
        if rng[i] <= q20[i] and median[i] > 0:
            if not _recent(out, "volatility_squeeze", i, df.index, 15):
                out.append(make_concept(
                    "volatility_squeeze", "neutral", df.index[i], df.index[i],
                    float(a["h"][i]), float(a["l"][i]),
                    min(70, 40 + 20 * (1 - rng[i] / median[i])), "active",
                    f"Volatility squeeze: range in bottom 20% of {window}-bar distribution.",
                    category="advanced",
                    meta={"range": float(rng[i]), "median": float(median[i])},
                ))
    return capped(out, 15)


@register("nr7", "advanced")
def detect_nr7(df: pd.DataFrame, period: int = 7) -> list[Concept]:
    """Narrowest range in 7 bars (NR7) - contraction before expansion."""
    if len(df) < period + 2:
        return []
    a = _arrays(df)
    rng = a["h"] - a["l"]
    out: list[Concept] = []
    for i in range(period - 1, len(df)):
        window = rng[i - period + 1:i + 1]
        if rng[i] <= np.min(window) and rng[i] > 0:
            out.append(make_concept(
                "nr7", "neutral", df.index[i], df.index[i],
                float(a["h"][i]), float(a["l"][i]), 45, "active",
                "NR7: narrowest 7-bar range, contraction often precedes expansion.",
                category="advanced", meta={"range": float(rng[i])},
            ))
    return capped(out, 20)


@register("inside_bar_cluster", "advanced")
def detect_inside_bar_cluster(df: pd.DataFrame, minimum: int = 2) -> list[Concept]:
    """Two or more consecutive inside bars (compression)."""
    if len(df) < 5:
        return []
    a = _arrays(df)
    out: list[Concept] = []
    streak = 0
    for i in range(1, len(df)):
        inside = a["h"][i] < a["h"][i - 1] and a["l"][i] > a["l"][i - 1]
        streak = streak + 1 if inside else 0
        if streak >= minimum:
            out.append(make_concept(
                "inside_bar_cluster", "neutral", df.index[i], df.index[i],
                float(a["h"][i]), float(a["l"][i]), 42, "active",
                f"{streak} consecutive inside bars - compression cluster.",
                category="advanced", meta={"bars": streak},
            ))
    return capped(out, 15)


# ---------------------------------------------------------------------------
# 8. Session statistics (gold-specific)
# ---------------------------------------------------------------------------

@register("session_statistics", "advanced")
def detect_session_statistics(df: pd.DataFrame, lookback: int = 60) -> list[Concept]:
    """Session/hour range stats, Asian break and where the daily H/L forms."""
    if len(df) < lookback or "Volume" not in df.columns and len(df) < 100:
        return []
    ts = pd.to_datetime(df.index, utc=True)
    minutes = pd.Series(ts.hour * 60 + ts.minute, index=df.index)
    if (minutes.diff().dropna() < 30).mean() < 0.5:
        return []  # daily data cannot resolve sessions -> N/A
    frame = df.copy()
    frame["session"] = [session_of(t) for t in df.index]
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "range": 0.0})
    for _, row in frame.iloc[-lookback * 24:].iterrows():
        key = str(row["session"])
        stats[key]["count"] += 1
        stats[key]["range"] += float(row["High"] - row["Low"])
    summary = {k: {"bars": int(v["count"]),
                   "avg_range": round(v["range"] / max(1, v["count"]), 3)}
               for k, v in stats.items()}
    high_session = max(stats.items(), key=lambda kv: kv[1]["range"])[0] if stats else "N/A"
    current = session_of(df.index[-1])
    asian = frame[frame["session"] == "asian"].tail(24)
    asian_high = float(asian["High"].max()) if len(asian) else None
    asian_low = float(asian["Low"].min()) if len(asian) else None
    last = float(df["Close"].iloc[-1])
    event = "in_asian_range"
    direction = "neutral"
    if asian_high is not None and last > asian_high:
        event, direction = "asian_range_break_up", "bullish"
    elif asian_low is not None and last < asian_low:
        event, direction = "asian_range_break_down", "bearish"
    return [make_concept(
        "session_statistics", direction, df.index[-1], df.index[-1],
        float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), 40, "active",
        f"Session stats ({current}): most active range in {high_session}; {event.replace('_', ' ')}.",
        category="advanced",
        meta={"summary": summary, "most_active_session": high_session,
              "event": event, "asian_high": asian_high, "asian_low": asian_low},
    )]
