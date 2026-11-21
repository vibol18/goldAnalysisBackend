"""Smart-money-concept detectors (causal, registry-driven).

Every public ``detect_*`` function is pure and strictly causal: a concept
stamped at bar ``i`` only depends on candles at or before ``i`` (status may
later evolve as price mitigates a level).  Geometry is never computed from
future data, so truncating the frame cannot rewrite a past concept's geometry.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from app.concepts.base import (
    UNLIMITED,
    Concept,
    atr,
    body,
    body as _body,
    candle_range,
    displacement_mask,
    ensure_frame,
    in_killzone,
    is_bear,
    is_bull,
    latest_atr,
    latest_price,
    lower_wick,
    make_concept,
    overlap,
    pivots_before,
    register,
    session_of,
    signed_body,
    swing_points,
    upper_wick,
    zone_midpoint,
)

ZONE_CAP = 30
EVENT_CAP = 10

ZONE_TYPES = {
    "fvg", "ifvg", "order_block", "breaker_block", "mitigation_block",
    "equal_highs", "equal_lows", "premium_discount", "ote", "dealing_range",
    "session_high", "session_low", "pdh", "pdl", "pwh", "pwl",
    "daily_open", "midnight_open",
}
EVENT_TYPES = {
    "liquidity_sweep", "displacement", "killzone", "power_of_three",
    "judas_swing", "smt_divergence",
}

KILLZONES: dict[str, tuple[int, int]] = {
    "asian": (23, 7), "london": (7, 12), "new_york": (12, 16),
}


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _prep(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise a frame; return None for empty/invalid input instead of raising."""
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    try:
        return ensure_frame(df)
    except Exception:
        return None


def _arrays(df: pd.DataFrame):
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    return o, h, l, c, valid


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Causal ATR with a short-frame fallback (expanding mean of ranges)."""
    a = atr(df, period)
    fallback = candle_range(df).expanding(min_periods=1).mean()
    try:
        return a.fillna(fallback)
    except Exception:
        return fallback

def _cap(concepts: list[Concept]) -> list[Concept]:
    """Keep at most ZONE_CAP/EVENT_CAP newest concepts per type, order preserved."""
    if UNLIMITED:
        return concepts
    counts: dict[str, int] = {}
    kept: list[Concept] = []
    for c in reversed(concepts):
        cap = EVENT_CAP if c.type in EVENT_TYPES else ZONE_CAP
        n = counts.get(c.type, 0)
        if n < cap:
            kept.append(c)
            counts[c.type] = n + 1
    kept.reverse()
    return kept

def _index_of(df: pd.DataFrame, ts) -> int | None:
    try:
        loc = df.index.get_loc(ts)
    except (KeyError, TypeError):
        return None
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, np.ndarray):
        return int(np.asarray(loc).ravel()[0]) if loc.size else None
    if isinstance(loc, (int, np.integer)):
        return int(loc)
    return None


def _zone_of(ts) -> str | None:
    hour = pd.Timestamp(ts).hour
    if hour >= 23 or hour < 7:
        return "asian"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "new_york"
    return None


def _zone_status(df: pd.DataFrame, index: int, direction: str,
                 top: float, bottom: float) -> str:
    """Generic zone mitigation status from candles after ``index``."""
    later = df.iloc[index + 1:]
    if later.empty:
        return "active"
    if direction == "bullish":
        touched = bool((later["Low"] <= top).any())
        through = bool((later["Close"] < bottom).any())
    else:
        touched = bool((later["High"] >= bottom).any())
        through = bool((later["Close"] > top).any())
    return "invalidated" if through else ("mitigated" if touched else "active")


# --------------------------------------------------------------------------- #
# pattern iterators
# --------------------------------------------------------------------------- #
def _fvg_events(df: pd.DataFrame, min_atr_frac: float = 0.1) -> list[dict]:
    n = len(df)
    if n < 3:
        return []
    a = _atr_series(df).to_numpy(float)
    o, h, l, c, valid = _arrays(df)
    idx = df.index
    out: list[dict] = []
    for i in range(2, n):
        if not (valid[i - 2] and valid[i - 1] and valid[i]):
            continue
        av = a[i]
        if not np.isfinite(av) or av <= 0:
            continue
        min_gap = min_atr_frac * av
        if l[i] > h[i - 2]:
            gap = float(l[i] - h[i - 2])
            if gap >= min_gap:
                out.append({"direction": "bullish", "top": float(l[i]),
                            "bottom": float(h[i - 2]), "gap": gap,
                            "start": idx[i], "end": idx[i], "index": i})
        elif h[i] < l[i - 2]:
            gap = float(l[i - 2] - h[i])
            if gap >= min_gap:
                out.append({"direction": "bearish", "top": float(l[i - 2]),
                            "bottom": float(h[i]), "gap": gap,
                            "start": idx[i], "end": idx[i], "index": i})
    return out


def _fvg_status(df: pd.DataFrame, ev: dict) -> str:
    i, top, bottom = ev["index"], ev["top"], ev["bottom"]
    later = df.iloc[i + 1:]
    if later.empty:
        return "active"
    if ev["direction"] == "bullish":
        touched = bool((later["Low"] <= top).any())
        fully = bool((later["Close"] <= bottom).any())
    else:
        touched = bool((later["High"] >= bottom).any())
        fully = bool((later["Close"] >= top).any())
    return "invalidated" if fully else ("mitigated" if touched else "active")


def _displacement_events(df: pd.DataFrame, mult: float = 1.5,
                         wick_ratio: float = 0.5) -> list[dict]:
    n = len(df)
    if n == 0:
        return []
    dm = displacement_mask(df, mult=mult, wick_ratio=wick_ratio,
                           atr_s=_atr_series(df)).to_numpy(bool)
    sb = signed_body(df).to_numpy(float)
    o, h, l, c, valid = _arrays(df)
    out: list[dict] = []
    for i in range(n):
        if not (dm[i] and valid[i]):
            continue
        direction = "bullish" if sb[i] > 0 else ("bearish" if sb[i] < 0 else "neutral")
        out.append({"direction": direction, "index": i, "start": df.index[i],
                    "end": df.index[i], "top": float(h[i]), "bottom": float(l[i]),
                    "body": abs(float(sb[i]))})
    return out


def _ob_events(df: pd.DataFrame, lookback: int = 3, mult: float = 1.5,
               wick_ratio: float = 0.5) -> list[dict]:
    """Last opposite-colour candle before a structure-breaking displacement."""
    n = len(df)
    if n < 2:
        return []
    o, h, l, c, valid = _arrays(df)
    sb = signed_body(df).to_numpy(float)
    dm = displacement_mask(df, mult=mult, wick_ratio=wick_ratio,
                           atr_s=_atr_series(df)).to_numpy(bool)
    out: list[dict] = []
    used: set[int] = set()
    for j in range(1, n):
        if not (dm[j] and valid[j]):
            continue
        direction = "bullish" if sb[j] > 0 else ("bearish" if sb[j] < 0 else None)
        if direction is None:
            continue
        lo = max(0, j - lookback)
        prior_h = h[lo:j]
        prior_l = l[lo:j]
        prior_h = prior_h[np.isfinite(prior_h)]
        prior_l = prior_l[np.isfinite(prior_l)]
        if direction == "bullish":
            if prior_h.size == 0 or not (c[j] > prior_h.max()):
                continue
        else:
            if prior_l.size == 0 or not (c[j] < prior_l.min()):
                continue
        ob = None
        for k in range(j - 1, -1, -1):
            if not valid[k]:
                continue
            if direction == "bullish" and sb[k] < 0:
                ob = k
                break
            if direction == "bearish" and sb[k] > 0:
                ob = k
                break
        if ob is None or ob in used:
            continue
        used.add(ob)
        out.append({"direction": direction, "index": j, "start": df.index[j],
                    "end": df.index[j], "ob_index": ob, "ob_time": df.index[ob],
                    "top": float(h[ob]), "bottom": float(l[ob])})
    return out


def _equal_levels(df: pd.DataFrame, kind: str,
                  tol_frac: float = 0.15) -> list[dict]:
    """Cluster confirmed swing highs/lows within an ATR price tolerance.

    A level is emitted exactly once, at the bar its second swing confirms.
    Its price and stamp therefore depend only on past pivots — adding later
    swings can create new levels but can never rewrite an existing one.
    """
    a = _atr_series(df).to_numpy(float)
    pivots = swing_points(df, lookback=1, confirm=1, kind=kind)
    if len(pivots) < 2:
        return []
    out: list[dict] = []
    cluster: list[dict] = [pivots[0]]
    emitted = False
    for p in pivots[1:]:
        ref = float(np.mean([x["price"] for x in cluster]))
        av = a[p["confirm_index"]]
        tol = tol_frac * av if np.isfinite(av) else 0.0
        if abs(p["price"] - ref) <= tol:
            cluster.append(p)
            if not emitted:
                members = cluster[:]
                out.append({
                    "kind": kind,
                    "price": float(np.mean([x["price"] for x in members])),
                    "index": int(p["confirm_index"]),
                    "start": p["confirm_time"], "end": p["confirm_time"],
                    "prices": [float(x["price"]) for x in members],
                    "times": [x["pivot_time"] for x in members],
                })
                emitted = True
        else:
            cluster = [p]
            emitted = False
    return out


def _resample_levels(df: pd.DataFrame, rule: str):
    """Completed-period high/low levels as (bar_index, price, side)."""
    out = []
    try:
        groups = [(k, g) for k, g in df.resample(rule) if len(g) > 0]
    except Exception:
        return out
    if len(groups) < 2:
        return out
    for _k, g in groups[:-1]:  # skip the in-progress final period
        ii = _index_of(df, g.index[-1])
        if ii is None:
            continue
        out.append((ii, float(g["High"].max()), "high"))
        out.append((ii, float(g["Low"].min()), "low"))
    return out


def _session_groups(df: pd.DataFrame):
    groups: OrderedDict[tuple, list] = OrderedDict()
    for ts in df.index:
        t = pd.Timestamp(ts)
        groups.setdefault((t.date(), session_of(t)), []).append(ts)
    return groups


def _session_levels(df: pd.DataFrame):
    groups = _session_groups(df)
    items = list(groups.items())
    out = []
    for _key, tss in items[:-1]:  # skip the in-progress final session
        sub = df.loc[tss]
        ii = _index_of(df, tss[-1])
        if ii is None:
            continue
        out.append((ii, float(sub["High"].max()), "high"))
        out.append((ii, float(sub["Low"].min()), "low"))
    return out


def _liquidity_levels(df: pd.DataFrame) -> list[tuple[int, float, str]]:
    out: list[tuple[int, float, str]] = []
    for ev in _equal_levels(df, "high"):
        out.append((ev["index"], ev["price"], "high"))
    for ev in _equal_levels(df, "low"):
        out.append((ev["index"], ev["price"], "low"))
    out.extend(_resample_levels(df, "D"))
    out.extend(_resample_levels(df, "W"))
    out.extend(_session_levels(df))
    return out


# --------------------------------------------------------------------------- #
# zone detectors
# --------------------------------------------------------------------------- #
@register("fvg", "smc")
def detect_fvg(df: pd.DataFrame, min_atr_frac: float = 0.1) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    a = _atr_series(frame)
    out: list[Concept] = []
    for ev in _fvg_events(frame, min_atr_frac):
        av = float(a.iloc[ev["index"]]) or 1e-9
        status = _fvg_status(frame, ev)
        strength = float(np.clip(45.0 + 15.0 * ev["gap"] / av, 0.0, 100.0))
        out.append(make_concept(
            "fvg", ev["direction"], ev["start"], ev["end"],
            ev["top"], ev["bottom"], strength, status,
            f"{ev['direction'].title()} fair value gap ({ev['gap']:.2f})",
            category="smc",
            meta={"gap": ev["gap"], "min_gap": min_atr_frac * av,
                  "fill_state": {"active": "open", "mitigated": "partially filled",
                                 "invalidated": "fully mitigated"}[status]},
        ))
    return _cap(out)


@register("ifvg", "smc")
def detect_ifvg(df: pd.DataFrame, min_atr_frac: float = 0.1) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    n = len(frame)
    o, h, l, c, valid = _arrays(frame)
    sb = signed_body(frame).to_numpy(float)
    dm = displacement_mask(frame, atr_s=_atr_series(frame)).to_numpy(bool)
    out: list[Concept] = []
    for fv in _fvg_events(frame, min_atr_frac):
        i, top, bottom = fv["index"], fv["top"], fv["bottom"]
        want = "bullish" if fv["direction"] == "bearish" else "bearish"
        for k in range(i + 1, n):
            if not (dm[k] and valid[k]):
                continue
            if (sb[k] > 0) != (want == "bullish"):
                continue
            through = c[k] > top if want == "bullish" else c[k] < bottom
            if not through:
                continue
            out.append(make_concept(
                "ifvg", want, frame.index[k], frame.index[k], top, bottom,
                80.0, "active",
                f"Inverted FVG: {fv['direction']} gap reclaimed by {want} displacement",
                category="smc",
                meta={"original_direction": fv["direction"],
                      "original_fvg_time": fv["start"], "inversion_time": frame.index[k]},
            ))
            break
    return _cap(out)


@register("order_block", "smc")
def detect_order_block(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    out: list[Concept] = []
    for ob in _ob_events(frame):
        status = _zone_status(frame, ob["index"], ob["direction"],
                              ob["top"], ob["bottom"])
        out.append(make_concept(
            "order_block", ob["direction"], ob["start"], ob["end"],
            ob["top"], ob["bottom"], 75.0, status,
            f"{ob['direction'].title()} order block (last opposite candle "
            f"before structure-breaking displacement)",
            category="smc",
            meta={"ob_time": ob["ob_time"], "ob_index": ob["ob_index"],
                  "displacement_time": ob["end"]},
        ))
    return _cap(out)


@register("breaker_block", "smc")
def detect_breaker_block(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    n = len(frame)
    o, h, l, c, valid = _arrays(frame)
    out: list[Concept] = []
    for ob in _ob_events(frame):
        j, top, bottom = ob["index"], ob["top"], ob["bottom"]
        direction = None
        taken = None
        for k in range(j + 1, n):
            if not valid[k]:
                continue
            if ob["direction"] == "bearish" and c[k] > top:
                direction, taken = "bullish", k
                break
            if ob["direction"] == "bullish" and c[k] < bottom:
                direction, taken = "bearish", k
                break
        if direction is None:
            continue
        out.append(make_concept(
            "breaker_block", direction, frame.index[taken], frame.index[taken],
            top, bottom, 75.0, "active",
            f"Failed {ob['direction']} order block reclaimed -> {direction} breaker",
            category="smc",
            meta={"reclaim_time": frame.index[taken],
                  "original_ob_time": ob["ob_time"],
                  "original_ob_direction": ob["direction"]},
        ))
    return _cap(out)


@register("mitigation_block", "smc")
def detect_mitigation_block(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    n = len(frame)
    o, h, l, c, valid = _arrays(frame)
    out: list[Concept] = []
    for ob in _ob_events(frame):
        j, top, bottom = ob["index"], ob["top"], ob["bottom"]
        pull = None
        for k in range(j + 1, min(j + 4, n)):
            if not valid[k]:
                continue
            if ob["direction"] == "bullish" and l[k] <= top and l[k] >= bottom:
                pull = k
                break
            if ob["direction"] == "bearish" and h[k] >= bottom and h[k] <= top:
                pull = k
                break
        if pull is None:
            continue
        out.append(make_concept(
            "mitigation_block", ob["direction"], frame.index[pull], frame.index[pull],
            top, bottom, 65.0, "active",
            f"{ob['direction'].title()} mitigation block: displacement pulled back "
            f"into prior opposite candle",
            category="smc",
            meta={"ob_time": ob["ob_time"], "displacement_time": ob["end"],
                  "pullback_time": frame.index[pull]},
        ))
    return _cap(out)


@register("equal_highs", "smc")
def detect_equal_highs(df: pd.DataFrame, tol_atr_frac: float = 0.15) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    a = _atr_series(frame).to_numpy(float)
    n = len(frame)
    o, h, l, c, valid = _arrays(frame)
    out: list[Concept] = []
    for ev in _equal_levels(frame, "high", tol_atr_frac):
        price = ev["price"]
        av = a[ev["index"]]
        tol = tol_atr_frac * av if np.isfinite(av) else 0.0
        swept = bool((h[ev["index"] + 1:] > price + tol).any())
        out.append(make_concept(
            "equal_highs", "bearish", ev["start"], ev["end"], price, price,
            55.0, "invalidated" if swept else "active",
            f"Equal highs ({len(ev['prices'])} swings) at {price:.2f}",
            category="smc",
            meta={"price": price, "count": len(ev["prices"]),
                  "prices": ev["prices"], "swept": swept},
        ))
    return _cap(out)


@register("equal_lows", "smc")
def detect_equal_lows(df: pd.DataFrame, tol_atr_frac: float = 0.15) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    a = _atr_series(frame).to_numpy(float)
    n = len(frame)
    o, h, l, c, valid = _arrays(frame)
    out: list[Concept] = []
    for ev in _equal_levels(frame, "low", tol_atr_frac):
        price = ev["price"]
        av = a[ev["index"]]
        tol = tol_atr_frac * av if np.isfinite(av) else 0.0
        swept = bool((l[ev["index"] + 1:] < price - tol).any())
        out.append(make_concept(
            "equal_lows", "bullish", ev["start"], ev["end"], price, price,
            55.0, "invalidated" if swept else "active",
            f"Equal lows ({len(ev['prices'])} swings) at {price:.2f}",
            category="smc",
            meta={"price": price, "count": len(ev["prices"]),
                  "prices": ev["prices"], "swept": swept},
        ))
    return _cap(out)


# --------------------------------------------------------------------------- #
# event detectors
# --------------------------------------------------------------------------- #
@register("liquidity_sweep", "smc")
def detect_liquidity_sweep(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    n = len(frame)
    o, h, l, c, valid = _arrays(frame)
    a = _atr_series(frame).to_numpy(float)
    out: list[Concept] = []
    for L, price, side in _liquidity_levels(frame):
        if not valid[L] or L >= n - 1:
            continue
        for j in range(L + 1, n):
            if not valid[j]:
                continue
            av = a[j] if np.isfinite(a[j]) else 0.0
            eps = 0.02 * av
            if side == "high" and h[j] > price + eps:
                for k in range(j, min(j + 4, n)):
                    if valid[k] and c[k] < price:
                        out.append(make_concept(
                            "liquidity_sweep", "bearish", frame.index[k], frame.index[k],
                            float(max(price, h[j])), float(min(price, h[j])),
                            70.0, "active",
                            f"Buy-side liquidity sweep: wick above {price:.2f}, "
                            f"closed back inside",
                            category="smc",
                            meta={"swept_price": price, "side": "high",
                                  "level_time": frame.index[L],
                                  "confirmation_time": frame.index[k]},
                        ))
                        break
                break
            if side == "low" and l[j] < price - eps:
                for k in range(j, min(j + 4, n)):
                    if valid[k] and c[k] > price:
                        out.append(make_concept(
                            "liquidity_sweep", "bullish", frame.index[k], frame.index[k],
                            float(max(price, l[j])), float(min(price, l[j])),
                            70.0, "active",
                            f"Sell-side liquidity sweep: wick below {price:.2f}, "
                            f"closed back inside",
                            category="smc",
                            meta={"swept_price": price, "side": "low",
                                  "level_time": frame.index[L],
                                  "confirmation_time": frame.index[k]},
                        ))
                        break
                break
    return _cap(out)


@register("displacement", "smc")
def detect_displacement(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    a = _atr_series(frame).to_numpy(float)
    out: list[Concept] = []
    for ev in _displacement_events(frame):
        av = a[ev["index"]]
        strength = float(np.clip(55.0 + 15.0 * ev["body"] / (av if np.isfinite(av) and av > 0 else 1e-9),
                                 0.0, 100.0))
        out.append(make_concept(
            "displacement", ev["direction"], ev["start"], ev["end"],
            ev["top"], ev["bottom"], strength, "active",
            f"{ev['direction'].title()} displacement candle",
            category="smc",
            meta={"body": ev["body"], "atr": float(av) if np.isfinite(av) else None},
        ))
    return _cap(out)


@register("killzone", "smc")
def detect_killzone(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    o, h, l, c, valid = _arrays(frame)
    out: list[Concept] = []
    prev: str | None = None
    for i, ts in enumerate(frame.index):
        zone = _zone_of(ts)
        if zone is None:
            prev = None
            continue
        if zone != prev:
            in_z = in_killzone(ts, KILLZONES)
            out.append(make_concept(
                "killzone", "neutral", ts, ts,
                float(h[i]) if valid[i] else None,
                float(l[i]) if valid[i] else None,
                30.0, "active", f"Entered {zone} killzone",
                category="smc",
                meta={"zone": zone, "session": session_of(ts),
                      "in_killzone": bool(in_z)},
            ))
        prev = zone
    return _cap(out)


@register("power_of_three", "smc")
def detect_power_of_three(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    groups: OrderedDict[tuple, list] = OrderedDict()
    for ts in frame.index:
        groups.setdefault(pd.Timestamp(ts).date(), []).append(ts)
    items = list(groups.items())
    out: list[Concept] = []
    for day, tss in items[:-1]:  # only fully completed UTC days
        sub = frame.loc[tss]
        hours = sub.index.hour
        asian = sub[(hours < 7)]
        london = sub[(hours >= 7) & (hours < 12)]
        ny = sub[(hours >= 12) & (hours < 16)]
        if asian.empty or london.empty or ny.empty:
            continue
        ah, al = float(asian["High"].max()), float(asian["Low"].min())
        lh, ll = float(london["High"].max()), float(london["Low"].min())
        manipulation = bool(lh > ah or ll < al)
        if not manipulation:
            continue
        nycl = float(ny["Close"].iloc[-1])
        if nycl > ah:
            direction = "bullish"
        elif nycl < al:
            direction = "bearish"
        else:
            continue
        stamp = tss[-1]
        out.append(make_concept(
            "power_of_three", direction, stamp, stamp, ah, al, 70.0, "active",
            f"Power of three on {day}: accumulation / manipulation / distribution",
            category="smc",
            meta={"day": str(day), "accumulation_high": ah, "accumulation_low": al,
                  "manipulation": manipulation, "distribution_close": nycl},
        ))
    return _cap(out)


@register("judas_swing", "smc")
def detect_judas_swing(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    n = len(frame)
    sb = signed_body(frame).to_numpy(float)
    dm = displacement_mask(frame, atr_s=_atr_series(frame)).to_numpy(bool)
    o, h, l, c, valid = _arrays(frame)
    out: list[Concept] = []
    for i, ts in enumerate(frame.index):
        if pd.Timestamp(ts).hour not in (7, 12):
            continue
        win = range(i + 1, min(i + 4, n))
        disp = [k for k in win if dm[k] and valid[k]]
        if not disp:
            continue
        d = disp[0]
        for r in range(d + 1, min(i + 4, n)):
            if dm[r] and valid[r] and np.sign(sb[r]) != np.sign(sb[d]):
                direction = "bullish" if sb[r] > 0 else "bearish"
                out.append(make_concept(
                    "judas_swing", direction, frame.index[r], frame.index[r],
                    float(h[r]), float(l[r]), 70.0, "active",
                    f"Judas swing at {session_of(ts)} open, "
                    f"reversed by {direction} displacement",
                    category="smc",
                    meta={"open_time": ts, "false_move_time": frame.index[d],
                          "reversal_time": frame.index[r]},
                ))
                break
    return _cap(out)


# --------------------------------------------------------------------------- #
# range / level detectors
# --------------------------------------------------------------------------- #
def _last_swings(df: pd.DataFrame):
    pivots = swing_points(df, lookback=1, confirm=1)
    lows = [p for p in pivots if p["kind"] == "low"]
    highs = [p for p in pivots if p["kind"] == "high"]
    return lows[-1] if lows else None, highs[-1] if highs else None


@register("premium_discount", "smc")
def detect_premium_discount(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    low, high = _last_swings(frame)
    if low is None or high is None:
        return []
    bottom, top = sorted((float(low["price"]), float(high["price"])))
    if top <= bottom:
        return []
    price = latest_price(frame)
    pct = float(np.clip((price - bottom) / (top - bottom) * 100.0, 0.0, 100.0))
    mid = zone_midpoint(top, bottom)
    direction = "bullish" if price < mid else ("bearish" if price > mid else "neutral")
    ts = frame.index[-1]
    return _cap([make_concept(
        "premium_discount", direction, ts, ts, top, bottom, 55.0, "active",
        f"Price in {'discount' if direction == 'bullish' else 'premium'} "
        f"zone ({pct:.1f}%)",
        category="smc",
        meta={"position_pct": pct, "equilibrium": mid,
              "swing_low": float(low["price"]), "swing_high": float(high["price"])},
    )])


@register("ote", "smc")
def detect_ote(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    low, high = _last_swings(frame)
    if low is None or high is None:
        return []
    H, L = float(high["price"]), float(low["price"])
    if H <= L:
        return []
    if low["confirm_index"] < high["confirm_index"]:
        top = H - 0.62 * (H - L)
        bottom = H - 0.79 * (H - L)
        direction = "bullish"
    else:
        top = L + 0.79 * (H - L)
        bottom = L + 0.62 * (H - L)
        direction = "bearish"
    ts = frame.index[-1]
    return _cap([make_concept(
        "ote", direction, ts, ts, top, bottom, 60.0, "active",
        f"{direction.title()} optimal trade entry (62%-79% retracement)",
        category="smc",
        meta={"retracement_low_pct": 0.62, "retracement_high_pct": 0.79,
              "impulse_low": L, "impulse_high": H},
    )])


@register("dealing_range", "smc")
def detect_dealing_range(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    low, high = _last_swings(frame)
    if low is None or high is None:
        return []
    bottom, top = sorted((float(low["price"]), float(high["price"])))
    if top <= bottom:
        return []
    mid = zone_midpoint(top, bottom)
    ts = frame.index[-1]
    return _cap([make_concept(
        "dealing_range", "neutral", ts, ts, top, bottom, 50.0, "active",
        "Current dealing range (last swing low to swing high)",
        category="smc",
        meta={"mid": mid, "upper": top, "lower": bottom, "equilibrium": mid},
    )])


def _day_open_events(df: pd.DataFrame) -> list[tuple]:
    groups: OrderedDict[tuple, list] = OrderedDict()
    for ts in df.index:
        groups.setdefault(pd.Timestamp(ts).date(), []).append(ts)
    o, h, l, c, valid = _arrays(df)
    out: list[tuple] = []
    for _day, tss in groups.items():
        first = tss[0]
        fi = _index_of(df, first)
        if fi is None or not valid[fi]:
            continue
        true_open = float(o[fi])
        midnight = next((t for t in tss if pd.Timestamp(t).hour == 0), first)
        mi = _index_of(df, midnight)
        midnight_open = float(o[mi]) if mi is not None and valid[mi] else true_open
        out.append((first, true_open, midnight_open))
    return out


@register("daily_open", "smc")
def detect_daily_open(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    out: list[Concept] = []
    for ts, true_open, midnight_open in _day_open_events(frame):
        out.append(make_concept(
            "daily_open", "neutral", ts, ts, true_open, true_open, 40.0, "active",
            f"Daily open at {true_open:.2f}",
            category="smc",
            meta={"true_day_open": true_open, "midnight_open": midnight_open},
        ))
    return _cap(out)


@register("midnight_open", "smc")
def detect_midnight_open(df: pd.DataFrame) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    out: list[Concept] = []
    for ts, true_open, midnight_open in _day_open_events(frame):
        out.append(make_concept(
            "midnight_open", "neutral", ts, ts, midnight_open, midnight_open,
            40.0, "active", f"Midnight (00:00 UTC) open at {midnight_open:.2f}",
            category="smc",
            meta={"true_day_open": true_open, "midnight_open": midnight_open},
        ))
    return _cap(out)


@register("pdh", "smc")
def detect_pdh(df: pd.DataFrame) -> list[Concept]:
    return _previous_level(df, "D", "pdh", "high", "bearish")


@register("pdl", "smc")
def detect_pdl(df: pd.DataFrame) -> list[Concept]:
    return _previous_level(df, "D", "pdl", "low", "bullish")


@register("pwh", "smc")
def detect_pwh(df: pd.DataFrame) -> list[Concept]:
    return _previous_level(df, "W", "pwh", "high", "bearish")


@register("pwl", "smc")
def detect_pwl(df: pd.DataFrame) -> list[Concept]:
    return _previous_level(df, "W", "pwl", "low", "bullish")


def _previous_level(df: pd.DataFrame, rule: str, type_: str, side: str,
                    direction: str) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    out: list[Concept] = []
    try:
        groups = [(k, g) for k, g in frame.resample(rule) if len(g) > 0]
    except Exception:
        return out
    if len(groups) < 2:
        return out
    for _k, g in groups[:-1]:  # completed periods only
        ts = g.index[-1]
        price = float(g["High"].max() if side == "high" else g["Low"].min())
        out.append(make_concept(
            type_, direction, ts, ts, price, price, 45.0, "active",
            f"Previous {rule} {'high' if side == 'high' else 'low'} at {price:.2f}",
            category="smc",
            meta={"level": price, "period": str(_k)[:10], "side": side},
        ))
    return _cap(out)


@register("session_high", "smc")
def detect_session_high(df: pd.DataFrame) -> list[Concept]:
    return _session_level_concept(df, "session_high", "high", "bearish")


@register("session_low", "smc")
def detect_session_low(df: pd.DataFrame) -> list[Concept]:
    return _session_level_concept(df, "session_low", "low", "bullish")


def _session_level_concept(df: pd.DataFrame, type_: str, side: str,
                           direction: str) -> list[Concept]:
    frame = _prep(df)
    if frame is None:
        return []
    groups = _session_groups(frame)
    items = list(groups.items())
    out: list[Concept] = []
    for (day, label), tss in items[:-1]:
        if label not in ("asian", "london", "new_york"):
            continue
        sub = frame.loc[tss]
        ts = tss[-1]
        price = float(sub["High"].max() if side == "high" else sub["Low"].min())
        out.append(make_concept(
            type_, direction, ts, ts, price, price, 45.0, "active",
            f"{label.title()} session {'high' if side == 'high' else 'low'} "
            f"at {price:.2f}",
            category="smc",
            meta={"level": price, "session": label, "day": str(day), "side": side},
        ))
    return _cap(out)


@register("smt_divergence", "smc")
def detect_smt_divergence(df: pd.DataFrame, dxy_df: pd.DataFrame | None = None,
                          events=None) -> list[Concept]:
    """SMT divergence vs DXY; returns [] when no comparable DXY frame is given."""
    frame = _prep(df)
    if frame is None or dxy_df is None:
        return []
    dxy = _prep(dxy_df)
    if dxy is None or len(dxy) < 2:
        return []
    horizon = frame.index[-1]
    gold = swing_points(frame, lookback=1, confirm=1)
    dsw = [p for p in swing_points(dxy, lookback=1, confirm=1)
           if pd.Timestamp(p["confirm_time"]) <= pd.Timestamp(horizon)]
    gh = [p for p in gold if p["kind"] == "high"]
    gl = [p for p in gold if p["kind"] == "low"]
    dh = [p for p in dsw if p["kind"] == "high"]
    dl = [p for p in dsw if p["kind"] == "low"]
    out: list[Concept] = []
    if len(gh) >= 2 and len(dh) >= 2:
        gold_hh = gh[-1]["price"] > gh[-2]["price"]
        dxy_lh = dh[-1]["price"] < dh[-2]["price"]
        if gold_hh and dxy_lh:
            ts = gh[-1]["confirm_time"]
            out.append(make_concept(
                "smt_divergence", "bearish", ts, ts,
                float(gh[-1]["price"]), float(gh[-2]["price"]), 80.0, "active",
                "Bearish SMT: gold higher high vs DXY lower high",
                category="smc",
                meta={"gold_highs": [gh[-2]["price"], gh[-1]["price"]],
                      "dxy_highs": [dh[-2]["price"], dh[-1]["price"]]},
            ))
    if len(gl) >= 2 and len(dl) >= 2:
        gold_ll = gl[-1]["price"] < gl[-2]["price"]
        dxy_hl = dl[-1]["price"] > dl[-2]["price"]
        if gold_ll and dxy_hl:
            ts = gl[-1]["confirm_time"]
            out.append(make_concept(
                "smt_divergence", "bullish", ts, ts,
                float(gl[-2]["price"]), float(gl[-1]["price"]), 80.0, "active",
                "Bullish SMT: gold lower low vs DXY higher low",
                category="smc",
                meta={"gold_lows": [gl[-2]["price"], gl[-1]["price"]],
                      "dxy_lows": [dl[-2]["price"], dl[-1]["price"]]},
            ))
    return _cap(out)


__all__ = [name for name in globals() if name.startswith("detect_")]
