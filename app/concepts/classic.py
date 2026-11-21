"""Classic technical-analysis concept detectors (pure and strictly causal).

Every detector returns a list of :class:`~app.concepts.base.Concept` objects.
A concept stamped at bar ``i`` is computed exclusively from bars ``0..i``;
patterns needing confirmation are stamped at the confirmation bar. Later price
may evolve ``status`` but never the stored geometry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from app.concepts.base import (
    atr,
    capped,
    ensure_frame,
    latest_atr,
    latest_price,
    make_concept,
    register,
    signed_body,
    swing_points,
    zone_midpoint,
)

MAX = 10


def _f(value: object) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _arr(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype=float)


def _ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, min_periods=window, adjust=False).mean()


def _cap(concepts: list) -> list:
    return capped(concepts, MAX)


def _state(df: pd.DataFrame, name: str, direction: str, top: float | None,
           bottom: float | None, strength: float, notes: str,
           meta: dict | None = None, status: str = "active") -> list:
    f = ensure_frame(df)
    end = f.index[-1] if len(f) else None
    return [make_concept(name, direction, end, end, top, bottom, strength,
                         status=status, notes=notes, category="classic",
                         meta=meta or {})]


@register("sr_zone", "classic")
def detect_sr_zone(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 8:
        return []
    a_vals = _arr(atr(f, 14))
    pivots = swing_points(f, lookback=2, confirm=2)
    clusters: list[dict] = []
    for p in pivots:
        av = _f(a_vals[p["confirm_index"]]) or 0.0
        tol = 0.3 * av if av > 0 else 0.003 * float(p["price"])
        for cl in clusters:
            if cl["kind"] != p["kind"]:
                continue
            mid = float(np.mean(cl["prices"]))
            if abs(float(p["price"]) - mid) <= tol:
                cl["prices"].append(float(p["price"]))
                cl["idxs"].append(int(p["confirm_index"]))
                cl["times"].append(p["confirm_time"])
                break
        else:
            clusters.append({
                "kind": p["kind"], "prices": [float(p["price"])],
                "idxs": [int(p["confirm_index"])], "times": [p["confirm_time"]],
            })
    price = latest_price(f)
    a_now = latest_atr(f, 14)
    out: list = []
    for cl in clusters:
        if len(cl["prices"]) < 2:
            continue
        top = max(cl["prices"])
        bottom = min(cl["prices"])
        order = np.argsort(cl["idxs"])
        times = [cl["times"][j] for j in order]
        support = cl["kind"] == "low"
        direction = "bullish" if support else "bearish"
        status = "active"
        if a_now > 0:
            if support and price < bottom - a_now:
                status = "invalidated"
            elif (not support) and price > top + a_now:
                status = "invalidated"
            elif bottom <= price <= top:
                status = "mitigated"
        out.append(make_concept(
            "sr_zone", direction, times[0], times[-1], top, bottom,
            min(100.0, 25.0 * len(cl["prices"])), status=status,
            notes=f"{len(cl['prices'])}-touch {'support' if support else 'resistance'} zone",
            category="classic",
            meta={"touches": len(cl["prices"]), "kind": cl["kind"],
                  "midpoint": zone_midpoint(top, bottom)},
        ))
    return _cap(out)


@register("supply_demand", "classic")
def detect_supply_demand(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 12:
        return []
    a_vals = _arr(atr(f, 14))
    sb = _arr(signed_body(f))
    high = _arr(f["High"])
    low = _arr(f["Low"])
    typical_body = pd.Series(np.abs(sb)).rolling(20, min_periods=5).median().to_numpy(dtype=float)
    out: list = []
    for i in range(3, len(f)):
        av = _f(a_vals[i])
        tb = _f(typical_body[i])
        if not av or av <= 0 or not np.isfinite(sb[i]):
            continue
        magnitude = abs(sb[i])
        if magnitude < 0.5 * av or (tb and magnitude < 1.8 * tb):
            continue
        up = sb[i] > 0
        start = i - 1
        j = i - 1
        while j >= 0 and (i - start) <= 6:
            bj = sb[j]
            flat = bj <= 0.3 * av if up else bj >= -0.3 * av
            if not flat or not np.isfinite(bj) or (high[j] - low[j]) > 2.5 * av:
                break
            start = j
            j -= 1
        if start >= i:
            start = max(0, i - 1)
        top = float(np.max(high[start:i]))
        bottom = float(np.min(low[start:i]))
        if not (np.isfinite(top) and np.isfinite(bottom)) or top <= bottom:
            continue
        strength = float(np.clip(35 + 25 * abs(sb[i]) / av, 0, 100))
        out.append(make_concept(
            "supply_demand", "bullish" if up else "bearish",
            f.index[start], f.index[i], top, bottom, strength,
            notes=("Demand: flat/bearish base before upward displacement"
                   if up else "Supply: flat/bullish base before downward displacement"),
            category="classic",
            meta={"zone": "demand" if up else "supply",
                  "displacement_atr": round(magnitude / av, 2)},
        ))
    return _cap(out)


def _trend_geometry(f: pd.DataFrame) -> dict | None:
    pivots = swing_points(f, lookback=2, confirm=2)
    n = len(f)
    best: dict | None = None
    for kind in ("low", "high"):
        pts = [p for p in pivots if p["kind"] == kind][-30:]
        if len(pts) < 2:
            continue
        chains: list[list[dict]] = []
        for p in pts:
            chain: list[dict] = []
            for q in reversed(chains):
                last = q[-1]
                ok = (p["price"] > last["price"]) if kind == "low" else (p["price"] < last["price"])
                if ok and p["index"] > last["index"]:
                    chain = q + [p]
                    break
            chains.append(chain or [p])
        for chain in chains:
            if len(chain) < 2:
                continue
            a0, a1 = chain[0], chain[-1]
            span = a1["index"] - a0["index"]
            if span <= 0:
                continue
            slope = (float(a1["price"]) - float(a0["price"])) / span
            level = float(a1["price"]) + slope * (n - 1 - a1["index"])
            if not np.isfinite(level) or not np.isfinite(slope):
                continue
            lo_i = max(0, a0["index"])
            highs = _arr(f["High"])[lo_i:]
            lows = _arr(f["Low"])[lo_i:]
            if kind == "low":
                opposite = float(np.max(highs))
                op_i = lo_i + int(np.argmax(highs))
            else:
                opposite = float(np.min(lows))
                op_i = lo_i + int(np.argmin(lows))
            if not np.isfinite(opposite):
                continue
            op_dev = opposite - (float(a0["price"]) + slope * (op_i - a0["index"]))
            upper = level + op_dev
            lower = level
            if upper < lower:
                upper, lower = lower, upper
            cand = {
                "kind": kind, "chain": chain, "slope": slope, "level": level,
                "upper": upper, "lower": lower,
                "start_time": chain[0]["confirm_time"],
                "end_time": chain[-1]["confirm_time"],
            }
            if best is None or len(chain) > len(best["chain"]) or (
                len(chain) == len(best["chain"]) and cand["end_time"] >= best["end_time"]
            ):
                best = cand
    return best


@register("trendline", "classic")
def detect_trendline(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 12:
        return []
    g = _trend_geometry(f)
    if g is None:
        return []
    bull = g["kind"] == "low"
    return [make_concept(
        "trendline", "bullish" if bull else "bearish",
        g["start_time"], f.index[-1], g["level"], g["level"],
        min(100.0, 40 + 15 * len(g["chain"])),
        notes=f"Auto trendline through {len(g['chain'])} {'ascending lows' if bull else 'descending highs'}",
        category="classic",
        meta={"slope": round(g["slope"], 6), "points": len(g["chain"]),
              "projected_level": g["level"]},
    )]


@register("channel", "classic")
def detect_channel(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 12:
        return []
    g = _trend_geometry(f)
    if g is None:
        return []
    bull = g["kind"] == "low"
    return [make_concept(
        "channel", "bullish" if bull else "bearish",
        g["start_time"], f.index[-1], g["upper"], g["lower"],
        min(100.0, 40 + 15 * len(g["chain"])),
        notes=f"Parallel channel from {len(g['chain'])}-point trendline",
        category="classic",
        meta={"slope": round(g["slope"], 6), "upper": g["upper"],
              "lower": g["lower"], "width": g["upper"] - g["lower"]},
    )]


def _breakout_retest(f: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    pivots = swing_points(f, lookback=2, confirm=2)
    a_vals = _arr(atr(f, 14))
    close = _arr(f["Close"])
    low = _arr(f["Low"])
    high = _arr(f["High"])
    breakouts: list[dict] = []
    retests: list[dict] = []
    last_breakout: dict | None = None
    seen: set[tuple] = set()
    for i in range(10, len(f)):
        av = _f(a_vals[i])
        if not av or av <= 0:
            continue
        past = [p for p in pivots if p["confirm_index"] < i]
        res = [float(p["price"]) for p in past if p["kind"] == "high" and p["price"] > close[i - 1]]
        sup = [float(p["price"]) for p in past if p["kind"] == "low" and p["price"] < close[i - 1]]
        res_level = min(res) if res else None
        sup_level = max(sup) if sup else None
        if last_breakout is not None and i - last_breakout["index"] <= 15:
            lv = last_breakout["level"]
            up = last_breakout["up"]
            touched = (low[i] <= lv + 0.3 * av) if up else (high[i] >= lv - 0.3 * av)
            held = close[i] >= lv - 0.1 * av if up else close[i] <= lv + 0.1 * av
            if touched and held:
                retests.append({"index": i, "level": lv, "up": up,
                                "breakout_index": last_breakout["index"]})
                last_breakout = None
        for level, up in ((res_level, True), (sup_level, False)):
            if level is None:
                continue
            key = (round(level, 2), up)
            broken = (close[i] > level + 0.1 * av and close[i - 1] <= level) if up else (
                close[i] < level - 0.1 * av and close[i - 1] >= level)
            if broken and key not in seen:
                seen.add(key)
                breakouts.append({"index": i, "level": level, "up": up})
                last_breakout = breakouts[-1]
                break
    return breakouts, retests


@register("breakout", "classic")
def detect_breakout(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 12:
        return []
    breakouts, _ = _breakout_retest(f)
    out = [make_concept(
        "breakout", "bullish" if b["up"] else "bearish",
        f.index[b["index"]], f.index[b["index"]], b["level"], b["level"], 55,
        notes=f"Close {'above' if b['up'] else 'below'} confirmed {'resistance' if b['up'] else 'support'}",
        category="classic", meta={"level": b["level"]},
    ) for b in breakouts]
    return _cap(out)


@register("retest", "classic")
def detect_retest(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 12:
        return []
    _, retests = _breakout_retest(f)
    out = [make_concept(
        "retest", "bullish" if r["up"] else "bearish",
        f.index[r["breakout_index"]], f.index[r["index"]],
        r["level"], r["level"], 60,
        notes=f"Return to broken {'resistance' if r['up'] else 'support'} without full reversal",
        category="classic", meta={"level": r["level"]},
    ) for r in retests]
    return _cap(out)


@register("ema_cross", "classic")
def detect_ema_cross(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    out: list = []
    for fast, slow, weight in ((9, 20, 40), (20, 50, 55), (50, 200, 75)):
        if len(f) < slow + 2:
            continue
        ef = _arr(_ema(f["Close"], fast))
        es = _arr(_ema(f["Close"], slow))
        for i in range(1, len(f)):
            if not (np.isfinite(ef[i]) and np.isfinite(es[i])
                    and np.isfinite(ef[i - 1]) and np.isfinite(es[i - 1])):
                continue
            up = ef[i - 1] <= es[i - 1] and ef[i] > es[i]
            down = ef[i - 1] >= es[i - 1] and ef[i] < es[i]
            if up or down:
                out.append(make_concept(
                    "ema_cross", "bullish" if up else "bearish",
                    f.index[i], f.index[i], float(ef[i]), float(es[i]), weight,
                    notes=f"{'Golden' if up else 'Death'} cross EMA{fast}/EMA{slow}",
                    category="classic", meta={"pair": f"{fast}/{slow}"},
                ))
    out.sort(key=lambda c: c.end_time)
    return _cap(out)


@register("price_vs_ema", "classic")
def detect_price_vs_ema(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 2:
        return []
    price = latest_price(f)
    emas: dict[str, float] = {}
    for p in (20, 50, 200):
        v = _f(_ema(f["Close"], p).iloc[-1])
        if v is not None:
            emas[f"ema{p}"] = v
    vals = list(emas.values())
    above = sum(1 for v in vals if price > v)
    below = sum(1 for v in vals if price < v)
    direction = "bullish" if above > below else "bearish" if below > above else "neutral"
    desc = ", ".join(f"ema{p}={v:.2f}" for p, v in emas.items())
    return _state(f, "price_vs_ema", direction,
                  max(vals) if vals else None, min(vals) if vals else None,
                  45, f"Price {price:.2f} vs {desc}", {"emas": emas, "price": price})


def _rsi(f: pd.DataFrame, window: int = 14) -> pd.Series:
    return ta.momentum.rsi(f["Close"], window=window)


@register("rsi_signal", "classic")
def detect_rsi_signal(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 16:
        return []
    val = _f(_rsi(f, 14).iloc[-1])
    if val is None:
        return []
    prev = _f(_rsi(f, 14).iloc[-2])
    slope = val - prev if prev is not None else 0.0
    if val > 70:
        zone, direction = "overbought", "bearish"
    elif val < 30:
        zone, direction = "oversold", "bullish"
    else:
        zone, direction = "neutral", "neutral"
    return _state(f, "rsi_signal", direction, val,
                  70.0 if val > 70 else 30.0, 50,
                  f"RSI(14) {val:.1f} {zone}; slope {slope:+.2f}",
                  {"rsi": round(val, 2), "slope": round(slope, 3), "zone": zone})


def _pivot_divergence(f: pd.DataFrame, indicator: pd.Series, name: str) -> list:
    pivots = swing_points(f, lookback=2, confirm=2)
    vals = _arr(indicator)
    out: list = []
    for kind in ("low", "high"):
        raw = [p for p in pivots if p["kind"] == kind]
        pts: list[dict] = []
        for p in raw:
            if not pts or p["index"] - pts[-1]["index"] >= 5:
                pts.append(p)
        for a, b in zip(pts, pts[1:]):
            if b["confirm_index"] <= a["confirm_index"]:
                continue
            ia = _f(vals[a["index"]])
            ib = _f(vals[b["index"]])
            if ia is None or ib is None:
                continue
            if kind == "low" and float(b["price"]) < float(a["price"]) and ib > ia:
                direction, label = "bullish", "price lower low / indicator higher low"
            elif kind == "high" and float(b["price"]) > float(a["price"]) and ib < ia:
                direction, label = "bearish", "price higher high / indicator lower high"
            else:
                continue
            out.append(make_concept(
                name, direction, a["confirm_time"], b["confirm_time"],
                max(float(a["price"]), float(b["price"])),
                min(float(a["price"]), float(b["price"])), 65,
                notes=f"{name} {direction} divergence", category="classic",
                meta={"pattern": label, "indicator_at_first": ia,
                      "indicator_at_second": ib, "kind": kind},
            ))
    return _cap(out)


@register("rsi_divergence", "classic")
def detect_rsi_divergence(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 20:
        return []
    return _pivot_divergence(f, _rsi(f, 14), "rsi_divergence")


def _macd(f: pd.DataFrame) -> ta.trend.MACD:
    return ta.trend.MACD(f["Close"], window_slow=26, window_fast=12, window_sign=9)


@register("macd_cross", "classic")
def detect_macd_cross(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 30:
        return []
    m = _macd(f)
    line = _arr(m.macd())
    sig = _arr(m.macd_signal())
    out: list = []
    for i in range(1, len(f)):
        if not (np.isfinite(line[i]) and np.isfinite(sig[i])
                and np.isfinite(line[i - 1]) and np.isfinite(sig[i - 1])):
            continue
        up = line[i - 1] <= sig[i - 1] and line[i] > sig[i]
        down = line[i - 1] >= sig[i - 1] and line[i] < sig[i]
        if up or down:
            out.append(make_concept(
                "macd_cross", "bullish" if up else "bearish",
                f.index[i], f.index[i], float(line[i]), float(sig[i]), 55,
                notes=f"MACD {'crosses above' if up else 'crosses below'} signal",
                category="classic", meta={"pair": "macd/signal"},
            ))
    return _cap(out)


@register("macd_divergence", "classic")
def detect_macd_divergence(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 30:
        return []
    return _pivot_divergence(f, _macd(f).macd(), "macd_divergence")


@register("stochastic", "classic")
def detect_stochastic(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 20:
        return []
    so = ta.momentum.StochasticOscillator(f["High"], f["Low"], f["Close"],
                                          window=14, smooth_window=3)
    k = _arr(so.stoch())
    d = _arr(so.stoch_signal())
    out: list = []
    for i in range(1, len(f)):
        if not (np.isfinite(k[i]) and np.isfinite(d[i])
                and np.isfinite(k[i - 1]) and np.isfinite(d[i - 1])):
            continue
        up = k[i - 1] <= d[i - 1] and k[i] > d[i]
        down = k[i - 1] >= d[i - 1] and k[i] < d[i]
        if not (up or down):
            continue
        zone = "overbought" if k[i] > 80 else "oversold" if k[i] < 20 else "mid"
        out.append(make_concept(
            "stochastic", "bullish" if up else "bearish",
            f.index[i], f.index[i], float(k[i]), float(d[i]), 50,
            notes=f"Stochastic %K {'crosses above' if up else 'crosses below'} %D ({zone})",
            category="classic", meta={"k": round(float(k[i]), 2),
                                      "d": round(float(d[i]), 2), "zone": zone},
        ))
    if not out:
        val = _f(k[-1])
        if val is not None:
            zone = "overbought" if val > 80 else "oversold" if val < 20 else "mid"
            return _state(f, "stochastic", "neutral", val, None, 40,
                          f"Stochastic %K {val:.1f} ({zone})", {"zone": zone})
    return _cap(out)


def _bb_width(f: pd.DataFrame) -> np.ndarray:
    bb = ta.volatility.BollingerBands(f["Close"], window=20, window_dev=2.0)
    hi = _arr(bb.bollinger_hband())
    lo = _arr(bb.bollinger_lband())
    mid = _arr(bb.bollinger_mavg())
    denom = np.where(np.isfinite(mid) & (mid != 0), mid, np.nan)
    return (hi - lo) / denom


@register("bollinger_squeeze", "classic")
def detect_bollinger_squeeze(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 40:
        return []
    w = _bb_width(f)
    out: list = []
    last = -100
    for i in range(20, len(f)):
        if not np.isfinite(w[i]):
            continue
        win = w[max(0, i - 9):i + 1]
        finite = win[np.isfinite(win)]
        long = w[max(0, i - 99):i + 1]
        long_finite = long[np.isfinite(long)]
        if len(finite) < 8 or len(long_finite) < 20:
            continue
        if (w[i] <= np.min(finite) + 1e-12
                and w[i] <= np.median(long_finite) and i - last >= 8):
            out.append(make_concept(
                "bollinger_squeeze", "neutral", f.index[i], f.index[i],
                float(w[i]), float(w[i]), 45,
                notes=f"Bollinger width {w[i]:.3f} at a local minimum (squeeze)",
                category="classic", meta={"width": round(float(w[i]), 4)},
            ))
            last = i
    return _cap(out)


@register("bollinger_expansion", "classic")
def detect_bollinger_expansion(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 40:
        return []
    w = _bb_width(f)
    out: list = []
    last = -100
    for i in range(21, len(f)):
        if not (np.isfinite(w[i]) and np.isfinite(w[i - 1])) or w[i - 1] <= 0:
            continue
        prior = w[max(0, i - 50):i]
        prior = prior[np.isfinite(prior)]
        if not len(prior):
            continue
        if w[i] > 1.2 * w[i - 1] and w[i] > 1.15 * np.median(prior) and i - last >= 8:
            out.append(make_concept(
                "bollinger_expansion", "neutral", f.index[i], f.index[i],
                float(w[i]), float(w[i - 1]), 50,
                notes=f"Bollinger width expands {w[i] / w[i - 1]:.2f}x after compression",
                category="classic",
                meta={"width": round(float(w[i]), 4),
                      "ratio": round(float(w[i] / w[i - 1]), 2)},
            ))
            last = i
    return _cap(out)


@register("atr_volatility", "classic")
def detect_atr_volatility(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 20:
        return []
    ap = _arr(atr(f, 14) / f["Close"] * 100.0)
    cur = _f(ap[-1])
    if cur is None or latest_price(f) <= 0:
        return []
    window = ap[max(0, len(ap) - 100):]
    window = window[np.isfinite(window)]
    regime = "normal"
    if len(window) >= 20:
        if cur <= np.percentile(window, 30):
            regime = "low"
        elif cur >= np.percentile(window, 70):
            regime = "high"
    a_now = latest_atr(f, 14)
    return _state(f, "atr_volatility", "neutral", a_now, a_now, 40,
                  f"ATR% {cur:.2f} — {regime} volatility regime",
                  {"atr_pct": round(cur, 3), "regime": regime})


@register("adx_trend_strength", "classic")
def detect_adx_trend_strength(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 30:
        return []
    ind = ta.trend.ADXIndicator(f["High"], f["Low"], f["Close"], window=14)
    adx = _f(ind.adx().iloc[-1])
    plus = _f(ind.adx_pos().iloc[-1])
    minus = _f(ind.adx_neg().iloc[-1])
    if adx is None:
        return []
    if adx > 25:
        regime = "trending"
        direction = "bullish" if (plus or 0) > (minus or 0) else "bearish"
        notes = f"ADX {adx:.1f} > 25: strong trend (+DI {plus:.1f} / -DI {minus:.1f})"
    elif adx < 20:
        regime, direction = "ranging", "neutral"
        notes = f"ADX {adx:.1f} < 20: ranging market"
    else:
        regime, direction = "transitional", "neutral"
        notes = f"ADX {adx:.1f}: weak/transitional trend"
    return _state(f, "adx_trend_strength", direction, adx, None, 50, notes,
                  {"adx": round(adx, 2), "di_plus": plus, "di_minus": minus,
                   "regime": regime})


@register("fibonacci", "classic")
def detect_fibonacci(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 12:
        return []
    pivots = swing_points(f, lookback=2, confirm=2)
    highs = [p for p in pivots if p["kind"] == "high"]
    lows = [p for p in pivots if p["kind"] == "low"]
    if not highs or not lows:
        return []
    hi, lo = highs[-1], lows[-1]
    if hi["index"] == lo["index"]:
        return []
    swing_high = max(float(hi["price"]), float(lo["price"]))
    swing_low = min(float(hi["price"]), float(lo["price"]))
    span = swing_high - swing_low
    if span <= 0:
        return []
    upswing = lo["index"] > hi["index"]
    levels = {}
    for r in (0.382, 0.5, 0.618):
        levels[str(r)] = (swing_high - r * span) if upswing else (swing_low + r * span)
    end_t = lo["confirm_time"] if upswing else hi["confirm_time"]
    start_t = hi["confirm_time"] if upswing else lo["confirm_time"]
    return [make_concept(
        "fibonacci", "bullish" if upswing else "bearish",
        start_t, end_t, max(levels.values()), min(levels.values()), 50,
        notes=f"Fib retracement of last {'up' if upswing else 'down'} swing",
        category="classic",
        meta={"levels": {k: round(v, 4) for k, v in levels.items()},
              "swing_high": swing_high, "swing_low": swing_low},
    )]


def _pivot_sets(h: float, l: float, c: float) -> dict[str, dict[str, float]]:
    p = (h + l + c) / 3.0
    rng = h - l
    classic = {
        "P": p, "R1": 2 * p - l, "S1": 2 * p - h,
        "R2": p + rng, "S2": p - rng,
        "R3": h + 2 * (p - l), "S3": l - 2 * (h - p),
    }
    fib = {
        "P": p,
        "R1": h + 0.382 * rng, "S1": l - 0.382 * rng,
        "R2": h + 0.5 * rng, "S2": l - 0.5 * rng,
        "R3": h + 0.618 * rng, "S3": l - 0.618 * rng,
    }
    cam = {
        "R1": c + 0.183 * rng, "S1": c - 0.183 * rng,
        "R2": c + 0.275 * rng, "S2": c - 0.275 * rng,
        "R3": c + 0.55 * rng, "S3": c - 0.55 * rng,
        "R4": c + 1.1 * rng, "S4": c - 1.1 * rng,
    }
    return {"classic": classic, "fibonacci": fib, "camarilla": cam}


@register("pivot_points", "classic")
def detect_pivot_points(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if not isinstance(f.index, pd.DatetimeIndex) or len(f) < 5:
        return []
    daily = f.resample("1D").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["High", "Low", "Close"])
    if len(daily) < 2:
        return []
    completed = daily.iloc[:-1]
    last = completed.iloc[-1]
    ts = completed.index[-1]
    h, l, c = float(last["High"]), float(last["Low"]), float(last["Close"])
    price = latest_price(f)
    out: list = []
    for name, levels in _pivot_sets(h, l, c).items():
        supports = [v for v in levels.values() if v < price]
        resistances = [v for v in levels.values() if v > price]
        level = max(supports) if supports else (min(resistances) if resistances else None)
        if level is None:
            continue
        out.append(make_concept(
            "pivot_points", "bullish" if level < price else "bearish",
            ts, ts, level, level, 40,
            notes=f"Nearest {name} pivot at {level:.2f} from last completed daily bar",
            category="classic",
            meta={"set": name, "levels": {k: round(v, 4) for k, v in levels.items()}},
        ))
    return _cap(out)


@register("vwap", "classic")
def detect_vwap(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 5 or float(f["Volume"].sum()) <= 0:
        return []
    tp = (f["High"] + f["Low"] + f["Close"]) / 3.0
    vol = f["Volume"].fillna(0.0)
    if isinstance(f.index, pd.DatetimeIndex):
        day = f.index.normalize()
        session_start = f.index[day == day[-1]][0]
        session = str(day[-1].date())
    else:
        day = pd.Series(0, index=f.index)
        session_start = f.index[0]
        session = "single"
    vv = vol.groupby(day).cumsum().replace(0, np.nan)
    val = _f(((tp * vol).groupby(day).cumsum() / vv).iloc[-1])
    if val is None:
        return []
    price = latest_price(f)
    direction = "bullish" if price > val else "bearish" if price < val else "neutral"
    return [make_concept(
        "vwap", direction, session_start, f.index[-1], val, val, 50,
        notes=f"Session VWAP {val:.2f} ({session})",
        category="classic", meta={"vwap": round(val, 4), "session": session},
    )]


@register("anchored_vwap", "classic")
def detect_anchored_vwap(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 10 or float(f["Volume"].sum()) <= 0:
        return []
    pivots = [p for p in swing_points(f, lookback=2, confirm=2)
              if p["confirm_index"] < len(f) - 1]
    if not pivots:
        return []
    anchor = pivots[-1]
    start = int(anchor["index"])
    tp = (f["High"] + f["Low"] + f["Close"]) / 3.0
    vol = f["Volume"].fillna(0.0)
    pv = (tp.iloc[start:] * vol.iloc[start:]).cumsum()
    vv = vol.iloc[start:].cumsum().replace(0, np.nan)
    val = _f((pv / vv).iloc[-1])
    if val is None:
        return []
    price = latest_price(f)
    up_anchor = anchor["kind"] == "low"
    direction = ("bullish" if up_anchor and price > val
                 else "bearish" if (not up_anchor) and price < val else "neutral")
    return [make_concept(
        "anchored_vwap", direction, anchor["pivot_time"], f.index[-1], val, val, 50,
        notes=f"VWAP anchored to latest confirmed swing {anchor['kind']}",
        category="classic",
        meta={"anchor_kind": anchor["kind"], "anchor_price": anchor["price"],
              "anchored_vwap": round(val, 4)},
    )]


@register("volume_profile", "classic")
def detect_volume_profile(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 5 or float(f["Volume"].fillna(0).sum()) <= 0:
        return []
    look = min(100, len(f))
    seg = f.iloc[-look:]
    tp = ((seg["High"] + seg["Low"] + seg["Close"]) / 3.0).to_numpy(dtype=float)
    vv = seg["Volume"].fillna(0.0).to_numpy(dtype=float)
    lo, hi = float(np.min(tp)), float(np.max(tp))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    bins = 40
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    hist = np.zeros(bins)
    for j, v in zip(idx, vv):
        hist[j] += v
    poc_i = int(np.argmax(hist))
    poc = float((edges[poc_i] + edges[poc_i + 1]) / 2.0)
    target = float(np.sum(hist)) * 0.7
    lo_i = hi_i = poc_i
    acc = float(hist[poc_i])
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        down = float(hist[lo_i - 1]) if lo_i > 0 else -1.0
        up = float(hist[hi_i + 1]) if hi_i < bins - 1 else -1.0
        if up >= down and hi_i < bins - 1:
            hi_i += 1
            acc += hist[hi_i]
        elif lo_i > 0:
            lo_i -= 1
            acc += hist[lo_i]
        else:
            break
    va_high = float(edges[hi_i + 1])
    va_low = float(edges[lo_i])
    end = f.index[-1]
    out = [make_concept(
        "volume_profile", "neutral", f.index[-look], end,
        va_high, va_low, 45,
        notes=f"Volume profile POC {poc:.2f}, 70% value area {va_low:.2f}-{va_high:.2f}",
        category="classic",
        meta={"poc": round(poc, 4), "value_area": [round(va_low, 4), round(va_high, 4)]},
    )]
    return _cap(out)


@register("ichimoku", "classic")
def detect_ichimoku(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 60:
        return []
    ind = ta.trend.IchimokuIndicator(f["High"], f["Low"],
                                     window1=9, window2=26, window3=52)
    tenkan = _arr(ind.ichimoku_conversion_line())
    kijun = _arr(ind.ichimoku_base_line())
    span_a = (tenkan + kijun) / 2.0
    high = _arr(f["High"])
    low = _arr(f["Low"])
    span_b = (pd.Series(high).rolling(52).max().to_numpy(dtype=float)
              + pd.Series(low).rolling(52).min().to_numpy(dtype=float)) / 2.0
    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)
    close = _arr(f["Close"])
    out: list = []
    for i in range(1, len(f)):
        if not (np.isfinite(cloud_top[i]) and np.isfinite(cloud_bottom[i])
                and np.isfinite(cloud_top[i - 1]) and np.isfinite(cloud_bottom[i - 1])):
            continue
        up = close[i - 1] <= cloud_top[i - 1] and close[i] > cloud_top[i]
        down = close[i - 1] >= cloud_bottom[i - 1] and close[i] < cloud_bottom[i]
        if up or down:
            out.append(make_concept(
                "ichimoku", "bullish" if up else "bearish",
                f.index[i], f.index[i], float(cloud_top[i]), float(cloud_bottom[i]), 55,
                notes=f"Price {'above' if up else 'below'} Ichimoku cloud (cross)",
                category="classic", meta={"tenkan": _f(tenkan[i]),
                                          "kijun": _f(kijun[i]), "event": "cloud_cross"},
            ))
    if not out:
        top = _f(cloud_top[-1])
        bottom = _f(cloud_bottom[-1])
        if top is not None and bottom is not None:
            price = latest_price(f)
            direction = "bullish" if price > top else "bearish" if price < bottom else "neutral"
            out.append(make_concept(
                "ichimoku", direction, f.index[-1], f.index[-1], top, bottom, 45,
                notes="Price relative to Ichimoku cloud", category="classic",
                meta={"tenkan": _f(tenkan[-1]), "kijun": _f(kijun[-1]),
                      "event": "state"},
            ))
    return _cap(out)


@register("market_regime", "classic")
def detect_market_regime(df: pd.DataFrame) -> list:
    f = ensure_frame(df)
    if len(f) < 50:
        return []
    ind = ta.trend.ADXIndicator(f["High"], f["Low"], f["Close"], window=14)
    adx = _f(ind.adx().iloc[-1])
    di_p = _f(ind.adx_pos().iloc[-1]) or 0.0
    di_m = _f(ind.adx_neg().iloc[-1]) or 0.0
    ap = _arr(atr(f, 14) / f["Close"] * 100.0)
    atrpct = _f(ap[-1])
    width = _bb_width(f)
    bw = _f(width[-1])
    if adx is None or atrpct is None or bw is None:
        return []
    trailing = ap[np.isfinite(ap)][-100:]
    hot_atr = atrpct >= np.percentile(trailing, 75) if len(trailing) >= 20 else False
    if hot_atr and bw > 0.12 and adx < 25:
        regime, direction = "volatile", "neutral"
    elif adx > 25:
        regime = "trending_up" if di_p > di_m else "trending_down"
        direction = "bullish" if di_p > di_m else "bearish"
    elif adx < 20:
        regime, direction = "ranging", "neutral"
    else:
        regime = "trending_up" if di_p > di_m else "trending_down"
        direction = "bullish" if di_p > di_m else "bearish"
    return _state(f, "market_regime", direction, atrpct, bw, 60,
                  f"Regime {regime} (ADX {adx:.1f}, ATR% {atrpct:.2f}, BB width {bw:.3f})",
                  {"regime": regime, "adx": round(adx, 2),
                   "atr_pct": round(atrpct, 3), "bb_width": round(bw, 4)})

