"""Macro concept detectors for gold (DXY, yields, risk, news and events).

Macro frames are trimmed to the latest bar of the core frame before use, so a
future print in a correlated series cannot be seen by the current bar.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.concepts.base import make_concept, register


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
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
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


def _move(frame: pd.DataFrame, bars: int = 5) -> tuple[float, int]:
    close = frame["Close"]
    if len(close) < 2:
        return 0.0, 0
    back = min(bars, len(close) - 1)
    first, last = float(close.iloc[-1 - back]), float(close.iloc[-1])
    if not np.isfinite(first) or first == 0:
        return 0.0, back
    return (last - first) / abs(first), back


def _direction(move: float, falling_is_bullish: bool = True, eps: float = 1e-5) -> str:
    if move > eps:
        return "bearish" if falling_is_bullish else "bullish"
    if move < -eps:
        return "bullish" if falling_is_bullish else "bearish"
    return "neutral"


def _strength(move: float, scale: float = 1500.0, floor: float = 20.0, cap: float = 95.0) -> float:
    return float(np.clip(floor + abs(move) * scale, floor, cap))


@register("dxy_correlation", "macro")
def detect_dxy_correlation(df: pd.DataFrame, dxy_df: pd.DataFrame | None = None) -> list[Any]:
    """Falling DXY is bullish gold; a rising DXY is bearish gold."""
    frame = _normalise(df)
    dxy = _truncate(dxy_df, _last_ts(frame))
    if frame is None or dxy is None or len(dxy) < 2:
        return []
    move, back = _move(dxy)
    direction = _direction(move)
    strength = _strength(move)
    trend = "rose" if move > 0 else "fell" if move < 0 else "was flat"
    notes = f"DXY {trend} {abs(move) * 100:.2f}% over {back} bars -> {direction} gold."
    meta = {
        "dxy_move_pct": round(move * 100, 4),
        "dxy_last": round(float(dxy["Close"].iloc[-1]), 4),
        "bars": back,
    }
    return [make_concept(
        "dxy_correlation", direction, dxy.index[max(0, len(dxy) - 1 - back)],
        _bar_time(dxy), None, None, strength, "active", notes,
        category="macro", meta=meta,
    )]


@register("us10y_correlation", "macro")
def detect_us10y_correlation(df: pd.DataFrame, us10y_df: pd.DataFrame | None = None) -> list[Any]:
    """Falling US 10Y yields are bullish gold; rising yields are bearish."""
    frame = _normalise(df)
    yields = _truncate(us10y_df, _last_ts(frame))
    if frame is None or yields is None or len(yields) < 2:
        return []
    move, back = _move(yields)
    direction = _direction(move)
    strength = _strength(move, scale=2000.0)
    trend = "rose" if move > 0 else "fell" if move < 0 else "was flat"
    notes = f"US10Y {trend} {abs(move) * 100:.2f}% over {back} bars -> {direction} gold."
    meta = {
        "us10y_move_pct": round(move * 100, 4),
        "us10y_last": round(float(yields["Close"].iloc[-1]), 4),
        "bars": back,
    }
    return [make_concept(
        "us10y_correlation", direction, yields.index[max(0, len(yields) - 1 - back)],
        _bar_time(yields), None, None, strength, "active", notes,
        category="macro", meta=meta,
    )]


@register("real_yield_proxy", "macro")
def detect_real_yield_proxy(df: pd.DataFrame, us10y_df: pd.DataFrame | None = None,
                            cpi_yoy: float | None = None) -> list[Any]:
    """Approximate real yield = nominal yield - CPI YoY; falling is bullish."""
    frame = _normalise(df)
    yields = _truncate(us10y_df, _last_ts(frame))
    if frame is None or yields is None or cpi_yoy is None:
        return []
    try:
        inflation = float(cpi_yoy)
    except (TypeError, ValueError):
        return []
    if not np.isfinite(inflation):
        return []
    nominal = float(yields["Close"].iloc[-1])
    real = nominal - inflation
    back = min(5, len(yields) - 1) if len(yields) > 1 else 0
    previous = float(yields["Close"].iloc[-1 - back]) if back else nominal
    delta = nominal - previous
    direction = _direction(delta, eps=0.01)
    strength = float(np.clip(25 + abs(delta) * 900 + (10 if real < 0 else 0), 25, 100))
    trend = "rose" if delta > 0 else "fell" if delta < 0 else "was flat"
    notes = (
        f"Real yield proxy {trend} {abs(delta):.3f}pp to {real:.3f} "
        f"(nominal {nominal:.3f} - CPI {inflation:.3f}) -> {direction} gold."
    )
    meta = {
        "nominal_yield": round(nominal, 4),
        "cpi_yoy": round(inflation, 4),
        "real_yield": round(real, 4),
        "real_yield_change": round(delta, 4),
    }
    return [make_concept(
        "real_yield_proxy", direction,
        yields.index[max(0, len(yields) - 1 - back)], _bar_time(yields),
        None, None, strength, "active", notes, category="macro", meta=meta,
    )]


@register("risk_regime", "macro")
def detect_risk_regime(df: pd.DataFrame, spx_df: pd.DataFrame | None = None) -> list[Any]:
    """Rising SPX is risk-on (mildly bearish gold); falling SPX is bullish."""
    frame = _normalise(df)
    spx = _truncate(spx_df, _last_ts(frame))
    if frame is None or spx is None or len(spx) < 2:
        return []
    move, back = _move(spx)
    direction = "bearish" if move > 0.005 else "bullish" if move < -0.005 else "neutral"
    strength = float(np.clip(15 + abs(move) * 1200, 15, 70))
    regime = "risk-on" if move > 0 else "risk-off" if move < 0 else "balanced"
    notes = f"SPX {regime}: {move * 100:.2f}% over {back} bars -> {direction} gold."
    meta = {"spx_move_pct": round(move * 100, 4), "regime": regime, "bars": back}
    return [make_concept(
        "risk_regime", direction, spx.index[max(0, len(spx) - 1 - back)],
        _bar_time(spx), None, None, strength, "active", notes,
        category="macro", meta=meta,
    )]


@register("news_sentiment", "macro")
def detect_news_sentiment(df: pd.DataFrame, news: dict[str, Any] | None = None) -> list[Any]:
    """Gold-specific news sentiment scored on a -1..1 scale."""
    frame = _normalise(df)
    if frame is None or not isinstance(news, dict):
        return []
    try:
        score = float(news.get("sentiment"))
    except (TypeError, ValueError):
        return []
    if not np.isfinite(score):
        return []
    score = float(np.clip(score, -1.0, 1.0))
    label = str(news.get("label", "news"))
    direction = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
    strength = float(np.clip(abs(score) * 100, 0, 100))
    notes = f"News sentiment '{label}' scored {score:+.2f} -> {direction} gold."
    meta = {"sentiment": round(score, 4), "label": label}
    return [make_concept(
        "news_sentiment", direction, _bar_time(frame), _bar_time(frame),
        None, None, strength, "active", notes, category="macro", meta=meta,
    )]


@register("event_risk", "macro")
def detect_event_risk(df: pd.DataFrame, events: list[dict[str, Any]] | None = None,
                      hours_before: float = 24.0) -> list[Any]:
    """Neutral veto concept for an imminent high-impact calendar event."""
    frame = _normalise(df)
    ts = _last_ts(frame)
    if frame is None or ts is None or not isinstance(events, (list, tuple)):
        return []
    try:
        window = float(hours_before)
    except (TypeError, ValueError):
        return []
    if window <= 0:
        return []
    now = pd.Timestamp(ts)
    best: tuple[float, pd.Timestamp, str] | None = None
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("impact", "")).lower() != "high":
            continue
        try:
            when = pd.Timestamp(event.get("time"))
            if pd.isna(when):
                continue
            delta_h = abs((when - now).total_seconds()) / 3600.0
        except (TypeError, ValueError):
            continue
        if delta_h <= window and (best is None or delta_h < best[0]):
            best = (delta_h, when, str(event.get("name", "event")))
    if best is None:
        return []
    delta_h, when, name = best
    strength = float(np.clip(100 - 60 * (delta_h / window), 25, 100))
    notes = (
        f"High-impact event '{name}' at {when.isoformat()}, {delta_h:.1f}h from "
        "the last bar; reduce positioning before the release."
    )
    meta = {
        "event_name": name,
        "event_time": when.isoformat(),
        "hours_away": round(delta_h, 4),
        "impact": "high",
    }
    return [make_concept(
        "event_risk", "neutral", now, when, None, None,
        strength, "active", notes, category="macro", meta=meta,
    )]


@register("macro_regime", "macro")
def detect_macro_regime(df: pd.DataFrame, dxy_df: pd.DataFrame | None = None,
                        us10y_df: pd.DataFrame | None = None,
                        spx_df: pd.DataFrame | None = None,
                        news: dict[str, Any] | None = None) -> list[Any]:
    """Aggregate DXY, US10Y, SPX and news into one gold-specific macro bias."""
    frame = _normalise(df)
    if frame is None:
        return []
    components: dict[str, dict[str, Any]] = {}
    for name, result in (
        ("dxy_correlation", detect_dxy_correlation(frame, dxy_df=dxy_df)),
        ("us10y_correlation", detect_us10y_correlation(frame, us10y_df=us10y_df)),
        ("risk_regime", detect_risk_regime(frame, spx_df=spx_df)),
        ("news_sentiment", detect_news_sentiment(frame, news=news)),
    ):
        if result:
            concept = result[0]
            components[name] = {
                "direction": concept.direction,
                "strength": round(concept.strength, 2),
            }
    if not components:
        return []
    score = 0.0
    for component in components.values():
        sign = 1 if component["direction"] == "bullish" else -1 if component["direction"] == "bearish" else 0
        score += sign * component["strength"] / 100.0
    average = score / len(components)
    direction = "bullish" if average > 0.1 else "bearish" if average < -0.1 else "neutral"
    strength = float(np.clip(20 + abs(average) * 90, 20, 100))
    notes = "Macro bias: " + ", ".join(
        f"{name} {component['direction']}" for name, component in components.items()
    )
    meta = {
        "components": components,
        "bias": direction,
        "score": round(average, 4),
    }
    return [make_concept(
        "macro_regime", direction, _bar_time(frame), _bar_time(frame),
        None, None, strength, "active", notes, category="macro", meta=meta,
    )]
