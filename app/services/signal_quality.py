"""Extended-move / signal-quality filter.

A directional setup is only tradeable when price is not already stretched in
the *opposite* direction.  A SELL taken far below EMA/VWAP, with oversold RSI or
sitting on higher-timeframe support, is usually late: risk/reward is poor and
mean reversion is more likely than continuation.  BUYs near resistance are the
mirror image.  A directional call also requires higher-timeframe agreement.

The module is deliberately causal: it only reads bars up to the last one in
``df`` and concepts whose ``end_time`` is at or before that bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EXTENDED_MOVE_ATR = 2.0
RSI_OVERSOLD = 32.0
RSI_OVERBOUGHT = 75.0
NEAR_ZONE_ATR = 0.75
MAX_ZONE_AGE_BARS = 72

SUPPORT_TYPES = {
    "htf_level", "sr_zone", "supply_demand", "order_block",
    "fvg", "ifvg", "breaker_block", "mitigation_block", "ote",
}
RESISTANCE_TYPES = SUPPORT_TYPES


@dataclass
class QualityCheck:
    """Result of the extended-move screen for one candidate action."""

    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reasons": list(self.reasons),
            "metrics": {k: round(v, 3) for k, v in self.metrics.items()},
        }


def _atr(df: pd.DataFrame) -> float:
    try:
        close = df["Close"]
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - close.shift(1)).abs(),
            (df["Low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        value = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        if np.isfinite(value) and value > 0:
            return value
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return max(abs(float(df["Close"].iloc[-1])) * 0.001, 1e-9)


def _rsi(df: pd.DataFrame, period: int = 14) -> float | None:
    try:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        if not np.isfinite(gain) or not np.isfinite(loss):
            return None
        if loss <= 0:
            return 100.0 if gain > 0 else 50.0
        return float(100.0 - 100.0 / (1.0 + gain / loss))
    except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError):
        return None


def _vwap(df: pd.DataFrame, period: int = 20) -> float | None:
    try:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
        volume = df["Volume"].astype(float).clip(lower=0.0)
        window = min(period, len(df))
        pv = (typical * volume).rolling(window).sum().iloc[-1]
        vol = volume.rolling(window).sum().iloc[-1]
        if np.isfinite(pv) and np.isfinite(vol) and vol > 0:
            return float(pv / vol)
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return None


def _bar_age(concept: Any, index: pd.DatetimeIndex) -> int | None:
    try:
        ts = pd.Timestamp(concept.end_time)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    pos = int(index.searchsorted(ts, side="right"))
    return max(0, len(index) - pos)


def _zone_distance(concept: Any, price: float, atr_value: float) -> float | None:
    level = None
    meta = concept.meta if isinstance(concept.meta, dict) else {}
    if concept.type == "htf_level":
        raw = meta.get("level")
        if isinstance(raw, (int, float)):
            level = float(raw)
    if level is None and concept.top is not None and concept.bottom is not None:
        level = (float(concept.top) + float(concept.bottom)) / 2.0
    if level is None:
        for key in ("price", "level", "swept_price", "target"):
            raw = meta.get(key)
            if isinstance(raw, (int, float)):
                level = float(raw)
                break
    if level is None:
        return None
    return abs(level - price) / max(atr_value, 1e-9)


def _near_htf_zone(concepts: list[Any], index: pd.DatetimeIndex, price: float,
                   atr_value: float, want_support: bool, near_atr: float) -> tuple[bool, str | None]:
    wanted_types = SUPPORT_TYPES if want_support else RESISTANCE_TYPES
    wanted_direction = "bullish" if want_support else "bearish"
    wanted_kind = "support" if want_support else "resistance"
    for concept in reversed(concepts):
        if concept.type not in wanted_types or concept.status == "invalidated":
            continue
        if concept.direction != wanted_direction:
            continue
        meta = concept.meta if isinstance(concept.meta, dict) else {}
        if concept.type == "htf_level" and meta.get("kind") not in (wanted_kind, None):
            continue
        age = _bar_age(concept, index)
        if age is not None and age > MAX_ZONE_AGE_BARS:
            continue
        distance = _zone_distance(concept, price, atr_value)
        if distance is None or distance > near_atr:
            continue
        label = meta.get("timeframe") or concept.type.replace("_", " ")
        return True, f"{label} {wanted_kind} about {distance:.2f} ATR away"
    return False, None


def check_extended_move(
    concepts: list[Any],
    df: pd.DataFrame,
    action: str,
    *,
    extended_atr: float = EXTENDED_MOVE_ATR,
    rsi_oversold: float = RSI_OVERSOLD,
    rsi_overbought: float = RSI_OVERBOUGHT,
    near_zone_atr: float = NEAR_ZONE_ATR,
) -> QualityCheck:
    """Screen a BUY/SELL candidate and report why it must be downgraded to HOLD."""
    check = QualityCheck()
    if action not in ("BUY", "SELL") or df is None or len(df) < 15:
        return check

    price = float(df["Close"].iloc[-1])
    atr_value = _atr(df)
    ema = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    vwap = _vwap(df)
    rsi = _rsi(df)
    check.metrics.update(price=price, atr=atr_value, ema20=ema, rsi=rsi if rsi is not None else 0.0)
    if vwap is not None:
        check.metrics["vwap"] = vwap

    ema_distance = (price - ema) / max(atr_value, 1e-9)
    vwap_distance = (price - vwap) / max(atr_value, 1e-9) if vwap is not None else 0.0
    check.metrics["ema_atr"] = ema_distance
    check.metrics["vwap_atr"] = vwap_distance

    if action == "SELL":
        if ema_distance <= -extended_atr:
            check.reasons.append(
                f"price is {abs(ema_distance):.1f} ATR below EMA20 — SELL is an extended move"
            )
        if vwap_distance <= -extended_atr:
            check.reasons.append(
                f"price is {abs(vwap_distance):.1f} ATR below VWAP — SELL is an extended move"
            )
        if rsi is not None and rsi <= rsi_oversold:
            check.reasons.append(f"RSI {rsi:.0f} is oversold — SELL into exhaustion")
        index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True).tz_localize(None))
        near, label = _near_htf_zone(concepts, index, price, atr_value, True, near_zone_atr)
        if near:
            check.reasons.append(f"price is at higher-timeframe support/demand ({label})")
    else:
        if ema_distance >= extended_atr:
            check.reasons.append(
                f"price is {ema_distance:.1f} ATR above EMA20 — BUY is an extended move"
            )
        if vwap_distance >= extended_atr:
            check.reasons.append(
                f"price is {vwap_distance:.1f} ATR above VWAP — BUY is an extended move"
            )
        if rsi is not None and rsi >= rsi_overbought:
            check.reasons.append(f"RSI {rsi:.0f} is overbought — BUY into exhaustion")
        index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True).tz_localize(None))
        near, label = _near_htf_zone(concepts, index, price, atr_value, False, near_zone_atr)
        if near:
            check.reasons.append(f"price is at higher-timeframe resistance/supply ({label})")

    required = "bullish" if action == "BUY" else "bearish"
    htf = [c for c in concepts if c.type == "mtf_bias" and c.status != "invalidated"]
    if htf:
        latest = htf[-1]
        bias = str((latest.meta or {}).get("htf_bias") or latest.direction)
        if bias != required:
            check.reasons.append(
                f"higher-timeframe bias is {bias} — {action} requires HTF agreement"
            )
    else:
        check.metrics["htf_confirmation"] = 0.0

    check.blocked = bool(check.reasons)
    return check
