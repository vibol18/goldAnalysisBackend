"""Shared contract, registry and strictly causal helpers for concept detectors.

Every detector is a pure function ``detect_x(df, ...) -> list[Concept]``.
A concept visible at bar ``i`` may only use data at or before ``i``. Patterns
that need confirmation are stamped at their confirmation bar, so editing a
future candle can never change a past detection's geometry. Status is allowed
to evolve as later price mitigates a level — that is not lookahead.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

CATEGORIES = ("structure", "smc", "classic", "price_action", "mtf", "macro", "advanced")
REQUIRED_FIELDS = (
    "type", "direction", "start_time", "end_time", "top", "bottom",
    "strength", "status", "notes",
)

UNLIMITED = False


def capped(items: list, limit: int) -> list:
    """Keep the newest ``limit`` items unless unlimited detection is active.

    Unlimited mode is required by backtests and ML features: a cap applied to
    a full-history detection would drop older concepts when future bars add
    newer ones, which is a lookahead artefact.
    """
    return items if UNLIMITED else items[-limit:]


def _concept_modules() -> list[Any]:
    return [m for name, m in sys.modules.items()
            if name.startswith("app.concepts.") and hasattr(m, "UNLIMITED")]


@contextmanager
def unlimited_mode():
    """Enable unlimited detection in base *and* every already-imported family.

    Family modules import ``UNLIMITED`` by value, so their local binding must be
    patched too; otherwise their internal caps would silently drop older concepts
    as new bars arrive (a lookahead artefact).
    """
    global UNLIMITED
    previous = UNLIMITED
    previous_modules = {m: m.UNLIMITED for m in _concept_modules()}
    UNLIMITED = True
    for module in previous_modules:
        module.UNLIMITED = True
    try:
        yield
    finally:
        UNLIMITED = previous
        for module, value in previous_modules.items():
            module.UNLIMITED = value


@dataclass
class Concept:
    """One detected trading concept, serialisable for the API and the chart."""

    type: str
    direction: str
    start_time: Any
    end_time: Any
    top: float | None
    bottom: float | None
    strength: float
    status: str
    notes: str
    category: str = "classic"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.direction = self.direction if self.direction in ("bullish", "bearish", "neutral") else "neutral"
        self.status = self.status if self.status in ("active", "mitigated", "invalidated") else "active"
        self.strength = float(np.clip(float(self.strength), 0.0, 100.0))
        self.top = _finite(self.top)
        self.bottom = _finite(self.bottom)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "direction": self.direction,
            "start_time": ts_to_str(self.start_time),
            "end_time": ts_to_str(self.end_time),
            "top": self.top,
            "bottom": self.bottom,
            "strength": round(self.strength, 2),
            "status": self.status,
            "notes": self.notes,
            "category": self.category,
            "meta": json_safe(self.meta),
        }


def json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return ts_to_str(value)
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def ts_to_str(ts: Any) -> str | None:
    """Normalise a pandas/numpy/datetime timestamp to an ISO-8601 string."""
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
        if pd.isna(t):
            return None
        return t.isoformat()
    except (ValueError, TypeError):
        return str(ts)


def make_concept(type_: str, direction: str, start_time: Any, end_time: Any,
                 top: float | None, bottom: float | None, strength: float,
                 status: str = "active", notes: str = "",
                 category: str = "classic",
                 meta: dict[str, Any] | None = None) -> Concept:
    """Build a validated Concept with the minimum required fields."""
    return Concept(type_, direction, start_time, end_time, top, bottom,
                   strength, status, notes, category, meta or {})


def serialise(concepts: list[Concept]) -> list[dict[str, Any]]:
    return [c.to_dict() for c in concepts]


DETECTORS: dict[str, Callable[..., list[Concept]]] = {}
DETECTOR_CATEGORIES: dict[str, str] = {}


def register(name: str, category: str):
    """Decorator registering a detector under a stable concept type name."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown concept category: {category}")

    def deco(fn: Callable[..., list[Concept]]) -> Callable[..., list[Concept]]:
        DETECTORS[name] = fn
        DETECTOR_CATEGORIES[name] = category
        return fn
    return deco


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR, causal (uses only bars up to each row)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def body(df: pd.DataFrame) -> pd.Series:
    return (df["Close"] - df["Open"]).abs()


def signed_body(df: pd.DataFrame) -> pd.Series:
    return df["Close"] - df["Open"]


def upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df[["Open", "Close"]].max(axis=1)


def lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["Open", "Close"]].min(axis=1) - df["Low"]


def candle_range(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df["Low"]


def is_bull(df: pd.DataFrame) -> pd.Series:
    return df["Close"] > df["Open"]


def is_bear(df: pd.DataFrame) -> pd.Series:
    return df["Close"] < df["Open"]


def displacement_mask(df: pd.DataFrame, mult: float = 1.5,
                      wick_ratio: float = 0.5,
                      atr_s: pd.Series | None = None) -> pd.Series:
    """Large real body vs ATR with relatively small wicks."""
    a = atr_s if atr_s is not None else atr(df)
    rng = candle_range(df).replace(0, np.nan)
    return (
        (body(df) > mult * a)
        & (body(df) / rng >= 1.0 - wick_ratio)
        & a.notna()
    ).fillna(False)


def swing_points(df: pd.DataFrame, lookback: int = 2, confirm: int = 1,
                 kind: str = "both") -> list[dict[str, Any]]:
    """Causal fractal pivots.

    A pivot at bar ``i`` needs ``lookback`` bars either side; it is emitted only
    at bar ``i + confirm``. The pivot's own time is kept in ``meta['pivot_time']``
    and its confirmation time drives ``start_time`` for no-lookahead safety.
    """
    n = len(df)
    highs, lows = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    out: list[dict[str, Any]] = []
    for i in range(lookback, n - lookback):
        left_h, right_h = highs[i - lookback:i], highs[i + 1:i + lookback + 1]
        left_l, right_l = lows[i - lookback:i], lows[i + 1:i + lookback + 1]
        is_high = highs[i] >= left_h.max() and highs[i] >= right_h.max()
        is_low = lows[i] <= left_l.min() and lows[i] <= right_l.min()
        at = min(i + confirm, n - 1)
        if is_high and kind in ("both", "high"):
            out.append({
                "kind": "high", "index": i, "price": float(highs[i]),
                "pivot_time": df.index[i], "confirm_index": at,
                "confirm_time": df.index[at],
            })
        if is_low and kind in ("both", "low"):
            out.append({
                "kind": "low", "index": i, "price": float(lows[i]),
                "pivot_time": df.index[i], "confirm_index": at,
                "confirm_time": df.index[at],
            })
    out.sort(key=lambda p: p["confirm_index"])
    return out


def pivots_before(pivots: list[dict[str, Any]], i: int,
                  kind: str | None = None) -> dict[str, Any] | None:
    """Most recently confirmed pivot at or before bar ``i``."""
    matches = [p for p in pivots if p["confirm_index"] <= i
               and (kind is None or p["kind"] == kind)]
    return matches[-1] if matches else None


def last_of_type(concepts: list[Concept], type_: str) -> Concept | None:
    found = [c for c in concepts if c.type == type_]
    return found[-1] if found else None


def active_at(concepts: list[Concept], ts: Any) -> list[Concept]:
    """Concepts that had started by ``ts`` and were not yet invalidated."""
    target = pd.Timestamp(ts)
    out: list[Concept] = []
    for c in concepts:
        start = pd.Timestamp(c.start_time) if c.start_time is not None else None
        if start is not None and start > target:
            continue
        if c.status == "invalidated":
            continue
        out.append(c)
    return out


def zone_midpoint(top: float | None, bottom: float | None) -> float | None:
    if top is None or bottom is None:
        return None
    return (float(top) + float(bottom)) / 2.0


def distance_to_zone(price: float, top: float | None, bottom: float | None,
                     atr_value: float) -> float | None:
    """Signed ATR-normalised distance to the nearest edge of a zone."""
    if top is None or bottom is None or not np.isfinite(atr_value) or atr_value <= 0:
        return None
    mid = zone_midpoint(top, bottom)
    if mid is None:
        return None
    return (price - mid) / atr_value


def overlap(a_top: float | None, a_bottom: float | None,
            b_top: float | None, b_bottom: float | None) -> bool:
    if None in (a_top, a_bottom, b_top, b_bottom):
        return False
    return max(a_bottom, b_bottom) <= min(a_top, b_top)


def session_of(ts: Any) -> str:
    """UTC session label for a timestamp."""
    hour = pd.Timestamp(ts).hour
    if 0 <= hour < 7:
        return "asian"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "new_york"
    if 16 <= hour < 21:
        return "new_york_pm"
    return "late"


def in_killzone(ts: Any, killzones: dict[str, tuple[int, int]] | None = None) -> bool:
    """Check UTC (start_hour, end_hour) windows, end exclusive."""
    zones = killzones or {
        "asian": (23, 7), "london": (7, 12), "new_york": (12, 16),
    }
    t = pd.Timestamp(ts)
    minutes = t.hour * 60 + t.minute
    for start, end in zones.values():
        s, e = start * 60, end * 60
        if s <= e:
            if s <= minutes < e:
                return True
        elif minutes >= s or minutes < e:
            return True
    return False


def ensure_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate/normalise an OHLCV frame for detectors."""
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    out = df.copy()
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.RangeIndex(len(out))
    return out.sort_index()


def latest_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def latest_atr(df: pd.DataFrame, period: int = 14) -> float:
    value = atr(df, period).iloc[-1]
    return float(value) if value is not None and np.isfinite(value) else 0.0
