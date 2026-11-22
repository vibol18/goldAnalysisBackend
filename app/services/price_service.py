"""Canonical XAUUSD price series.

Every consumer (chart, header, detectors, signals, outcomes, backtests, alerts,
Telegram, chat) must read candles and ticks through this module. It wraps the
provider chain once, stamps source/broker metadata and a dataset hash, and
records a boundary whenever the active source changes so series are never mixed.
"""
from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app import config
from app.services import data, prices

_LOCK = threading.RLock()
_SERIES: dict[str, dict[str, Any]] = {}
_BOUNDARIES: list[dict[str, Any]] = []


def _key(timeframe: str) -> str:
    return f"{config.GOLD_TICKER}|{timeframe}"


def hash_series(frame: Any) -> str:
    """Stable fingerprint of the exact OHLCV data used for a computation."""
    try:
        df = getattr(frame, "df", frame)
        columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df]
        values = np.ascontiguousarray(df[columns].to_numpy(dtype=np.float64))
        digest = hashlib.sha1()
        digest.update(str(df.shape).encode())
        digest.update(str(list(df.columns)).encode())
        digest.update(str(df.index[0]).encode() if len(df) else b"")
        digest.update(str(df.index[-1]).encode() if len(df) else b"")
        digest.update(values.tobytes())
        return digest.hexdigest()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _broker(source: str | None) -> str | None:
    if source != "mt5":
        return None
    try:
        from app.services import bridge
        return str(bridge.status().get("broker") or config.MT5_BROKER_NAME or "MT5")
    except Exception:  # noqa: BLE001
        return config.MT5_BROKER_NAME or "MT5"


def get_series(timeframe: str = "1h", force: bool = False) -> Any:
    """Return the single canonical CacheEntry for a timeframe."""
    entry = (data.get_ohlcv(config.GOLD_TICKER, timeframe, force=True)
             if force else data.get_ohlcv(config.GOLD_TICKER, timeframe))
    source = str(getattr(entry, "source", None) or "unknown")
    broker = _broker(source)
    digest = hash_series(entry.df)
    key = _key(timeframe)
    with _LOCK:
        previous = _SERIES.get(key)
        if previous and previous["source"] != source:
            _BOUNDARIES.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "timeframe": timeframe,
                "from": previous["source"],
                "to": source,
                "from_hash": previous["hash"],
                "to_hash": digest,
            })
            del _BOUNDARIES[:-50]
        _SERIES[key] = {
            "source": source, "broker": broker, "hash": digest,
            "at": time.time(), "rows": len(entry.df),
        }
        entry.meta = {
            **(getattr(entry, "meta", None) or {}),
            "series_source": source, "series_broker": broker,
            "dataset_hash": digest, "canonical": True,
        }
    return entry


def get_tick() -> dict[str, Any]:
    """Return the single canonical latest tick."""
    return prices.FEED.snapshot()


def series_meta(timeframe: str = "1h") -> dict[str, Any]:
    with _LOCK:
        meta = dict(_SERIES.get(_key(timeframe)) or {})
        boundaries = [b for b in _BOUNDARIES if b["timeframe"] == timeframe]
    return {
        **meta,
        "symbol": "XAU/USD",
        "ticker": config.GOLD_TICKER,
        "broker": meta.get("broker") or config.MT5_BROKER_NAME or None,
        "boundaries": boundaries[-10:],
    }


def status() -> dict[str, Any]:
    """Header-level combined feed and source status."""
    tick = get_tick()
    tick.update({
        "symbol": "XAU/USD",
        "broker": _broker(tick.get("source")),
        "source_priority": list(config.DATA_SOURCE_PRIORITY),
    })
    return tick


def diagnostics() -> dict[str, Any]:
    """Explain exactly which source is active and why the others are not."""
    snapshot = prices.FEED.snapshot()
    ticks = prices.FEED.all_ticks()
    reasons: list[str] = []
    if not config.BRIDGE_API_KEY:
        reasons.append("BRIDGE_API_KEY is empty - MT5 bridge cannot authenticate")
    if "mt5" not in config.DATA_SOURCE_PRIORITY:
        reasons.append("DATA_SOURCE_PRIORITY does not include 'mt5'")
    try:
        from app.services import db
        meta = db.bridge_get_meta()
        if not meta:
            reasons.append("No broker metadata received - MT5 bridge has never connected")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"Bridge metadata unavailable: {exc}")
    if "mt5" not in ticks:
        reasons.append("No MT5 tick has ever been posted to /api/bridge/tick")

    active_price = snapshot.get("price")
    side_by_side: dict[str, Any] = {}
    for source in dict.fromkeys(list(config.DATA_SOURCE_PRIORITY) + list(ticks)):
        tick = ticks.get(source)
        delta = None
        if tick and active_price is not None:
            delta = round(float(tick["bid"]) - float(active_price), 4)
        side_by_side[source] = {
            "tick": tick,
            "difference_vs_active_usd": delta,
            "configured": source in config.DATA_SOURCE_PRIORITY,
        }
    return {
        **snapshot,
        "symbol": "XAU/USD",
        "broker": _broker(snapshot.get("source")),
        "cached": bool(snapshot.get("time")),
        "cache_age_limit_seconds": 5,
        "source_priority": list(config.DATA_SOURCE_PRIORITY),
        "configured_sources": side_by_side,
        "why_bridge_not_used": reasons,
        "series": {key: value for key, value in _SERIES.items()},
    }


def verification() -> dict[str, Any]:
    """Cross-source truth check used by the banner and signal gating."""
    diagnostics_payload = diagnostics()
    deltas = [v["difference_vs_active_usd"]
              for v in diagnostics_payload["configured_sources"].values()
              if v["difference_vs_active_usd"] is not None]
    max_diff = max((abs(float(d)) for d in deltas), default=0.0)
    tolerance = float(config.PRICE_DIVERGENCE_TOLERANCE_USD)
    unverified = (max_diff > tolerance or
                  diagnostics_payload["state"] in (prices.STALE, prices.CLOSED))
    return {
        "state": "UNVERIFIED" if unverified else "VERIFIED",
        "max_difference_usd": round(max_diff, 4),
        "tolerance_usd": tolerance,
        "reason": (
            "cross-source divergence above tolerance" if max_diff > tolerance else
            diagnostics_payload.get("reason") or ""
        ),
    }


def reset() -> None:
    with _LOCK:
        _SERIES.clear()
        _BOUNDARIES.clear()
