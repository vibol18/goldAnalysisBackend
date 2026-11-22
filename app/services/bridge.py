"""MT5 bridge data access: UTC normalisation, freshness and source labels.

The bridge stores *bid* candles from the user's MT5 terminal. Broker time is an
epoch value expressed in the broker's own timezone; subtracting the reported
offset gives real UTC. Nothing here ever mixes bridge candles with fallback
provider candles — each chart is built from exactly one source.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app import config
from app.services import data, db
from app.services.candle_aggregation import aggregate_ohlcv, clean_ohlcv

logger = logging.getLogger(__name__)

TF_ALIASES = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D1": "1D", "1D": "1D",
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h",
}
CANONICAL_TF = {"1m", "5m", "15m", "30m", "1h", "4h", "1D"}


def normalise_timeframe(value: str) -> str | None:
    """Map MT5 (M15/H1/H4/D1) and UI (15m/1h/4h/1D) names to one key."""
    return TF_ALIASES.get(str(value or "").strip().upper(), None) or (
        str(value).strip() if str(value).strip() in CANONICAL_TF else None
    )


def normalise_time(broker_time: int | float, offset_seconds: int | float) -> int:
    """Convert a broker-local epoch timestamp to a real UTC epoch timestamp."""
    return int(broker_time) - int(offset_seconds or 0)


def _age_seconds(stamp: Any) -> float | None:
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(str(stamp))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - when).total_seconds()
    except (ValueError, TypeError):
        return None


def symbol() -> str:
    meta = db.bridge_get_meta() or {}
    return str(meta.get("symbol") or config.MT5_SYMBOL)


def status() -> dict[str, Any]:
    """Current bridge status for the dashboard (no secrets included)."""
    meta = db.bridge_get_meta() or {}
    tick = db.bridge_get_latest_tick() or {}
    tick_age = _age_seconds(meta.get("last_tick_at"))
    candle_age = _age_seconds(meta.get("last_candle_at"))
    live = (tick_age is not None and tick_age <= config.BRIDGE_TICK_MAX_AGE_SECONDS)
    receiving = live or (candle_age is not None
                         and candle_age <= config.BRIDGE_CANDLE_MAX_AGE_SECONDS)
    broker = str(meta.get("broker") or config.MT5_BROKER_NAME or "MT5")
    sym = str(meta.get("symbol") or config.MT5_SYMBOL)
    if not meta:
        label = "MT5 bridge offline"
        message = "MT5 bridge offline — using fallback provider"
    elif live:
        label = f"MT5 ({broker} {sym}) LIVE"
        message = label
    elif receiving:
        label = f"MT5 ({broker} {sym}) — no fresh tick"
        message = label
    else:
        label = "MT5 bridge offline"
        message = "MT5 bridge offline — using fallback provider"
    return {
        "enabled": bool(config.BRIDGE_API_KEY),
        "online": bool(receiving),
        "live": bool(live),
        "broker": broker,
        "server": str(meta.get("server") or ""),
        "symbol": sym,
        "digits": int(meta.get("digits") or 2),
        "point": float(meta.get("point") or 0.01),
        "contract_size": float(meta.get("contract_size") or 100.0),
        "timezone_offset_seconds": int(meta.get("timezone_offset_seconds") or 0),
        "bid": tick.get("bid"),
        "ask": tick.get("ask"),
        "spread": tick.get("spread"),
        "tick_age_seconds": round(tick_age, 1) if tick_age is not None else None,
        "candle_age_seconds": round(candle_age, 1) if candle_age is not None else None,
        "last_tick_at": meta.get("last_tick_at"),
        "last_candle_at": meta.get("last_candle_at"),
        "label": label,
        "message": message,
    }


def is_fresh() -> bool:
    """True when a tick arrived within the configured freshness window."""
    st = status()
    return bool(st["live"])


def load_frame(timeframe: str) -> data.CacheEntry | None:
    """Read bridge candles for one timeframe as a UTC-indexed CacheEntry."""
    sym = symbol()
    tf = normalise_timeframe(timeframe) or "1h"
    rows = db.bridge_get_candles(sym, tf, limit=3000)
    logger.info(
        "candle-series source=mt5 symbol=%s broker=%s timeframe=%s "
        "broker_time_offset_seconds=%s timezone_store=UTC "
        "higher_timeframe_source=local-anchor-aggregation",
        sym,
        (db.bridge_get_meta(sym) or {}).get("broker") or config.MT5_BROKER_NAME,
        tf,
        int((db.bridge_get_meta(sym) or {}).get("timezone_offset_seconds") or 0),
    )
    meta = db.bridge_get_meta(sym) or {}
    base_tf = "1m"
    base_rows = db.bridge_get_candles(sym, base_tf, limit=10000)
    if len(base_rows) < 50:
        base_tf = "5m"
        base_rows = db.bridge_get_candles(sym, base_tf, limit=10000)
    if tf in {"1m", "5m"}:
        base_rows = db.bridge_get_candles(sym, tf, limit=10000)
    if len(base_rows) < 50 and tf not in {"1m", "5m"}:
        if len(rows) < 50:
            return None
        base_rows = rows
        base_tf = tf
    if len(base_rows) < 50:
        return None
    offset = int(meta.get("timezone_offset_seconds") or 0)
    frame = pd.DataFrame(base_rows)
    frame["time"] = [normalise_time(value, offset) for value in frame["time"]]
    frame["Date"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_localize(None)
    frame = frame.set_index("Date").sort_index()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }).dropna(subset=["Open", "High", "Low", "Close"])
    frame = clean_ohlcv(
        frame,
        digits=int(meta.get("digits") or 2),
        atr_multiplier=config.OUTLIER_ATR_MULTIPLIER,
    )
    if tf not in {"1m", "5m"}:
        frame = aggregate_ohlcv(
            frame,
            tf,
            digits=int(meta.get("digits") or 2),
            broker_offset_seconds=config.BROKER_SERVER_ANCHOR_OFFSET_SECONDS,
        )
    if frame.empty:
        return None
    st = status()
    return data.CacheEntry(
        df=frame[["Open", "High", "Low", "Close", "Volume"]],
        fetched_at=time.time(),
        stale=not st["live"],
        source="mt5",
        meta={
            **st,
            "source_kind": "mt5",
            "approximate": False,
            "price_side": "bid",
            "higher_timeframe_build": (
                f"cleaned UTC base={base_tf}; aggregated locally to {tf} "
                "with NY 17:00 DST-safe anchor"
            ),
            "symbol_label": "XAUUSD",
            "bid": st.get("bid"),
            "ask": st.get("ask"),
            "spread": st.get("spread"),
        },
    )


def source_label(entry: Any) -> str:
    """Human-readable source label for a CacheEntry (or None)."""
    source = getattr(entry, "source", None)
    meta = getattr(entry, "meta", None) or {}
    if source == "mt5":
        broker = str(meta.get("broker") or "MT5")
        sym = str(meta.get("symbol_label") or "XAUUSD")
        return f"MT5 ({broker} {sym}) " + ("LIVE" if not getattr(entry, "stale", False) else "STALE")
    names = {
        "twelvedata": "Twelve Data",
        "kraken-xaut": "Kraken XAUT/USD",
        "yahoo-chart": "Yahoo Finance",
        "yfinance": "Yahoo Finance",
        "gold-api": "Gold API",
    }
    if entry is not None and getattr(entry, "stale", False):
        return "STALE (last cached snapshot)"
    if source:
        base = names.get(str(source), str(source))
        return f"{base} — approximate (not broker prices)"
    return "STALE (no live source)"


def is_approximate(entry: Any) -> bool:
    meta = getattr(entry, "meta", None) or {}
    if getattr(entry, "source", None) == "mt5":
        return False
    return bool(meta.get("approximate", True))
