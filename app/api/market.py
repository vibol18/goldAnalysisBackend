"""Market data endpoints: candles, overview, indicators."""
from __future__ import annotations

import bisect
import math
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.services import data, db
from app.services.indicators import add_all

router = APIRouter(prefix="/api/market", tags=["market"])


def _signal_markers(candle_times: list[int], timeframe: str) -> list[dict[str, Any]]:
    """Deprecated: canonical markers come from /api/chart/annotations.

    Returning markers here duplicated arrows (and their text) on every candle.
    Kept as a function so older callers remain compatible.
    """
    return []
    signals = [
        s for s in db.list_signals(limit=500)
        if s.get("timeframe") == timeframe and s.get("action") in ("BUY", "SELL")
    ]
    markers: dict[int, dict[str, Any]] = {}
    for sig in signals:
        try:
            ts = datetime.fromisoformat(str(sig["created_at"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            stamp = int(ts.timestamp())
        except (TypeError, ValueError):
            continue
        index = bisect.bisect_right(candle_times, stamp) - 1
        if index < 0:
            continue
        action = sig["action"]
        buy = action == "BUY"
        markers[candle_times[index]] = {
            "time": candle_times[index],
            "position": "belowBar" if buy else "aboveBar",
            "color": "#26a69a" if buy else "#ef5350",
            "shape": "arrowUp" if buy else "arrowDown",
            "text": f"{action} {float(sig.get('confidence') or 0):.0f}%",
        }
    return [markers[t] for t in sorted(markers)]


@router.get("/overview")
def overview() -> dict[str, Any]:
    """Latest gold price, change, day range and correlated markets."""
    return data.get_market_overview()


@router.get("/candles")
def candles(timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$"),
            ticker: str | None = None,
            force: bool = False) -> dict[str, Any]:
    """OHLCV candles with all indicator columns precomputed for charting."""
    # XAU/USD is the only user-selectable market; the futures symbol is only
    # used internally as a data-source fallback.
    tk = config.GOLD_TICKER
    entry = data.get_ohlcv(tk, timeframe, force=force)
    if entry.df.empty:
        empty = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        return {
            "ticker": tk,
            "symbol": "XAU/USD (Gold)",
            "timeframe": timeframe,
            "stale": True,
            "synthetic": False,
            "error": entry.error or "no live XAU/USD data available",
            "candles": empty,
            "overlays": {},
            "panes": {},
            "levels": {"support": None, "resistance": None},
            "markers": [],
        }
    df = add_all(entry.df)
    # keep the payload bounded
    limit = {"15m": 400, "1h": 500, "4h": 500, "1D": 800}.get(timeframe, 500)
    df = df.tail(limit)

    def f(v: Any) -> float | None:
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    import pandas as pd
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    times = [int(ts.timestamp()) for ts in idx]
    candles_out = {
        "time": times,
        "open": [f(v) for v in df["Open"]],
        "high": [f(v) for v in df["High"]],
        "low": [f(v) for v in df["Low"]],
        "close": [f(v) for v in df["Close"]],
        "volume": [f(v) or 0 for v in df["Volume"]],
    }
    overlays = {
        "ema20": [f(v) for v in df.get("ema20")],
        "ema50": [f(v) for v in df.get("ema50")],
        "ema200": [f(v) for v in df.get("ema200")],
        "bb_high": [f(v) for v in df.get("bb_high")],
        "bb_mid": [f(v) for v in df.get("bb_mid")],
        "bb_low": [f(v) for v in df.get("bb_low")],
        "support": [f(v) for v in df.get("support")],
        "resistance": [f(v) for v in df.get("resistance")],
    }
    panes = {
        "rsi": [f(v) for v in df.get("rsi")],
        "macd": [f(v) for v in df.get("macd")],
        "macd_signal": [f(v) for v in df.get("macd_signal")],
        "macd_hist": [f(v) for v in df.get("macd_hist")],
        "atr": [f(v) for v in df.get("atr")],
    }
    last = df.iloc[-1]
    return {
        "ticker": tk,
        "symbol": "XAU/USD (Gold)",
        "timeframe": timeframe,
        "stale": entry.stale,
        "synthetic": entry.synthetic,
        "error": entry.error,
        "candles": candles_out,
        "overlays": overlays,
        "panes": panes,
        "levels": {
            "support": f(last.get("support")),
            "resistance": f(last.get("resistance")),
        },
        "markers": _signal_markers(times, timeframe),
    }


@router.post("/refresh")
def refresh() -> dict[str, str]:
    """Force-refresh the market data cache."""
    data.clear_cache()
    return {"status": "cache cleared"}
