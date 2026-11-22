"""Market data endpoints: candles, overview, indicators."""
from __future__ import annotations

import bisect
import math
import logging
from datetime import datetime, timezone
from typing import Any
import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File

from app import config
from app.services import auth
from app.services import data, db, price_service
from app.services.indicators import add_all

router = APIRouter(prefix="/api/market", tags=["market"])
logger = logging.getLogger(__name__)


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


@router.get("/live")
def live() -> dict[str, Any]:
    """Current canonical XAU/USD tick snapshot for reconnect/resync fallback."""
    payload = price_service.status()
    payload["verification"] = price_service.verification()
    return payload


@router.get("/candle-check")
def candle_check(timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$")) -> dict[str, Any]:
    """Diagnostic view of the exact series used for one chart timeframe."""
    entry = price_service.get_series(timeframe)
    frame = entry.df.tail(50).copy()
    meta = entry.meta or {}
    source = entry.source or "none"
    side = "bid" if source == "mt5" else str(meta.get("price_side") or "mid")
    approximate = bool(meta.get("approximate", source not in {"mt5"}))
    status = "STALE" if entry.stale else ("APPROXIMATE" if approximate else "LIVE")

    try:
        local_zone = ZoneInfo(config.CANDLE_DISPLAY_TIMEZONE)
    except ZoneInfoNotFoundError:
        local_zone = ZoneInfo("UTC")
        local_name = "UTC"
    else:
        local_name = config.CANDLE_DISPLAY_TIMEZONE

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    import pandas as pd
    if not frame.empty:
        index = pd.DatetimeIndex(frame.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        local_index = index.tz_convert(local_zone)
        previous = None
        seconds = {"15m": 900, "1h": 3600, "4h": 14400, "1D": 86400}[timeframe]
        for position, (utc_ts, local_ts) in enumerate(zip(index, local_index)):
            row = frame.iloc[position]
            epoch = int(utc_ts.timestamp())
            if previous is not None:
                gap = epoch - previous
                previous_dt = datetime.fromtimestamp(previous, timezone.utc)
                saturday_closed = previous_dt.weekday() == 4 and gap > seconds * 48
                sunday_closed = previous_dt.weekday() == 5
                if gap > seconds * 1.5 and not saturday_closed and not sunday_closed:
                    missing.append(datetime.fromtimestamp(previous, timezone.utc).isoformat())
            previous = epoch
            rows.append({
                "utc": utc_ts.isoformat(),
                "local": local_ts.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0) or 0),
            })

    meta_payload = {
        "source": source,
        "broker": meta.get("broker"),
        "server": meta.get("server"),
        "symbol": meta.get("symbol") or meta.get("symbol_label") or "XAU/USD",
        "digits": meta.get("digits"),
        "timezone_offset_seconds": meta.get("timezone_offset_seconds"),
        "storage_timezone": "UTC",
        "display_timezone": local_name,
        "higher_timeframe_build": meta.get("higher_timeframe_build")
        or ("broker-provided; no local aggregation" if source == "mt5" else "provider-provided/resampled"),
        "price_side": side,
        "status": status,
    }
    logger.info(
        "candle-check source=%s timeframe=%s storage_timezone=UTC display_timezone=%s "
        "price_side=%s higher_timeframe_build=%s missing_gap_reports=%d",
        source, timeframe, local_name, side,
        meta_payload["higher_timeframe_build"], len(missing),
    )
    return {
        "timeframe": timeframe,
        "meta": meta_payload,
        "missing_bars": missing[:20],
        "candles": rows,
        "disclaimer": "Educational analysis only. Not financial advice.",
    }


@router.post("/compare-tradingview")
async def compare_tradingview(
    file: UploadFile = File(...),
    timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$"),
) -> dict[str, Any]:
    """Compare a user-exported TradingView CSV against Meas Pro's active series."""
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "CSV is too large")
    try:
        from app.services.tradingview_compare import (
            align_and_compare,
            detect_timeframe,
            parse_tradingview_csv,
        )
        reference = parse_tradingview_csv(content)
    except Exception as exc:  # noqa: BLE001 - CSV parse errors are user errors
        raise HTTPException(422, f"Could not parse TradingView CSV: {exc}") from exc

    entry = price_service.get_series(timeframe)
    if entry.df is None or entry.df.empty:
        raise HTTPException(503, entry.error or "No app candles available")
    app_frame = entry.df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close"
    })
    result = align_and_compare(reference, app_frame)
    return {
        "timeframe": timeframe,
        "detected_reference_timeframe": detect_timeframe(
            pd.DatetimeIndex(reference.index)
        ),
        "source": entry.source,
        "source_status": "STALE" if entry.stale else (
            "APPROXIMATE" if bool((entry.meta or {}).get("approximate", False)) else "LIVE"
        ),
        "price_side": (entry.meta or {}).get("price_side", "bid"),
        **result,
    }


@router.get("/candles")
def candles(timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$"),
            ticker: str | None = None,
            force: bool = False) -> dict[str, Any]:
    """OHLCV candles with all indicator columns precomputed for charting."""
    # XAU/USD is the only user-selectable market; the futures symbol is only
    # used internally as a data-source fallback.
    tk = config.GOLD_TICKER
    entry = price_service.get_series(timeframe, force=force)
    if entry.df.empty:
        empty = {"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        return {
            "ticker": tk,
            "symbol": "XAU/USD (Gold)",
            "timeframe": timeframe,
            "stale": True,
            "synthetic": False,
            "source": entry.source,
            "source_status": "STALE",
            "price_side": "bid",
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
        "source": entry.source,
        "source_status": (
            "STALE" if entry.stale
            else ("APPROXIMATE" if bool((entry.meta or {}).get("approximate", False))
                  else "LIVE")
        ),
        "price_side": (entry.meta or {}).get("price_side", "bid"),
        "broker": (entry.meta or {}).get("series_broker"),
        "dataset_hash": (entry.meta or {}).get("dataset_hash"),
        "canonical": (entry.meta or {}).get("canonical", False),
        "source_boundaries": price_service.series_meta(timeframe)["boundaries"],
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
def refresh(user: dict | None = Depends(auth.optional_user)) -> dict[str, str]:
    """Force-refresh the market data cache."""
    if user and user.get("role") != "admin":
        raise HTTPException(403, "Administrator access required")
    data.clear_cache()
    return {"status": "cache cleared"}
