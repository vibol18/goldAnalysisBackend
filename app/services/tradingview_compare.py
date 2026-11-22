"""Statistical comparison of locally uploaded TradingView CSV exports."""
from __future__ import annotations

import csv
import io
import statistics
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


def _parse_time(value: str) -> pd.Timestamp:
    text = str(value).strip()
    if not text:
        raise ValueError("empty candle time")
    if text.isdigit():
        number = int(text)
        if number > 10_000_000_000:
            number //= 1000
        return pd.Timestamp(datetime.fromtimestamp(number, timezone.utc))
    return pd.Timestamp(text)


def parse_tradingview_csv(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV contains no rows")
    aliases = {
        "time": ("time", "timestamp", "datetime", "date", "candle_time"),
        "open": ("open",),
        "high": ("high",),
        "low": ("low",),
        "close": ("close",),
    }
    headers = {str(key).strip().lower(): key for key in rows[0].keys() if key}

    def find(name: str) -> str:
        for alias in aliases[name]:
            if alias in headers:
                return headers[alias]
        raise ValueError(f"CSV missing required {name} column")

    time_col, open_col, high_col, low_col, close_col = (
        find("time"), find("open"), find("high"), find("low"), find("close")
    )
    records = []
    for row in rows:
        try:
            ts = _parse_time(str(row[time_col]))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            record = {
                "time": ts,
                "open": float(row[open_col]),
                "high": float(row[high_col]),
                "low": float(row[low_col]),
                "close": float(row[close_col]),
            }
            if all(value is not None for value in record.values()):
                records.append(record)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    frame = pd.DataFrame(records).set_index("time").sort_index()
    if frame.empty:
        raise ValueError("CSV has no valid OHLC rows")
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def detect_timeframe(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "unknown"
    diffs = pd.Series(index[1:].astype("int64") - index[:-1].astype("int64")) // 1_000_000_000
    seconds = int(diffs.median()) if len(diffs) else 0
    if seconds <= 120:
        return "1m"
    if seconds <= 360:
        return "5m"
    if seconds <= 1200:
        return "15m"
    if seconds <= 5400:
        return "1h"
    if seconds <= 21600:
        return "4h"
    if seconds <= 172800:
        return "1D"
    return f"{max(1, round(seconds / 86400))}D"


def align_and_compare(reference: pd.DataFrame, app: pd.DataFrame) -> dict:
    ref = reference.copy()
    app = app.copy()
    if not isinstance(ref.index, pd.DatetimeIndex):
        ref.index = pd.to_datetime(ref.index, utc=True)
    if not isinstance(app.index, pd.DatetimeIndex):
        app.index = pd.to_datetime(app.index, utc=True)
    ref.index = pd.DatetimeIndex(ref.index).tz_convert("UTC") if ref.index.tz is not None else pd.DatetimeIndex(ref.index).tz_localize("UTC")
    app.index = pd.DatetimeIndex(app.index).tz_convert("UTC") if app.index.tz is not None else pd.DatetimeIndex(app.index).tz_localize("UTC")

    joined = ref.join(app, how="inner", lsuffix="_ref", rsuffix="_app")
    joined = joined[
        (joined.filter(like="_ref").gt(0).all(axis=1))
        & (joined.filter(like="_app").gt(0).all(axis=1))
    ]
    if joined.empty:
        return {
            "matched": 0, "metrics": {}, "worst": [], "differences": [],
            "diagnosis": {"likely_cause": "No exact candle-open matches",
                          "suggestion": "Check the time zone, 17:00 New York anchor and timeframe."},
        }

    metrics = {}
    for field in ("open", "high", "low", "close"):
        differences = (joined[f"{field}_app"] - joined[f"{field}_ref"]).astype(float)
        absolute = differences.abs()
        metrics[field] = {
            "average": round(float(differences.mean()), 4),
            "median": round(float(differences.median()), 4),
            "maximum": round(float(absolute.max()), 4),
        }
    all_close = (joined["close_app"] - joined["close_ref"]).abs()
    differences = [
        {
            "time": ts.isoformat(),
            "close_diff": round(float(joined.loc[ts, "close_app"] - joined.loc[ts, "close_ref"]), 4),
        }
        for ts in joined.index
    ]

    worst = []
    for ts in joined.index:
        diffs = [abs(float(joined.loc[ts, f"{field}_app"] - joined.loc[ts, f"{field}_ref"]))
                 for field in ("open", "high", "low", "close")]
        worst.append({
            "time": ts.isoformat(),
            "open": round(float(joined.loc[ts, "open_app"]), 2),
            "high": round(float(joined.loc[ts, "high_app"]), 2),
            "low": round(float(joined.loc[ts, "low_app"]), 2),
            "close": round(float(joined.loc[ts, "close_app"]), 2),
            "reference_close": round(float(joined.loc[ts, "close_ref"]), 2),
            "max_difference": round(max(diffs), 4),
        })
    worst.sort(key=lambda row: row["max_difference"], reverse=True)

    medians = [abs(metrics[field]["median"]) for field in ("open", "high", "low", "close")]
    median_mad = statistics.median(medians)
    if median_mad > 0.50:
        diagnosis = {
            "likely_cause": "Constant or near-constant price basis difference",
            "suggestion": "If the difference is stable, enable a basis offset. Otherwise choose a closer MT5 or spot broker feed.",
        }
    elif max(medians) > 0.20:
        diagnosis = {
            "likely_cause": "Mixed price aggregation or data-source mismatch",
            "suggestion": "Check the active source, BID/MID/ASK setting and 4h/1D/1h aggregation boundaries.",
        }
    else:
        diagnosis = {
            "likely_cause": "Close agreement; residual differences are random",
            "suggestion": "Verify candle cleaning and source spread if differences increase at live candles.",
        }
    return {
        "matched": len(joined),
        "metrics": metrics,
        "within_20cents": round(float((all_close <= 0.20).mean() * 100), 2),
        "worst": worst[:20],
        "differences": differences,
        "diagnosis": diagnosis,
    }
