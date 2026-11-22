"""UTC candle aggregation and cleaning with DST-safe session anchors."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DAILY_ANCHOR = (17, 0)
NY_ZONE = ZoneInfo("America/New_York")


def utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    out.index = index
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def clean_ohlcv(
    frame: pd.DataFrame,
    *,
    digits: int = 2,
    atr_multiplier: float = 8.0,
) -> pd.DataFrame:
    """Remove duplicate, non-positive, zero-range and isolated spike bars."""
    out = utc_index(frame)
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column not in out:
            out[column] = 0.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Open", "High", "Low", "Close"]
    )
    positive = (
        (out["Open"] > 0)
        & (out["High"] > 0)
        & (out["Low"] > 0)
        & (out["Close"] > 0)
    )
    out = out[positive]
    out = out[
        (out["High"] >= out[["Open", "Low", "Close"]].max(axis=1))
        & (out["Low"] <= out[["Open", "High", "Close"]].min(axis=1))
    ]
    if len(out) >= 3:
        median_close = out["Close"].rolling(20, min_periods=3).median()
        atr = (out["High"] - out["Low"]).rolling(20, min_periods=3).median()
        deviation = (out["Close"] - median_close).abs()
        spikes = deviation > atr_multiplier * atr.fillna(np.inf)
        reverted = pd.Series(False, index=out.index)
        for i in range(1, len(out) - 1):
            if spikes.iloc[i]:
                left = float(out["Close"].iloc[i - 1])
                center = float(out["Close"].iloc[i])
                right = float(out["Close"].iloc[i + 1])
                reverted.iloc[i] = abs(right - left) < abs(center - left)
        out = out[~(spikes & reverted)]

    for column in ("Open", "High", "Low", "Close"):
        out[column] = out[column].round(digits)
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0).clip(lower=0)
    return out


def _ny_close_anchor(date: datetime) -> datetime:
    if date.tzinfo is not None:
        date = date.astimezone(NY_ZONE)
    local = datetime(
        date.year, date.month, date.day,
        DAILY_ANCHOR[0], DAILY_ANCHOR[1],
        tzinfo=NY_ZONE,
    )
    return local.astimezone(timezone.utc)


def daily_anchor_before(timestamp: datetime) -> datetime:
    ts = timestamp.astimezone(timezone.utc)
    ny = ts.astimezone(NY_ZONE)
    today = datetime(ny.year, ny.month, ny.day, tzinfo=NY_ZONE)
    candidate = _ny_close_anchor(today)
    if ts >= candidate:
        return candidate
    return _ny_close_anchor(today - timedelta(days=1))


def interval_anchor(
    timestamp: datetime,
    interval_seconds: int,
    *,
    broker_offset_seconds: int = 0,
) -> datetime:
    ts = timestamp.astimezone(timezone.utc)
    ny = ts.astimezone(NY_ZONE)
    today = datetime(ny.year, ny.month, ny.day, tzinfo=NY_ZONE)
    anchor = _ny_close_anchor(today) + timedelta(seconds=broker_offset_seconds)
    if ts < anchor:
        anchor = _ny_close_anchor(today - timedelta(days=1)) + timedelta(seconds=broker_offset_seconds)
    elapsed = int((ts - anchor).total_seconds())
    bucket = elapsed // interval_seconds
    return anchor + timedelta(seconds=bucket * interval_seconds)


def aggregate_ohlcv(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    digits: int = 2,
    broker_offset_seconds: int = 0,
) -> pd.DataFrame:
    """Aggregate cleaned UTC bars without filling gaps or weekend candles."""
    out = clean_ohlcv(frame, digits=digits)
    if out.empty:
        return out
    interval_seconds = {
        "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400,
    }.get(timeframe)
    if timeframe == "1D":
        labels = [daily_anchor_before(ts) for ts in out.index]
    elif interval_seconds is not None:
        labels = [
            interval_anchor(ts.to_pydatetime(), interval_seconds,
                            broker_offset_seconds=broker_offset_seconds)
            for ts in out.index
        ]
    else:
        raise ValueError(f"unsupported timeframe: {timeframe}")

    out = out.copy()
    out["__bucket"] = pd.to_datetime(labels, utc=True)
    result = out.groupby("__bucket", sort=True).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })
    result.index.name = "Date"
    return clean_ohlcv(result, digits=digits)
