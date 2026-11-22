"""DST-safe forex-gold market hours.

Gold spot trades Sunday evening through Friday evening New York time with a
daily break around the New York close. All decisions are made from a single
``now`` value so tests are deterministic.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from zoneinfo import ZoneInfo

from app import config


def tz() -> ZoneInfo:
    try:
        return ZoneInfo(config.MARKET_TIMEZONE)
    except Exception:  # noqa: BLE001 - bad config must not crash the feed
        return ZoneInfo("America/New_York")


def local_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(tz())


def hour_float(value: datetime) -> float:
    return value.hour + value.minute / 60.0


def is_market_open(now: datetime | None = None) -> bool:
    value = local_now(now)
    local_time = hour_float(value)
    weekday = value.weekday()  # Mon=0 ... Sun=6
    close = float(config.MARKET_CLOSE_HOUR)
    open_hour = float(config.MARKET_OPEN_HOUR)
    if weekday == 6:  # Sunday
        return local_time >= open_hour
    if weekday == 4:  # Friday: closes at the New York close
        return local_time < close
    if weekday in (0, 1, 2, 3):  # Mon-Thu: daily break after the close
        return local_time < close or local_time >= close + 1.0
    return False  # Saturday


def seconds_to_reopen(now: datetime | None = None) -> float:
    """Seconds until the next session opens (0 while open)."""
    value = local_now(now)
    if is_market_open(value):
        return 0.0
    zone = tz()
    close = float(config.MARKET_CLOSE_HOUR)
    open_hour = float(config.MARKET_OPEN_HOUR)
    local_time = hour_float(value)
    # Handle the daily Mon-Thu break first (reopens one hour after the close).
    if value.weekday() in (0, 1, 2, 3) and close <= local_time < close + 1.0:
        reopen = value.replace(hour=int(close) + 1, minute=0, second=0, microsecond=0)
        return max(0.0, (reopen - value).total_seconds())
    # Otherwise find the next Sunday/Monday open.
    days_ahead = (6 - value.weekday()) % 7
    if days_ahead == 0 and local_time >= open_hour:
        days_ahead = 7
    reopen = (value.replace(hour=int(open_hour), minute=0, second=0, microsecond=0)
              + timedelta(days=days_ahead))
    return max(0.0, (reopen.astimezone(timezone.utc) - value.astimezone(timezone.utc)).total_seconds())


def status(now: datetime | None = None) -> dict[str, Any]:
    value = local_now(now)
    open_now = is_market_open(value)
    return {
        "open": open_now,
        "timezone": config.MARKET_TIMEZONE,
        "local_time": value.isoformat(),
        "close_hour": float(config.MARKET_CLOSE_HOUR),
        "reopen_in_seconds": 0 if open_now else round(seconds_to_reopen(value), 1),
        "message": "" if open_now else
        f"MARKET CLOSED, reopens in {int(seconds_to_reopen(value) // 60)} min",
    }
