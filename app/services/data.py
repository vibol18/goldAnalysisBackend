"""Market data service: priority providers + TTL cache.

Only real market data is ever served. When every provider fails, the last real
cached snapshot is returned with ``stale=True``; if none exists the response is
empty and carries an error. Synthetic/fabricated prices are never generated.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app import config
from app.services.providers import PROVIDERS, get_live_gold_quote

logger = logging.getLogger(__name__)

TIMEFRAMES: dict[str, tuple[str, str, int]] = {
    "15m": ("15m", "5d", config.CACHE_TTL_INTRADAY),
    "1h": ("1h", "60d", config.CACHE_TTL_INTRADAY),
    "4h": ("1h", "60d", config.CACHE_TTL_INTRADAY),
    "1D": ("1d", "max", config.CACHE_TTL_DAILY),
}
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class CacheEntry:
    """A cached real dataframe plus freshness metadata."""

    df: pd.DataFrame
    fetched_at: float
    stale: bool = False
    synthetic: bool = False
    error: str | None = None
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def last_updated(self) -> str | None:
        return datetime.fromtimestamp(self.fetched_at, timezone.utc).isoformat()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _last_close(entry: CacheEntry) -> float | None:
    if entry.df is None or entry.df.empty or "Close" not in entry.df:
        return None
    value = entry.df["Close"].iloc[-1]
    return float(value) if pd.notna(value) else None


_cache: dict[str, CacheEntry] = {}
_cache_lock = threading.Lock()
_fetch_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_provider_cooldown: dict[tuple[str, str], tuple[float, str]] = {}
_cooldown_seconds = 120.0
PROVIDER_TIMEOUTS = {"twelvedata": 12.0, "yfinance": 5.0, "kraken-xaut": 8.0, "yahoo-chart": 14.0}


def _key(ticker: str, interval: str) -> str:
    return f"{ticker}|{interval}"


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _fetch_locks.setdefault(key, threading.Lock())


def _fresh(entry: CacheEntry | None, ttl: int, force: bool) -> bool:
    return (entry is not None and not force and not entry.stale
            and (time.time() - entry.fetched_at) < ttl)


def _fetch_with_timeout(provider: Any, ticker: str, timeframe: str,
                        timeout: float) -> pd.DataFrame:
    """Run a provider in a daemon thread so slow clients cannot block exit."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["df"] = provider.fetch(ticker, timeframe)
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True,
                              name=f"provider-{provider.name}")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"{provider.name} timed out after {timeout:g}s")
    if "error" in box:
        raise box["error"]
    return box["df"]


def get_ohlcv(ticker: str, timeframe: str, force: bool = False) -> CacheEntry:
    """Return real OHLCV candles, trying providers in priority order.

    Priority: Twelve Data (XAU/USD) → yfinance (XAUUSD=X, then GC=F) → last real
    cached snapshot (stale). No synthetic data is ever produced.
    """
    if timeframe not in TIMEFRAMES:
        timeframe = "1h"
    key = _key(ticker, timeframe)
    ttl = TIMEFRAMES[timeframe][2]
    with _cache_lock:
        entry = _cache.get(key)
    if _fresh(entry, ttl, force):
        return entry  # type: ignore[return-value]

    with _lock_for(key):
        with _cache_lock:
            entry = _cache.get(key)
        if _fresh(entry, ttl, force):
            return entry  # type: ignore[return-value]

        errors: list[str] = []
        fetched: CacheEntry | None = None
        for provider in PROVIDERS:
            if not provider.supports(ticker):
                continue
            cooldown_key = (provider.name, ticker)
            until, previous_error = _provider_cooldown.get(cooldown_key, (0.0, ""))
            if time.time() < until:
                errors.append(f"{provider.name}: temporarily unavailable ({previous_error[:80]})")
                continue
            try:
                df = _fetch_with_timeout(
                    provider, ticker, timeframe,
                    PROVIDER_TIMEOUTS.get(provider.name, 12.0),
                )
                _provider_cooldown.pop(cooldown_key, None)
                fetched = CacheEntry(df=df, fetched_at=time.time(),
                                     source=provider.name)
                break
            except Exception as exc:  # noqa: BLE001 - providers may raise anything
                message = f"{provider.name}: {exc}"
            errors.append(message)
            _provider_cooldown[cooldown_key] = (time.time() + _cooldown_seconds, message)
            logger.warning("provider failed for %s/%s: %s", ticker, timeframe, message)

        if fetched is not None:
            entry = fetched
        elif entry is not None and not entry.df.empty:
            entry = CacheEntry(df=entry.df, fetched_at=entry.fetched_at, stale=True,
                               error="; ".join(errors), source=entry.source)
        else:
            entry = CacheEntry(df=_empty_frame(), fetched_at=time.time(), stale=True,
                               error="; ".join(errors) or "no live data source available")

    with _cache_lock:
        _cache[key] = entry
    return entry


def refresh_realtime_candles() -> CacheEntry | None:
    """Fetch a fresh 15m gold candle from a fast source and refresh the cache.

    Deliberately skips yfinance: its DNS/retry behaviour can block for tens of
    seconds, which would make chat feel frozen.
    """
    for provider in PROVIDERS:
        if provider.name == "yfinance" or not provider.supports(config.GOLD_TICKER):
            continue
        try:
            df = _fetch_with_timeout(provider, config.GOLD_TICKER, "15m", 8.0)
            entry = CacheEntry(df=df, fetched_at=time.time(), source=provider.name)
            with _cache_lock:
                _cache[_key(config.GOLD_TICKER, "15m")] = entry
            return entry
        except Exception as exc:  # noqa: BLE001
            logger.debug("realtime refresh via %s failed: %s", provider.name, exc)
    return None


def get_market_overview() -> dict[str, Any]:
    """Latest real gold price and macro context. Missing values are reported as null."""
    out: dict[str, Any] = {"gold": {}, "correlations": {}, "meta": {}}
    intraday = get_ohlcv(config.GOLD_TICKER, "15m")
    daily = get_ohlcv(config.GOLD_TICKER, "1D")

    candle_price = _last_close(intraday)
    if candle_price is None:
        candle_price = _last_close(daily)

    quote_price = None
    quote_updated = None
    if candle_price is None:
        try:
            quote_price, quote_updated = get_live_gold_quote()
        except Exception as exc:  # noqa: BLE001
            logger.warning("live gold quote failed: %s", exc)

    price = candle_price if candle_price is not None else quote_price
    prev_close = _last_close(daily)
    if len(daily.df) > 1:
        try:
            prev_close = float(daily.df["Close"].iloc[-2])
        except (ValueError, TypeError):
            pass
    if price is None:
        price = prev_close

    hi = lo = None
    if not intraday.df.empty:
        try:
            today = intraday.df.index.max()
            day_mask = intraday.df.index.normalize() == pd.Timestamp(today).normalize()
            hi = float(intraday.df.loc[day_mask, "High"].max()) if day_mask.any() else None
            lo = float(intraday.df.loc[day_mask, "Low"].min()) if day_mask.any() else None
        except Exception:  # noqa: BLE001
            hi = lo = None

    out["gold"] = {
        "ticker": config.GOLD_TICKER,
        "symbol": "XAU/USD (Gold)",
        "price": price,
        "prev_close": prev_close,
        "change_pct": round((price - prev_close) / prev_close * 100.0, 4)
        if price is not None and prev_close else None,
        "day_high": hi,
        "day_low": lo,
        "open": float(daily.df["Open"].iloc[-1]) if not daily.df.empty else None,
        "as_of": quote_updated or (
            str(intraday.df.index[-1]) if not intraday.df.empty else None
        ),
        "stale": candle_price is None and intraday.stale,
        "source": (intraday.source if candle_price is not None
                   else ("gold-api" if quote_price is not None else None)),
        "error": intraday.error if price is None else None,
    }

    # ^TNX is quoted as yield x 10 by Yahoo, so scale it to a real percentage.
    for label, ticker, inverse, scale, unit in (
        ("DXY", config.DXY_TICKER, True, 1.0, ""),
        ("US10Y", config.TNX_TICKER, True, 10.0, "%"),
        ("SPX", config.SPX_TICKER, False, 1.0, ""),
    ):
        entry = get_ohlcv(ticker, "1D")
        last = _last_close(entry)
        prev = None
        if len(entry.df) > 1:
            try:
                prev = float(entry.df["Close"].iloc[-2])
            except (ValueError, TypeError):
                pass
        available = last is not None
        out["correlations"][label] = {
            "ticker": ticker,
            "price": round(last / scale, 3) if available else None,
            "change_pct": round((last - prev) / prev * 100.0, 4)
            if available and prev else None,
            "unit": unit,
            "stale": entry.stale,
            "synthetic": False,
            "error": None if available else (entry.error or "no live data available"),
        }

    fresh_entries = [e for e in (intraday, daily) if not e.stale and not e.df.empty]
    out["meta"] = {
        "stale": candle_price is None and (intraday.stale or daily.stale),
        "synthetic": False,
        "source": (intraday.source if candle_price is not None
                   else ("gold-api" if quote_price is not None else None)),
        "last_updated": (quote_updated if quote_price is not None else
                         max((e.last_updated for e in fresh_entries), default=None)),
        "error": intraday.error if price is None else None,
    }
    return out


def clear_cache() -> None:
    """Drop all cached data (used by tests / settings refresh)."""
    with _cache_lock:
        _cache.clear()
