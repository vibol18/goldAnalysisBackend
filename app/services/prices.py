"""Single in-memory price feed shared by every client.

One tick stream is ingested from whichever source is healthiest; all users read
the same latest tick. No fetch is performed per user or per request.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app import config
from app.services import market_hours

logger = logging.getLogger(__name__)

LIVE = "LIVE"
DELAYED = "DELAYED"
STALE = "STALE"
CLOSED = "MARKET_CLOSED"


@dataclass
class Tick:
    source: str
    bid: float
    ask: float
    ts: float
    broker_time: int | None = None


class PriceFeed:
    """Latest tick per source plus active-source failover."""

    def __init__(self) -> None:
        self._ticks: dict[str, Tick] = {}
        self._lock = threading.RLock()
        self._active: str | None = None
        self._reason: str = ""

    def update(self, source: str, bid: float, ask: float,
               ts: float | None = None, broker_time: int | None = None) -> None:
        source = str(source or "").strip().lower()
        bid, ask = float(bid), float(ask)
        if not (bid > 0 and ask >= bid):
            return
        tick = Tick(source=source, bid=bid, ask=ask,
                    ts=float(ts) if ts is not None else time.time(),
                    broker_time=broker_time)
        with self._lock:
            self._ticks[source] = tick
            if self._active is None:
                self._active = source
            self._reason = ""

    def mark_error(self, source: str, reason: str) -> None:
        with self._lock:
            if self._active == source:
                self._reason = str(reason)[:160]

    def reset(self) -> None:
        with self._lock:
            self._ticks.clear()
            self._active = None
            self._reason = ""

    def _fresh_sources(self, now: float) -> list[str]:
        priority = list(config.DATA_SOURCE_PRIORITY)
        for source in self._ticks:
            if source not in priority:
                priority.append(source)
        return [s for s in priority if s in self._ticks
                and now - self._ticks[s].ts <= config.PRICE_FAILOVER_SECONDS]

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = float(now) if now is not None else time.time()
        market = market_hours.status(datetime.fromtimestamp(now, timezone.utc))
        with self._lock:
            ticks = dict(self._ticks)
            active = self._active
            reason = self._reason
        fresh = self._fresh_sources(now)
        if active not in fresh:
            active = fresh[0] if fresh else active
        tick = ticks.get(active) if active else None
        if tick is None and ticks:
            tick = max(ticks.values(), key=lambda t: t.ts)
            active = tick.source
        age = (now - tick.ts) if tick else None
        if not market["open"]:
            state = CLOSED
        elif age is None:
            state = STALE
        elif age < config.PRICE_LIVE_SECONDS:
            state = LIVE
        elif age <= config.PRICE_DELAYED_SECONDS:
            state = DELAYED
        else:
            state = STALE
        effective_reason = reason or (
            "market closed" if state == CLOSED else
            "no tick received" if state == STALE else ""
        )
        bid = tick.bid if tick else None
        ask = tick.ask if tick else None
        price = _selected_price(bid, ask)
        return {
            "state": state,
            "source": active or "none",
            "bid": bid,
            "ask": ask,
            "spread": round(ask - bid, 4) if bid is not None and ask is not None else None,
            "price": price,
            "time": datetime.fromtimestamp(tick.ts, timezone.utc).isoformat() if tick else None,
            "age_seconds": round(age, 2) if age is not None else None,
            "reason": effective_reason,
            "market": market,
        }

    def all_ticks(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                source: {
                    "source": tick.source, "bid": tick.bid, "ask": tick.ask,
                    "spread": round(tick.ask - tick.bid, 4),
                    "time": datetime.fromtimestamp(tick.ts, timezone.utc).isoformat(),
                    "age_seconds": round(time.time() - tick.ts, 2),
                    "broker_time": tick.broker_time,
                }
                for source, tick in self._ticks.items()
            }

    def received_any(self) -> bool:
        with self._lock:
            return bool(self._ticks)

    def is_stale(self, now: float | None = None) -> bool:
        if not self.received_any():
            return False
        return self.snapshot(now)["state"] in (STALE, CLOSED)


def _selected_price(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return bid if bid is not None else ask
    mode = config.PRICE_BID_MODE
    if mode == "ask":
        return ask
    if mode == "mid":
        return (bid + ask) / 2.0
    return bid


FEED = PriceFeed()
