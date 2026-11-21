"""Small in-process sliding-window rate limiter (per IP and per user)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

_LOCK = threading.Lock()
_EVENTS: dict[str, deque[float]] = defaultdict(deque)


def check(key: str, limit: int, window: int = 60) -> None:
    """Raise 429 when `key` exceeds `limit` requests in `window` seconds."""
    now = time.monotonic()
    with _LOCK:
        bucket = _EVENTS[key]
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry = max(1, int(window - (now - bucket[0])))
            raise HTTPException(429, f"Rate limit exceeded. Retry in {retry}s")
        bucket.append(now)
        if len(_EVENTS) > 20000:
            for stale in [k for k, v in _EVENTS.items() if not v]:
                _EVENTS.pop(stale, None)


def ip_key(scope: str, ip: str) -> str:
    return f"{scope}:ip:{ip}"


def user_key(scope: str, user_id: int) -> str:
    return f"{scope}:user:{user_id}"
