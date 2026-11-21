"""Gold news via free RSS feeds + a tiny lexicon sentiment score."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=gold+price+XAU/USD&hl=en-US&gl=US&ceid=US:en",
    "https://www.investing.com/rss/news_301.rss",
    "https://www.investing.com/rss/news_62.rss",  # commodities news
]

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

BULLISH = {
    "surge", "rally", "jump", "gain", "rise", "high", "record", "bullish",
    "boost", "support", "demand", "safe-haven", "safe haven", "hawkish-fed",
    "rate cut", "cuts", "weaker dollar", "inflation", "higher", "soars",
    "climbs", "advances", "up", "strong", "outperform",
}
BEARISH = {
    "drop", "fall", "decline", "slide", "sink", "bearish", "plunge", "lower",
    "weak", "sell-off", "selloff", "correction", "dovish", "hawkish",
    "strong dollar", "yields rise", "rate hike", "hikes", "down", "tumbles",
    "slips", "retreats", "pressure", "resistance",
}

_cache: dict[str, Any] = {"fetched_at": 0.0, "items": [], "sentiment": 0.0,
                          "error": None}
CACHE_TTL = 900.0


def _score_text(text: str) -> float:
    """Lexicon sentiment in [-1, 1]."""
    words = set(text.lower().replace("-", " ").split())
    b = sum(1 for w in BULLISH if w.replace("-", " ") in text.lower())
    s = sum(1 for w in BEARISH if w.replace("-", " ") in text.lower())
    total = b + s
    if total == 0:
        return 0.0
    return (b - s) / total


def fetch_news(force: bool = False) -> dict[str, Any]:
    """Fetch headlines from RSS feeds with caching; failures are non-fatal."""
    import time

    if not force and _cache["items"] and (time.time() - _cache["fetched_at"]) < CACHE_TTL:
        return _cache

    items: list[dict[str, str]] = []
    errors: list[str] = []
    for url in RSS_FEEDS:
        try:
            import feedparser

            with httpx.Client(timeout=8.0, follow_redirects=True,
                              headers=_HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            for e in feed.entries[:8]:
                title = getattr(e, "title", "") or ""
                if not title:
                    continue
                link = getattr(e, "link", "")
                source = ""
                src = getattr(e, "source", None)
                if src is not None and getattr(src, "title", None):
                    source = src.title
                elif src is not None and isinstance(src, str):
                    source = src
                items.append({
                    "title": title,
                    "link": link,
                    "published": getattr(e, "published", ""),
                    "source": source or url.split("/")[2],
                })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
            logger.info("rss failed %s: %s", url, exc)
        if len(items) >= 12:
            break

    scores = [_score_text(i["title"]) for i in items]
    sentiment = sum(scores) / len(scores) if scores else 0.0
    _cache.update({
        "fetched_at": time.time(),
        "items": items,
        "sentiment": round(sentiment, 3),
        "sentiment_label": (
            "bullish" if sentiment > 0.15 else
            "bearish" if sentiment < -0.15 else "neutral"),
        "error": "; ".join(errors[:2]) if errors and not items else None,
    })
    return _cache
