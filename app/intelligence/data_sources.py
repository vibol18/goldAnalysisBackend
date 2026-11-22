"""Real macro / intermarket / positioning data with source attribution.

Every public loader returns a :class:`SeriesResult` that is explicitly marked
``available=False`` with an ``error`` when a provider fails.  Nothing is
fabricated: missing data is rendered as N/A by the API and the UI.  All series
are cached with a TTL and fall back to the last real snapshot (flagged stale).
"""
from __future__ import annotations

import io
import json
import logging
import math
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd

from app import config
from app.services import data as market_data

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, "SeriesResult"]] = {}
_LOCK = threading.RLock()
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Meas-Pro/1.0)"}


@dataclass
class SeriesResult:
    """A real, cached data series plus provenance."""

    key: str
    source: str
    frame: pd.DataFrame | None = None
    updated_at: str | None = None
    available: bool = False
    error: str | None = None
    stale: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def observations(self) -> int:
        return 0 if self.frame is None else int(len(self.frame))

    def to_dict(self, include_frame: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "source": self.source,
            "available": self.available,
            "observations": self.observations,
            "updated_at": self.updated_at,
            "stale": self.stale,
            "error": self.error,
            "meta": self.meta,
        }
        if include_frame and self.frame is not None:
            out["data"] = self.frame.reset_index().assign(
                lambda d: d.iloc[:, 0].astype(str)
            ).to_dict(orient="records")
        return out


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _cached(key: str, ttl: float, loader: Callable[[], SeriesResult]) -> SeriesResult:
    """Return a fresh result, else the last real (stale) snapshot."""
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < ttl and hit[1].available:
            return hit[1]
    try:
        result = loader()
    except Exception as exc:  # noqa: BLE001 - provider failures are data, not crashes
        logger.warning("data loader %s failed: %s", key, exc)
        result = SeriesResult(key=key, source="unavailable", error=f"{type(exc).__name__}: {exc}")
    result.updated_at = result.updated_at or _iso(_now())
    with _LOCK:
        if result.available:
            _CACHE[key] = (now, result)
        elif hit is not None:
            stale = SeriesResult(
                key=key,
                source=hit[1].source,
                frame=hit[1].frame,
                updated_at=hit[1].updated_at,
                available=True,
                error=result.error,
                stale=True,
                meta=hit[1].meta,
            )
            return stale
    return result


def _http_get(url: str, timeout: float = 20.0) -> str:
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=_HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _http_get_bytes(url: str, timeout: float = 20.0) -> bytes:
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=_HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _numeric_frame(text: str, value_col: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 2:
        raise ValueError("expected two columns from FRED CSV")
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])
    frame["date"] = frame["date"].dt.tz_localize(None)
    out = frame.set_index("date")[["value"]].sort_index()
    out.index.name = "date"
    out.attrs["value_col"] = value_col
    return out


def fred_series(series_id: str, years: int = 8, ttl: float | None = None) -> SeriesResult:
    """Fetch a real FRED series via its keyless CSV endpoint."""
    ttl = config.MACRO_TTL_SECONDS if ttl is None else ttl
    start = (_now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    def load() -> SeriesResult:
        url = f"{config.FRED_BASE_URL}?id={series_id}&cosd={start}"
        text = _http_get(url)
        frame = _numeric_frame(text, series_id)
        if frame.empty:
            raise ValueError(f"FRED returned no observations for {series_id}")
        return SeriesResult(
            key=f"fred:{series_id}",
            source=f"FRED {series_id}",
            frame=frame,
            updated_at=_iso(_now()),
            available=True,
            meta={"series_id": series_id, "units": "percent" if series_id not in {"DFII10"} else "percent"},
        )

    return _cached(f"fred:{series_id}", ttl, load)


def ohlcv_daily(ticker: str, label: str | None = None) -> SeriesResult:
    """Fetch real daily OHLCV through the existing provider chain."""
    key = f"ohlcv:{ticker}"

    def load() -> SeriesResult:
        entry = market_data.get_ohlcv(ticker, "1D")
        frame = entry.df
        if frame is None or frame.empty:
            raise ValueError(entry.error or "no candles returned")
        return SeriesResult(
            key=key,
            source=f"{label or ticker} via {entry.source or 'provider'}",
            frame=frame,
            updated_at=entry.last_updated,
            available=not entry.stale,
            error=entry.error,
            stale=entry.stale,
            meta={"ticker": ticker, "timeframe": "1D"},
        )

    return _cached(key, config.MACRO_TTL_SECONDS, load)


# ---------------------------------------------------------------------------
# Intermarket: gold correlations and break flags
# ---------------------------------------------------------------------------

DRIVERS: dict[str, tuple[str, str, bool]] = {
    "dxy": (config.DXY_TICKER, "US dollar index", True),
    "eurusd": (config.EURUSD_TICKER, "EUR/USD", False),
    "usdjpy": (config.USDJPY_TICKER, "USD/JPY", True),
    "silver": (config.SILVER_TICKER, "Silver", False),
    "copper": (config.COPPER_TICKER, "Copper", False),
    "oil": (config.OIL_TICKER, "WTI crude", False),
    "gld": (config.GLD_TICKER, "GLDvolume", False),
    "gdx": (config.GDX_TICKER, "GDX volume", False),
}


def _returns(frame: pd.DataFrame | None) -> pd.Series:
    """Daily log returns indexed by calendar date (aligns FX, futures and ETFs)."""
    if frame is None or frame.empty or "Close" not in frame:
        return pd.Series(dtype=float)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    close = close.where(close > 0)
    idx = pd.to_datetime(close.index, errors="coerce", utc=True).tz_localize(None).normalize()
    close.index = idx
    close = close[~close.index.duplicated(keep="last")].dropna().sort_index()
    return np.log(close).diff().replace([np.inf, -np.inf], np.nan).dropna()


def intermarket_snapshot() -> dict[str, Any]:
    """Rolling gold/driver correlations (20/60/120d) with break flags."""
    key = "intermarket"
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < config.MACRO_TTL_SECONDS:
            return hit[1].meta

    gold = _returns(ohlcv_daily(config.GOLD_TICKER, "XAU/USD").frame)
    result: dict[str, Any] = {"available": False, "drivers": {}, "windows": list(config.ROLL_CORR_WINDOWS)}
    if gold.empty:
        result["error"] = "gold daily data unavailable"
        return result

    for name, (ticker, label, inverse) in DRIVERS.items():
        series = _returns(ohlcv_daily(ticker, label).frame)
        joined = pd.concat([gold.rename("gold"), series.rename("driver")], axis=1, join="inner").dropna()
        entry: dict[str, Any] = {
            "label": label,
            "ticker": ticker,
            "inverse": inverse,
            "available": len(joined) > 30,
            "observations": int(len(joined)),
            "source": ohlcv_daily(ticker, label).source,
            "updated_at": ohlcv_daily(ticker, label).updated_at,
        }
        if entry["available"]:
            for window in config.ROLL_CORR_WINDOWS:
                corr = joined["gold"].rolling(window, min_periods=max(10, window // 2)).corr(joined["driver"])
                valid = corr.dropna()
                if valid.empty:
                    continue
                latest = float(valid.iloc[-1])
                baseline = float(valid.iloc[-min(len(valid), 120):].mean())
                entry[f"corr_{window}"] = round(latest, 3)
                entry[f"baseline_{window}"] = round(baseline, 3)
                entry[f"break_{window}"] = bool(
                    len(valid) > window and (latest * baseline < 0 or abs(latest - baseline) > 0.45)
                )
            entry["latest"] = entry.get("corr_60")
            signed = -entry["latest"] if inverse and entry.get("latest") is not None else entry.get("latest")
            entry["gold_signal"] = "bullish" if signed is not None and signed < -0.15 else (
                "bearish" if signed is not None and signed > 0.15 else "neutral"
            )
        result["drivers"][name] = entry
    result["available"] = any(v["available"] for v in result["drivers"].values())
    result["gold_observations"] = int(len(gold))
    with _LOCK:
        _CACHE[key] = (now, SeriesResult(key=key, source="yfinance/provider chain", meta=result, available=True, updated_at=_iso(_now())))
    return result


def _close_by_date(frame: pd.DataFrame) -> pd.Series:
    """Daily close keyed by calendar date, deduped (FX/futures/ETF alignment)."""
    close = pd.to_numeric(frame["Close"], errors="coerce")
    close.index = pd.to_datetime(close.index, errors="coerce", utc=True).tz_localize(None).normalize()
    close = close[close > 0]
    return close[~close.index.duplicated(keep="last")].dropna().sort_index()


def intermarket_correlation_frame() -> pd.DataFrame:
    """Rolling correlation series by driver/window, indexed by date (real data)."""
    gold = _returns(ohlcv_daily(config.GOLD_TICKER, "XAU/USD").frame)
    out = pd.DataFrame(index=gold.index)
    for name, (ticker, label, inverse) in DRIVERS.items():
        series = _returns(ohlcv_daily(ticker, label).frame)
        joined = pd.concat([gold.rename("gold"), series.rename("driver")], axis=1,
                           join="inner").dropna()
        for window in config.ROLL_CORR_WINDOWS:
            corr = joined["gold"].rolling(window, min_periods=max(10, window // 2)).corr(joined["driver"])
            out[f"corr_{name}_{window}"] = corr.reindex(out.index) * (-1 if inverse else 1)
    return out.replace([np.inf, -np.inf], np.nan)


def gold_silver_ratio() -> dict[str, Any]:
    gold = ohlcv_daily(config.GOLD_TICKER, "XAU/USD")
    silver = ohlcv_daily(config.SILVER_TICKER, "Silver")
    if not (gold.available and silver.available and gold.frame is not None and silver.frame is not None):
        return {"available": False, "value": None, "source": "N/A", "updated_at": None}
    g = _close_by_date(gold.frame)
    s = _close_by_date(silver.frame)
    ratio = g.reindex(s.index).ffill() / s.replace(0, np.nan)
    valid = ratio.dropna()
    if valid.empty:
        return {"available": False, "value": None, "source": "N/A", "updated_at": None}
    return {
        "available": True,
        "value": round(float(valid.iloc[-1]), 2),
        "change_20d": round(float(valid.iloc[-1] - valid.iloc[-21]), 2) if len(valid) > 21 else None,
        "source": f"{gold.source} / {silver.source}",
        "updated_at": max(gold.updated_at or "", silver.updated_at or "") or None,
    }


# ---------------------------------------------------------------------------
# CFTC Commitments of Traders (free weekly data)
# ---------------------------------------------------------------------------

def cot_gold() -> SeriesResult:
    """Managed-money net long, weekly change and 3-year extremes for gold."""
    ttl = config.COT_TTL_SECONDS
    key = "cftc:gold"

    def load() -> SeriesResult:
        frames = []
        current_year = _now().year
        for year in (current_year, current_year - 1, current_year - 2, current_year - 3):
            url = config.CFTC_GOLD_HISTORY_URL.format(year=year)
            try:
                raw = _http_get_bytes(url, timeout=35)
                archive = zipfile.ZipFile(io.BytesIO(raw))
                name = next(n for n in archive.namelist() if n.endswith(".txt"))
                frames.append(pd.read_csv(archive.open(name), low_memory=False))
            except Exception as exc:  # noqa: BLE001
                logger.debug("CFTC %s unavailable: %s", year, exc)
                continue
        if not frames:
            raise ValueError("no CFTC annual reports available")
        all_rows = pd.concat(frames, ignore_index=True)
        names = all_rows["Market_and_Exchange_Names"].astype(str).str.upper()
        market = names.str.startswith("GOLD -") & ~names.str.contains("MICRO")
        fut = all_rows.get("FutOnly_or_Combined", pd.Series("FutOnly", index=all_rows.index)).astype(str).eq("FutOnly")
        rows = all_rows[market & fut].copy()
        if rows.empty:
            raise ValueError("no gold managed-money rows in CFTC data")
        rows["date"] = pd.to_datetime(rows["Report_Date_as_YYYY-MM-DD"], errors="coerce", utc=True).dt.tz_localize(None)
        for col in ("M_Money_Positions_Long_All", "M_Money_Positions_Short_All", "Open_Interest_All"):
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
        rows = rows.dropna(subset=["date", "M_Money_Positions_Long_All", "M_Money_Positions_Short_All"])
        rows = rows.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        longs = rows["M_Money_Positions_Long_All"]
        shorts = rows["M_Money_Positions_Short_All"]
        frame = pd.DataFrame({
            "long": longs,
            "short": shorts,
            "net_long": longs - shorts,
            "open_interest": rows.get("Open_Interest_All", np.nan),
        })
        frame["weekly_change"] = frame["net_long"].diff()
        frame["pct_3y"] = frame["net_long"].rolling(156, min_periods=26).rank(pct=True)
        frame["extreme"] = np.where(frame["pct_3y"] >= 0.9, "crowded_long",
                            np.where(frame["pct_3y"] <= 0.1, "crowded_short", "normal"))
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        latest = frame.iloc[-1]
        return SeriesResult(
            key=key,
            source="CFTC Disaggregated Futures (free weekly)",
            frame=frame,
            updated_at=_iso(_now()),
            available=True,
            meta={
                "net_long": int(latest["net_long"]),
                "weekly_change": int(latest["weekly_change"]) if pd.notna(latest["weekly_change"]) else None,
                "percentile_3y": round(float(latest["pct_3y"]), 3) if pd.notna(latest["pct_3y"]) else None,
                "extreme": str(latest["extreme"]),
                "report_date": frame.index[-1].strftime("%Y-%m-%d"),
                "release_lag_days": 3,
                "source": "CFTC Disaggregated Futures (free weekly)",
            },
        )

    return _cached(key, ttl, load)


# ---------------------------------------------------------------------------
# Economic calendar: countdown + measured gold reaction by event type
# ---------------------------------------------------------------------------

_EVENT_TYPES = (
    ("nfp", (r"nonfarm\s+payroll", r"non-farm\s+payroll")),
    ("cpi", (r"\bcpi\b", r"consumer\s+price\s+index")),
    ("fomc", (r"fomc", r"federal\s+funds\s+rate", r"fomc\s+statement")),
    ("pce", (r"\bpce\b", r"personal\s+consumption")),
    ("gdp", (r"\bgdp\b", r"gross\s+domestic")),
    ("jobless_claims", (r"jobless\s+claims", r"initial\s+claims")),
)


def _classify_event(title: str) -> str:
    low = title.lower()
    for kind, patterns in _EVENT_TYPES:
        if any(re.search(p, low) for p in patterns):
            return kind
    return "other"


def fetch_calendar_events() -> list[dict[str, Any]]:
    """Fetch a free real calendar, merge with the editable file, never invent data."""
    key = "calendar:events"
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < config.CALENDAR_TTL_SECONDS:
            return list(hit[1].meta.get("events", []))

    from app.services.concept_context import load_events
    events = list(load_events())
    error = None
    if config.ECONOMIC_CALENDAR_URL:
        try:
            raw = _http_get(config.ECONOMIC_CALENDAR_URL, timeout=15)
            decoded = raw if "<event" in raw else raw.encode("latin-1").decode("cp1252", errors="replace")
            blocks = re.findall(r"<event>(.*?)</event>", decoded, re.S | re.I)

            def field(block: str, name: str) -> str:
                match = re.search(rf"<{name}>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</{name}>", block, re.S | re.I)
                return re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else ""

            for block in blocks:
                title = field(block, "title")
                date_s = field(block, "date")
                time_s = field(block, "time")
                if not (title and date_s):
                    continue
                try:
                    when = pd.to_datetime(f"{date_s} {time_s or '00:00'}",
                                          format="%m-%d-%Y %H:%M", errors="coerce")
                except (ValueError, TypeError):
                    continue
                if pd.isna(when):
                    continue
                impact = field(block, "impact").lower()
                if impact not in {"high", "medium", "low"}:
                    continue
                events.append({
                    "type": _classify_event(title),
                    "title": title,
                    "country": field(block, "country"),
                    "time": when.replace(tzinfo=timezone.utc).isoformat(),
                    "impact": impact,
                    "expected": field(block, "forecast"),
                    "actual": field(block, "actual"),
                    "previous": field(block, "previous"),
                    "source": "ForexFactory free feed",
                })
        except Exception as exc:  # noqa: BLE001 - rate limits / outages are N/A
            error = f"{type(exc).__name__}: {exc}"
            logger.debug("economic calendar unavailable: %s", exc)
    events.sort(key=lambda e: str(e.get("time", "")))
    with _LOCK:
        _CACHE[key] = (now, SeriesResult(
            key=key, source="ForexFactory + events.json" if events else "unavailable",
            available=bool(events), error=error, updated_at=_iso(_now()),
            meta={"events": events},
        ))
    return events


def calendar_countdown(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return the next event and a countdown, or N/A when the calendar is empty."""
    if events is None:
        events = fetch_calendar_events()
    now = _now()
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for event in events:
        raw = event.get("time") or event.get("datetime") or event.get("date")
        if not raw:
            continue
        try:
            when = pd.to_datetime(raw, utc=True, errors="coerce")
            if pd.isna(when):
                continue
            dt = when.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            parsed.append((dt, event))
        except (ValueError, TypeError):
            continue
    upcoming = sorted(((dt, e) for dt, e in parsed if dt >= now), key=lambda x: x[0])
    if not upcoming:
        return {"available": False, "next": None, "reason": "no future events in calendar"}
    when, event = upcoming[0]
    delta = when - now
    return {
        "available": True,
        "next": {**event, "time": _iso(when)},
        "hours_until": round(delta.total_seconds() / 3600, 2),
        "days_until": round(delta.total_seconds() / 86400, 2),
        "count": len(upcoming),
        "source": "ForexFactory free feed + editable events.json",
        "updated_at": _iso(now),
    }


def calendar_reaction_stats(events: list[dict[str, Any]] | None = None,
                            gold_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Measure realised gold move after each past event type (no lookahead in features)."""
    if events is None:
        events = fetch_calendar_events()
    if gold_frame is None:
        gold_frame = ohlcv_daily(config.GOLD_TICKER, "XAU/USD").frame
    if gold_frame is None or gold_frame.empty:
        return {"available": False, "by_type": {}, "reason": "gold data unavailable"}
    bars = gold_frame.copy()
    bars.index = pd.to_datetime(bars.index, utc=True).tz_localize(None)
    stats: dict[str, list[dict[str, float]]] = {}
    for event in events:
        raw = event.get("time") or event.get("datetime") or event.get("date")
        if not raw or event.get("actual") in (None, ""):
            continue
        try:
            when = pd.to_datetime(raw, utc=True).tz_localize(None)
            expected = float(event.get("expected"))
            actual = float(event["actual"])
        except (ValueError, TypeError):
            continue
        pos = bars.index.searchsorted(when)
        if pos + 6 >= len(bars):
            continue
        base = float(bars["Close"].iloc[pos])
        if base <= 0:
            continue
        surprise = (actual - expected) / max(abs(expected), 1e-9)
        fwd = [float(bars["Close"].iloc[pos + k]) / base - 1 for k in (4, 12, 24)]
        stats.setdefault(str(event.get("type", "other")), []).append({
            "surprise": surprise,
            "abs_surprise": abs(surprise),
            "r4": abs(fwd[0]) * 100,
            "r12": abs(fwd[1]) * 100,
            "r24": abs(fwd[2]) * 100,
            "dir12": fwd[1],
        })
    out: dict[str, Any] = {"available": bool(stats), "by_type": {}, "source": "real calendar + real XAU/USD"}
    for kind, rows in stats.items():
        frame = pd.DataFrame(rows)
        out["by_type"][kind] = {
            "events": len(rows),
            "avg_abs_move_24bars_pct": round(float(frame["r24"].mean()), 3),
            "avg_abs_move_12bars_pct": round(float(frame["r12"].mean()), 3),
            "corr_surprise_move": round(float(frame["abs_surprise"].corr(frame["r12"])), 3)
            if len(rows) > 2 and frame["abs_surprise"].nunique() > 1 else None,
            "up_fraction_12bars": round(float((frame["dir12"] > 0).mean()), 3),
        }
    return out


# ---------------------------------------------------------------------------
# News sentiment: LLM classification with recency weighting (lexicon fallback)
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS = {
    "fed": ("fed", "fomc", "powell", "rate cut", "rate hike", "interest rate"),
    "inflation": ("inflation", "cpi", "pce", "price index"),
    "geopolitics": ("war", "geopolit", "sanction", "missile", "strike", "ukraine", "middle east"),
    "dollar": ("dollar", "dxy", "greenback", "usd"),
}

_TOPIC_SYSTEM = (
    "You classify gold-market headlines. Reply with JSON only: "
    '{"score": <number from -1 bearish to +1 bullish>, "topic": <fed|inflation|geopolitics|dollar|other>}. '
    "Judge the impact on XAU/USD gold price. Never invent facts."
)


def _lexicon_score(text: str) -> float:
    from app.services.news import _score_text  # local import avoids circulars in tests
    return float(_score_text(text))


def _topic_of(text: str) -> str:
    low = text.lower()
    for topic, words in _TOPIC_KEYWORDS.items():
        if any(w in low for w in words):
            return topic
    return "other"


def _llm_classify(headlines: list[str]) -> list[dict[str, Any]] | None:
    """Batch-classify headlines with the configured LLM; None means unavailable."""
    if not (config.NEWS_LLM_ENABLED and config.llm_configured()):
        return None
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL or (
        config.ANTHROPIC_DEFAULT_MODEL if provider == "anthropic" else config.OPENAI_DEFAULT_MODEL
    )
    prompt = "Classify each headline:\n" + "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    try:
        if provider == "anthropic":
            url, headers, payload = "https://api.anthropic.com/v1/messages", {
                "x-api-key": config.LLM_API_KEY, "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }, {"model": model, "max_tokens": 700, "system": _TOPIC_SYSTEM,
                "messages": [{"role": "user", "content": prompt}]}
            text = _http_post_json(url, headers, payload)
            content = "".join(b.get("text", "") for b in text.get("content", []) if isinstance(b, dict))
        else:
            url, headers, payload = "https://api.openai.com/v1/chat/completions", {
                "Authorization": f"Bearer {config.LLM_API_KEY}", "content-type": "application/json",
            }, {"model": model, "temperature": 0,
                "messages": [{"role": "system", "content": _TOPIC_SYSTEM}, {"role": "user", "content": prompt}]}
            text = _http_post_json(url, headers, payload)
            content = text["choices"][0]["message"]["content"]
        blocks = re.findall(r"\{[^{}]+\}", content)
        parsed = []
        for i in range(len(headlines)):
            match = next((b for b in blocks if re.search(rf'\b{i}\b', b)), None)
            item = json.loads(match) if match else {}
            parsed.append({"score": float(item.get("score", 0.0)), "topic": str(item.get("topic", "other"))})
        return parsed if len(parsed) == len(headlines) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM news classification failed: %s", exc)
        return None


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=25.0, headers=headers) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def news_intelligence(force: bool = False) -> dict[str, Any]:
    """LLM/lexicon sentiment + topic, recency weighted, stored as a feature."""
    from app.services.news import fetch_news

    payload = fetch_news(force=force)
    items = payload.get("items", [])
    if not items:
        return {"available": False, "score": None, "label": "N/A", "items": [],
                "error": payload.get("error") or "no headlines", "source": "N/A"}
    headlines = [str(i.get("title", "")) for i in items]
    llm = _llm_classify(headlines)
    half_life = max(1.0, config.NEWS_SENTIMENT_HALFLIFE_HOURS)
    rows = []
    now = _now()
    for i, (item, title) in enumerate(zip(items, headlines)):
        raw = item.get("published")
        try:
            published = pd.to_datetime(raw, utc=True).to_pydatetime() if raw else now
        except (ValueError, TypeError):
            published = now
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (now - published).total_seconds() / 3600)
        weight = 0.5 ** (age_hours / half_life)
        if llm and i < len(llm):
            score = max(-1.0, min(1.0, llm[i]["score"]))
            topic = llm[i]["topic"]
            method = "llm"
        else:
            score = _lexicon_score(title)
            topic = _topic_of(title)
            method = "lexicon"
        rows.append({
            "title": title, "link": item.get("link"), "published": item.get("published"),
            "score": round(score, 3), "topic": topic, "weight": round(weight, 3),
            "age_hours": round(age_hours, 1), "method": method,
        })
    total_weight = sum(r["weight"] for r in rows) or 1.0
    score = sum(r["score"] * r["weight"] for r in rows) / total_weight
    label = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
    return {
        "available": True,
        "score": round(score, 3),
        "label": label,
        "method": "llm" if llm else "lexicon",
        "half_life_hours": half_life,
        "items": rows,
        "source": "RSS + " + ("LLM" if llm else "lexicon"),
        "updated_at": _iso(now),
    }


# ---------------------------------------------------------------------------
# Aggregate snapshot used by the API, report and chat
# ---------------------------------------------------------------------------

def macro_snapshot(force: bool = False) -> dict[str, Any]:
    """One call returning every real macro input with source + update time."""
    real_yield = fred_series(config.REAL_YIELD_SERIES)
    breakeven = fred_series(config.BREAKEVEN_SERIES)
    drivers = intermarket_snapshot()
    cot = cot_gold()
    countdown = calendar_countdown()
    news = news_intelligence(force=force)
    ratio = gold_silver_ratio()

    def summary(result: SeriesResult, change_window: int = 20) -> dict[str, Any]:
        base = result.to_dict()
        if result.frame is not None and not result.frame.empty:
            values = result.frame["value"].dropna()
            base["latest"] = round(float(values.iloc[-1]), 3) if len(values) else None
            base["change"] = round(float(values.iloc[-1] - values.iloc[-change_window]), 3) if len(values) > change_window else None
        return base

    return {
        "available": True,
        "generated_at": _iso(_now()),
        "real_yield": summary(real_yield),
        "breakeven_inflation": summary(breakeven),
        "intermarket": drivers,
        "gold_silver_ratio": ratio,
        "cot": cot.to_dict(),
        "calendar": countdown,
        "event_reactions": calendar_reaction_stats(),
        "news": {k: v for k, v in news.items() if k != "items"} | {"headlines": news.get("items", [])[:8]},
        "disclaimer": "Educational analysis only. Not financial advice. Signals can be wrong. Never risk money you can't afford to lose.",
    }


def as_of_value(series: pd.DataFrame | None, when: pd.Timestamp,
                lag_days: int = 0) -> tuple[float | None, pd.Timestamp | None]:
    """Point-in-time value of a daily series with a release lag (no lookahead)."""
    if series is None or series.empty:
        return None, None
    cutoff = pd.Timestamp(when).tz_localize(None) - pd.Timedelta(days=lag_days)
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    eligible = series.iloc[: idx.searchsorted(cutoff, side="right")]
    if eligible.empty:
        return None, None
    last = eligible.iloc[-1]
    value = last.get("value", last.get("net_long", last.get("Close", np.nan)))
    if pd.isna(value) and len(eligible.columns):
        value = last.iloc[0]
    return (float(value), pd.Timestamp(eligible.index[-1])) if pd.notna(value) else (None, None)
