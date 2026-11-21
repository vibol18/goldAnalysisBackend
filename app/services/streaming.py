"""Shared WebSocket streaming service: one fetch loop feeding many subscribers."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app import config
from app.services import chart_annotations, data, db, news

logger = logging.getLogger(__name__)


STREAM_INTERVAL_SECONDS = max(1.0, config.STREAM_INTERVAL_SECONDS)
FORCE_REFRESH_SECONDS = max(1.0, config.STREAM_FORCE_REFRESH_SECONDS)
MACRO_INTERVAL_SECONDS = max(1.0, config.STREAM_MACRO_INTERVAL_SECONDS)
NEWS_INTERVAL_SECONDS = max(1.0, config.STREAM_NEWS_INTERVAL_SECONDS)
ANNOTATION_INTERVAL_SECONDS = max(1.0, config.STREAM_ANNOTATION_INTERVAL_SECONDS)
QUEUE_MAXSIZE = max(1, config.STREAM_QUEUE_MAXSIZE)
SUPPORTED_TIMEFRAMES = tuple(data.TIMEFRAMES)

_TF_ALIASES = {
    "15m": "15m", "15min": "15m", "1h": "1h", "60m": "1h", "1hour": "1h",
    "4h": "4h", "240m": "4h", "1d": "1D", "1day": "1D", "daily": "1D",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(type_: str, **payload: Any) -> dict[str, Any]:
    """Build a stream event with a UTC ISO timestamp."""
    return {"type": type_, **payload, "ts": _now_iso()}


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if pd.notna(result) else default


def normalize_timeframe(value: Any) -> str:
    """Map client timeframe spellings to the canonical set."""
    text = str(value or "").strip()
    if text in SUPPORTED_TIMEFRAMES:
        return text
    return _TF_ALIASES.get(text.lower(), "1h")


class StreamHub:
    """Fan-out hub: every connection owns one bounded asyncio queue."""

    def __init__(self, maxsize: int = QUEUE_MAXSIZE) -> None:
        self._maxsize = max(1, int(maxsize))
        self._queues: dict[asyncio.Queue, str] = {}

    async def subscribe(self, timeframe: str = "1h") -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._queues[queue] = normalize_timeframe(timeframe)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._queues.pop(queue, None)

    def set_timeframe(self, queue: asyncio.Queue, timeframe: str) -> None:
        if queue in self._queues:
            self._queues[queue] = normalize_timeframe(timeframe)

    def timeframes(self) -> set[str]:
        return set(self._queues.values())

    @property
    def subscribers_count(self) -> int:
        return len(self._queues)

    def enqueue(self, queue: asyncio.Queue, event: dict[str, Any]) -> None:
        if queue in self._queues:
            self._put(event, (queue,))

    def publish(self, event: dict[str, Any]) -> None:
        self._put(event, tuple(self._queues))

    def _put(self, event: dict[str, Any], queues: tuple[asyncio.Queue, ...]) -> None:
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.debug("dropped stream event for a slow subscriber")


HUB = StreamHub()

_SIDE_TASKS: set[asyncio.Task] = set()
_SIDE_RUNNING: set[str] = set()


def _run_side(name: str, coro: Any) -> asyncio.Task | None:
    """Run a slow provider task without blocking the tick/candle loop."""
    if name in _SIDE_RUNNING:
        return None
    _SIDE_RUNNING.add(name)
    task = asyncio.create_task(coro, name=f"goldsight-{name}")
    _SIDE_TASKS.add(task)

    def done(finished: asyncio.Task) -> None:
        _SIDE_RUNNING.discard(name)
        _SIDE_TASKS.discard(finished)
        if not finished.cancelled():
            exc = finished.exception()
            if exc:
                logger.warning("stream side task %s failed: %s", name, exc)

    task.add_done_callback(done)
    return task


async def _cancel_side_tasks() -> None:
    for task in list(_SIDE_TASKS):
        task.cancel()
    if _SIDE_TASKS:
        await asyncio.gather(*_SIDE_TASKS, return_exceptions=True)
    _SIDE_TASKS.clear()
    _SIDE_RUNNING.clear()


@dataclass
class _Candle:
    time: int
    values: tuple[float, float, float, float, float]
    payload: dict[str, Any]


@dataclass
class _CandleState:
    time: int | None = None
    values: tuple[float, float, float, float, float] | None = None


_CANDLE_STATES: dict[str, _CandleState] = {}


def _candle_payload(time_: int, values: tuple[float, ...]) -> dict[str, Any]:
    return {
        "time": int(time_),
        "open": float(values[0]),
        "high": float(values[1]),
        "low": float(values[2]),
        "close": float(values[3]),
        "volume": float(values[4]),
    }


def _last_candle(frame: pd.DataFrame | None) -> _Candle | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        row = frame.iloc[-1]
        stamp = pd.Timestamp(frame.index[-1])
        time_ = int(stamp.timestamp())
        values = tuple(float(_number(row.get(col), 0.0) or 0.0)
                       for col in ("Open", "High", "Low", "Close", "Volume"))
        return _Candle(time_, values, _candle_payload(time_, values))
    except (KeyError, TypeError, ValueError):
        return None


def candle_events(
    timeframe: str,
    frame: pd.DataFrame | None,
    states: dict[str, _CandleState] | None = None,
) -> list[dict[str, Any]]:
    """Return update/close events for the forming candle of ``timeframe``."""
    states = _CANDLE_STATES if states is None else states
    candle = _last_candle(frame)
    if candle is None:
        return []
    state = states.setdefault(timeframe, _CandleState())
    if state.time is None:
        state.time = candle.time
        state.values = candle.values
        return [event("candle_update", timeframe=timeframe, candle=candle.payload)]
    if candle.time != state.time:
        closed = _candle_payload(state.time, state.values or candle.values)
        state.time = candle.time
        state.values = candle.values
        return [
            event("candle_close", timeframe=timeframe, candle=closed),
            event("candle_update", timeframe=timeframe, candle=candle.payload),
        ]
    if candle.values != state.values:
        state.values = candle.values
        return [event("candle_update", timeframe=timeframe, candle=candle.payload)]
    return []


_RESAMPLE_RULES = {"1h": "1h", "4h": "4h", "1D": "1D"}
_RESAMPLE_AGG = {
    "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
}


def resample_frame(frame: pd.DataFrame | None, timeframe: str) -> pd.DataFrame | None:
    """Resample real 15m candles into a causal forming candle for a timeframe."""
    if frame is None or getattr(frame, "empty", True):
        return frame
    rule = _RESAMPLE_RULES.get(timeframe)
    if rule is None:
        return frame
    try:
        ordered = frame.sort_index()
        agg = {key: value for key, value in _RESAMPLE_AGG.items()
               if key in ordered.columns}
        result = ordered.resample(rule, label="left", closed="left").agg(agg)
        keys = [key for key in ("Open", "Close") if key in result.columns]
        return result.dropna(subset=keys) if keys else result
    except Exception:  # noqa: BLE001 - keep the shared loop alive
        logger.debug("resample to %s failed", timeframe, exc_info=True)
        return frame


def entry_for(entry: Any, timeframe: str, frame: pd.DataFrame) -> Any:
    """Wrap the resampled frame in a CacheEntry for annotation building."""
    if _RESAMPLE_RULES.get(timeframe) is None:
        return entry
    converted = resample_frame(frame, timeframe)
    return data.CacheEntry(
        df=converted if converted is not None else frame,
        fetched_at=float(getattr(entry, "fetched_at", time.time())),
        stale=bool(getattr(entry, "stale", False)),
        source=getattr(entry, "source", None),
    )


def tick_event(frame: pd.DataFrame | None,
               overview: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a tick from the real overview/candle, or None when price is absent."""
    gold = overview.get("gold") if isinstance(overview, dict) else None
    gold = gold if isinstance(gold, dict) else {}
    price = _number(gold.get("price"))
    previous = _number(gold.get("prev_close"))
    change = _number(gold.get("change"))
    change_pct = _number(gold.get("change_pct"))
    time_str = gold.get("as_of")
    candle = _last_candle(frame)
    if price is None and candle is not None:
        price = candle.values[3]
        if frame is not None and len(frame) >= 2 and "Close" in frame.columns:
            previous = _number(frame["Close"].iloc[-2])
        time_str = time_str or datetime.fromtimestamp(candle.time, timezone.utc).isoformat()
    if price is None:
        return None
    if change is None:
        change = price - previous if previous is not None else 0.0
    if change_pct is None:
        change_pct = change / previous * 100.0 if previous else 0.0
    return event("tick", price=float(price), change=float(change),
                 change_pct=float(change_pct), time=str(time_str or _now_iso()))


def macro_event(overview: dict[str, Any] | None) -> dict[str, Any]:
    correlations = overview.get("correlations") if isinstance(overview, dict) else None
    correlations = correlations if isinstance(correlations, dict) else {}
    macro: dict[str, Any] = {}
    for key in ("DXY", "US10Y", "SPX"):
        info = correlations.get(key)
        info = info if isinstance(info, dict) else {}
        macro[key] = {
            "price": _number(info.get("price")),
            "change_pct": _number(info.get("change_pct")),
        }
    return event("macro", **macro)


def news_event(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    items = payload.get("items")
    return event(
        "news",
        items=items if isinstance(items, list) else [],
        sentiment=float(_number(payload.get("sentiment"), 0.0) or 0.0),
        label=str(payload.get("sentiment_label") or "neutral"),
    )


def parse_signal(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a stored signal row and its JSON explanation."""
    explanation = row.get("explanation")
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except (TypeError, ValueError):
            explanation = {"summary": explanation}
    if not isinstance(explanation, dict):
        explanation = {"summary": str(explanation)} if explanation is not None else None
    return {
        "id": int(_number(row.get("id"), 0.0) or 0),
        "timeframe": str(row.get("timeframe") or ""),
        "action": str(row.get("action") or "HOLD").upper(),
        "grade": str(row.get("grade") or (explanation or {}).get("grade") or ""),
        "confidence": float(_number(row.get("confidence"), 0.0) or 0.0),
        "price": float(_number(row.get("price"), 0.0) or 0.0),
        "entry": float(_number(row.get("entry"), 0.0) or 0.0),
        "stop_loss": float(_number(row.get("stop_loss"), 0.0) or 0.0),
        "take_profit1": float(_number(row.get("take_profit1"), 0.0) or 0.0),
        "take_profit2": float(_number(row.get("take_profit2"), 0.0) or 0.0),
        "take_profit3": _number(row.get("take_profit3")),
        "signal_time": str(row.get("signal_time") or row.get("created_at") or ""),
        "status": str(row.get("status") or "open"),
        "explanation": explanation,
    }


def _annotations_hash(annotations: list[dict[str, Any]]) -> str:
    try:
        raw = json.dumps(annotations, sort_keys=True, default=str)
    except (TypeError, ValueError):
        raw = repr(annotations)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


_ANNOTATION_HASHES: dict[str, str] = {}


def annotation_event(timeframe: str, entry: Any) -> dict[str, Any] | None:
    """Emit annotations only when their serialised hash actually changed."""
    try:
        payload = chart_annotations.build_payload(entry, timeframe)
    except Exception:  # noqa: BLE001 - detection must not kill the stream
        logger.debug("annotation build failed for %s", timeframe, exc_info=True)
        return None
    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if not isinstance(annotations, list):
        return None
    digest = _annotations_hash(annotations)
    if _ANNOTATION_HASHES.get(timeframe) == digest:
        return None
    _ANNOTATION_HASHES[timeframe] = digest
    return event("annotations", timeframe=timeframe, annotations=annotations)


_LAST_STATUS: tuple[str, str | None, str | None] | None = None


def status_event(state: str, last_update: str | None = None,
                 source: str | None = None) -> dict[str, Any] | None:
    global _LAST_STATUS
    key = (state, last_update, source)
    if _LAST_STATUS == key:
        return None
    _LAST_STATUS = key
    return event("status", state=state, last_update=last_update, source=source)


def _emit_status(state: str, last_update: str | None = None,
                 source: str | None = None) -> None:
    status = status_event(state, last_update, source)
    if status is not None:
        HUB.publish(status)


_SEEN_SIGNALS: set[int] = set()
_SEEN_RESULTS: dict[int, tuple[Any, ...]] = {}


def publish_signal_events(rows: Any) -> None:
    """Emit new graded BUY/SELL signals and result transitions by id."""
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        signal_id = int(_number(row.get("id"), 0.0) or 0)
        if signal_id <= 0:
            continue
        result_key = (str(row.get("status") or ""), row.get("hit"), row.get("result_pct"))
        previous = _SEEN_RESULTS.get(signal_id)
        _SEEN_RESULTS[signal_id] = result_key
        if previous is not None and previous != result_key:
            HUB.publish(event("signal_result", id=signal_id, status=result_key[0],
                              hit=result_key[1], result_pct=result_key[2]))
        if signal_id in _SEEN_SIGNALS:
            continue
        _SEEN_SIGNALS.add(signal_id)
        if str(row.get("action") or "").upper() not in ("BUY", "SELL"):
            continue
        HUB.publish(event("signal", signal=parse_signal(row)))


_last_force_at = 0.0
_last_overview_at = 0.0
_last_news_at = 0.0
_last_annotations_at = 0.0
_overview_cache: dict[str, Any] | None = None
_task: asyncio.Task | None = None


async def _background_overview() -> None:
    global _overview_cache
    try:
        overview = await asyncio.to_thread(data.get_market_overview)
        _overview_cache = overview
        HUB.publish(macro_event(overview))
    except Exception as exc:  # noqa: BLE001
        logger.warning("stream macro fetch failed: %s", exc)
        _emit_status("stale")


async def _background_news() -> None:
    try:
        payload = await asyncio.to_thread(news.fetch_news)
        HUB.publish(news_event(payload))
    except Exception as exc:  # noqa: BLE001
        logger.warning("stream news fetch failed: %s", exc)


async def _background_annotations(entry: Any, frame: Any) -> None:
    for timeframe in sorted(HUB.timeframes()):
        try:
            annotations = await asyncio.to_thread(
                annotation_event, timeframe, entry_for(entry, timeframe, frame))
            if annotations is not None:
                HUB.publish(annotations)
        except Exception as exc:  # noqa: BLE001
            logger.warning("stream annotation build failed: %s", exc)


async def stream_cycle(background: bool = False) -> None:
    """Fetch real data once, then broadcast derived events to all subscribers.

    The live loop passes ``background=True`` so slow macro/news/annotation
    providers never delay ticks. Direct callers (tests, diagnostics) get all
    events in order by awaiting those side tasks.
    """
    global _last_force_at, _last_overview_at, _last_news_at, _last_annotations_at, _overview_cache
    started = time.monotonic()
    side_tasks: list[asyncio.Task] = []
    entry: Any = None
    force = (started - _last_force_at) >= FORCE_REFRESH_SECONDS
    try:
        if force:
            entry = await asyncio.to_thread(
                data.get_ohlcv, config.GOLD_TICKER, "15m", True)
            _last_force_at = time.monotonic()
        else:
            entry = await asyncio.to_thread(
                data.get_ohlcv, config.GOLD_TICKER, "15m")
    except Exception:  # noqa: BLE001 - provider errors must not stop the loop
        logger.warning("stream OHLCV fetch failed", exc_info=True)
        _emit_status("stale")
    frame = getattr(entry, "df", None) if entry is not None else None
    if frame is None or getattr(frame, "empty", True):
        _emit_status("stale", None, getattr(entry, "error", None))
    else:
        _emit_status("live", getattr(entry, "last_updated", None),
                     getattr(entry, "source", None))
        tick = tick_event(frame, _overview_cache)
        if tick is not None:
            HUB.publish(tick)
        for timeframe in SUPPORTED_TIMEFRAMES:
            for candle_event_ in candle_events(
                    timeframe, resample_frame(frame, timeframe)):
                HUB.publish(candle_event_)
    if started - _last_overview_at >= MACRO_INTERVAL_SECONDS:
        _last_overview_at = started
        task = _run_side("macro", _background_overview())
        if task:
            side_tasks.append(task)
    if started - _last_news_at >= NEWS_INTERVAL_SECONDS:
        _last_news_at = started
        task = _run_side("news", _background_news())
        if task:
            side_tasks.append(task)
    if (frame is not None and not getattr(frame, "empty", True)
            and started - _last_annotations_at >= ANNOTATION_INTERVAL_SECONDS):
        _last_annotations_at = started
        task = _run_side("annotations", _background_annotations(entry, frame))
        if task:
            side_tasks.append(task)
    try:
        rows = await asyncio.to_thread(db.list_signals, 20)
    except Exception:  # noqa: BLE001
        logger.warning("stream signal history failed", exc_info=True)
        rows = []
    publish_signal_events(rows)
    if side_tasks and not background:
        await asyncio.gather(*side_tasks, return_exceptions=True)


async def stream_loop() -> None:
    """Run the shared fetch/broadcast loop until cancelled."""
    _emit_status("reconnecting")
    while True:
        try:
            await stream_cycle(background=True)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - keep the live feed alive
            logger.exception("stream cycle failed")
            _emit_status("stale")
        await asyncio.sleep(STREAM_INTERVAL_SECONDS)


async def start_streaming() -> asyncio.Task | None:
    """Start the single background stream task (idempotent)."""
    global _task
    if _task is not None and not _task.done():
        return _task
    _task = asyncio.create_task(stream_loop(), name="goldsight-stream")
    return _task


async def stop_streaming() -> None:
    """Cancel the background stream task (idempotent)."""
    global _task
    task = _task
    _task = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001
        logger.warning("stream task stopped with an error", exc_info=True)
    await _cancel_side_tasks()
