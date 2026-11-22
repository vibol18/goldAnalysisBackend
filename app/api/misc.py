"""Backtest, chat, risk, news and settings endpoints."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app import config
from app.services import auth
from app.services import chat, data, db, news, price_service, risk as risk_service, stats as stats_service
from app.services.backtest import run_backtest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["backtest", "chat", "risk", "news"])


@router.get("/debug/price")
def debug_price() -> dict[str, Any]:
    """Latest price, every configured source and the exact bridge diagnosis."""
    payload = price_service.diagnostics()
    payload["verification"] = price_service.verification()
    return payload


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@router.get("/backtest")
def backtest(years: int = config.BACKTEST_YEARS) -> dict[str, Any]:
    """Run (or serve cached) multi-year backtest of the strategy."""
    try:
        gold = price_service.get_series("1D").df
        dxy = data.get_ohlcv(config.DXY_TICKER, "1D").df
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"market data unavailable: {exc}") from exc
    try:
        return run_backtest(gold, dxy, years=max(1, min(years, 10)))
    except Exception as exc:  # noqa: BLE001
        logger.exception("backtest failed")
        raise HTTPException(500, f"backtest failed: {exc}") from exc


@router.get("/stats/results")
def stats_results(years: int = config.BACKTEST_YEARS) -> dict[str, Any]:
    """Backtest / paper / live results side by side, split by setup and session."""
    return stats_service.build_results(max(1, min(years, 10)))


@router.get("/calendar")
def calendar_endpoint() -> dict[str, Any]:
    """Editable events, merged feed, countdown and current signal-blocking state."""
    from app.intelligence import data_sources as ds
    from app.services import concept_context

    events = ds.fetch_calendar_events()
    countdown = ds.calendar_countdown(events)
    now = ds._now()
    from datetime import datetime, timezone
    blocking: dict[str, Any] | None = None
    try:
        current = datetime.now(timezone.utc)
        for event in events:
            if str(event.get("impact", "")).lower() != "high":
                continue
            when = pd_to_datetime(event.get("time"))
            if when is None:
                continue
            delta_h = abs((when - current).total_seconds()) / 3600.0
            if delta_h <= 12.0 and (blocking is None or delta_h < blocking["hours_away"]):
                blocking = {
                    "name": event.get("title") or event.get("name"),
                    "time": event.get("time"),
                    "hours_away": round(delta_h, 2),
                }
    except Exception as exc:  # noqa: BLE001
        logger.debug("calendar blocking check failed: %s", exc)
    return {
        "events": events,
        "editable_events": concept_context.load_events(),
        "countdown": countdown,
        "blocking": blocking,
        "blocking_now": blocking is not None,
        "block_window_hours": 12.0,
        "news_filter": "high-impact events block signals within the window",
        "checked_at": now,
        "disclaimer": (
            "Educational analysis only. Not financial advice. Signals can be wrong. "
            "Never risk money you can't afford to lose."
        ),
    }


def pd_to_datetime(value: Any):
    import pandas as pd
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(ts) else ts.to_pydatetime()
    except (ValueError, TypeError):
        return None


@router.get("/risk/daily")
def risk_daily() -> dict[str, Any]:
    """Today's realised result and daily loss-limit status."""
    return risk_service.daily_status()


@router.get("/journal")
def journal_list(limit: int = 200, user: dict | None = Depends(auth.optional_user)) -> dict[str, Any]:
    """Trade journal; resolved signals are imported automatically on read."""
    uid = (user or {}).get("id")
    risk_service.sync_journal()
    trades = (
        db.list_journal(min(limit, 1000))
        if uid is None
        else db.list_journal(min(limit, 1000), user_id=uid)
    )
    return {
        "trades": trades,
        "daily": risk_service.daily_status(),
        "disclaimer": (
            "Educational analysis only. Not financial advice. Signals can be wrong. "
            "Never risk money you can't afford to lose."
        ),
    }


@router.post("/journal")
def journal_create(body: dict[str, Any], user: dict | None = Depends(auth.optional_user)) -> dict[str, Any]:
    """Log a manual trade in the journal."""
    try:
        result_pct = body.get("result_pct")
        if result_pct is not None:
            result_pct = float(result_pct)
        entry_id = db.save_journal_entry({**body, "result_pct": result_pct}, user_id=(user or {}).get("id"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"invalid journal entry: {exc}") from exc
    return {"id": entry_id, "status": "saved"}


@router.delete("/journal/{entry_id}")
def journal_delete(entry_id: int, user: dict | None = Depends(auth.optional_user)) -> dict[str, Any]:
    return {"deleted": db.delete_journal_entry(entry_id, user_id=(user or {}).get("id"))}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def _build_snapshot() -> dict[str, Any]:
    """Live context injected into every LLM call (fresh price + latest candle)."""
    overview = data.get_market_overview()
    latest_candle: dict[str, Any] | None = None
    try:
        fresh = data.refresh_realtime_candles()
        if fresh is not None and not fresh.df.empty:
            last = fresh.df.iloc[-1]
            latest_candle = {
                "time": str(fresh.df.index[-1]),
                "open": float(last["Open"]),
                "high": float(last["High"]),
                "low": float(last["Low"]),
                "close": float(last["Close"]),
                "volume": float(last.get("Volume", 0) or 0),
                "stale": fresh.stale,
                "source": fresh.source,
            }
            overview["gold"]["price"] = latest_candle["close"]
            overview["gold"]["as_of"] = latest_candle["time"]
            overview["meta"]["source"] = fresh.source
            overview["meta"]["stale"] = fresh.stale
    except Exception:  # noqa: BLE001
        logger.warning("fresh chat candle fetch failed", exc_info=True)
    sig = None
    concept_payload: dict[str, Any] = {
        "concepts": [], "confluence": None, "signal_explanation": None,
    }
    try:
        entry = price_service.get_series("1h")
        from app.services.concept_context import build_context
        from app.services.signal_engine import generate_signal
        context = build_context("1h")
        dxy = overview.get("correlations", {}).get("DXY", {})
        s = generate_signal(entry.df, "1h",
                            dxy_change_pct=dxy.get("change_pct") if isinstance(dxy, dict) else None,
                            train_model=False, concept_context=context)
        sig = s.to_dict()
        concept_payload = {
            "concepts": sig.get("concepts", [])[:60],
            "confluence": sig.get("confluence"),
            "signal_explanation": (sig.get("explanation") or {}).get("narrative"),
        }
    except Exception:  # noqa: BLE001
        logger.warning("concept snapshot for chat failed", exc_info=True)
    try:
        ind = None
        from app.services.indicators import add_all
        df = add_all(price_service.get_series("1D").df)
        last = df.iloc[-1]
        ind = {
            "rsi": float(last["rsi"]) if last.get("rsi") == last.get("rsi") else None,
            "macd": float(last["macd"]),
            "macd_signal": float(last["macd_signal"]),
            "ema20": float(last["ema20"]),
            "ema50": float(last["ema50"]),
            "ema200": float(last["ema200"]),
            "atr": float(last["atr"]),
            "support": float(last["support"]) if last.get("support") == last.get("support") else None,
            "resistance": float(last["resistance"]) if last.get("resistance") == last.get("resistance") else None,
        }
    except Exception:  # noqa: BLE001
        ind = None
    n = news.fetch_news()
    return {
        "price": latest_candle["close"] if latest_candle else overview["gold"].get("price"),
        "latest_candle": latest_candle,
        "market": overview,
        "indicators_daily": ind,
        "current_signal": sig,
        "concepts": concept_payload["concepts"],
        "confluence": concept_payload["confluence"],
        "signal_explanation": concept_payload["signal_explanation"],
        "recent_signals": [
            {
                "id": s.get("id"), "action": s.get("action"),
                "grade": (s.get("explanation") or {}).get("grade"),
                "entry": s.get("entry"), "stop_loss": s.get("stop_loss"),
                "take_profit1": s.get("take_profit1"),
                "take_profit2": s.get("take_profit2"),
                "signal_time": s.get("signal_time"), "status": s.get("status"),
                "hit": s.get("hit"),
                "why": ((s.get("explanation") or {}).get("narrative") or "")[:600],
            }
            for s in db.list_signals(limit=5)
        ],
        "news": {"sentiment": n.get("sentiment"), "label": n.get("sentiment_label"),
                 "headlines": [i["title"] for i in n.get("items", [])[:6]]},
        "disclaimer": ("Educational analysis only. Not financial advice. "
                       "Trading gold is risky and signals can be wrong."),
    }


@router.get("/chat/history")
def chat_history(user: dict | None = Depends(auth.optional_user)) -> dict[str, Any]:
    """Persisted chat messages + provider configuration status."""
    return {
        "messages": db.list_chat(limit=200, user_id=(user or {}).get("id")),
        "llm_configured": config.llm_configured(),
        "provider": config.LLM_PROVIDER if config.llm_configured() else None,
    }


@router.delete("/chat/history")
def chat_clear(user: dict | None = Depends(auth.optional_user)) -> dict[str, str]:
    db.clear_chat(user_id=(user or {}).get("id"))
    return {"status": "cleared"}


@router.post("/chat")
async def chat_post(
    body: dict[str, Any],
    request: Request,
    user: dict | None = Depends(auth.optional_user),
) -> StreamingResponse:
    """Stream an assistant reply (SSE) with live context injected."""
    if not config.origin_is_allowed(request.headers.get("origin")):
        raise HTTPException(403, "Origin not allowed")
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "message is required")
    if len(message) > 4000:
        raise HTTPException(400, "message too long (max 4000 chars)")

    uid = (user or {}).get("id")
    history = db.list_chat(limit=20, user_id=uid)
    snapshot = await run_in_threadpool(_build_snapshot)
    db.save_chat("user", message, user_id=uid)

    async def event_stream() -> AsyncIterator[str]:
        yield "event: meta\ndata: " + json.dumps(
            {"llm_configured": config.llm_configured(),
             "price": snapshot.get("price"),
             "candle": snapshot.get("latest_candle")}) + "\n\n"
        parts: list[str] = []
        try:
            async for delta in chat.stream_chat(message, snapshot, history):
                parts.append(delta)
                yield "data: " + json.dumps({"delta": delta}) + "\n\n"
        except RuntimeError as exc:
            yield "event: error\ndata: " + json.dumps({"error": str(exc)}) + "\n\n"
            if not parts:
                fallback = (
                    "AI chat is not configured. Set LLM_PROVIDER and LLM_API_KEY "
                    "in backend/.env (see .env.example). Meanwhile, here is the "
                    "live context I would have used:\n"
                    + json.dumps(snapshot, default=str, ensure_ascii=False)[:1500]
                )
                db.save_chat("assistant", fallback, user_id=uid)
                yield "data: " + json.dumps({"delta": fallback}) + "\n\n"
                yield "event: done\ndata: {}\n\n"
                return
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream failed")
            yield "event: error\ndata: " + json.dumps(
                {"error": f"LLM request failed: {exc}"}) + "\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        full = "".join(parts)
        if full:
            db.save_chat("assistant", full, user_id=uid)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Risk tools
# ---------------------------------------------------------------------------
@router.post("/risk/position-size")
def position_size(body: dict[str, Any]) -> dict[str, Any]:
    """Position size = risk amount / stop distance (units of gold)."""
    try:
        account = float(body["account_size"])
        risk_pct = float(body["risk_pct"])
        stop_distance = float(body["stop_distance"])
        price = float(body.get("price") or 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "account_size, risk_pct, stop_distance required") from exc
    if account <= 0 or risk_pct <= 0 or stop_distance <= 0:
        raise HTTPException(400, "all inputs must be positive")
    risk_amount = account * risk_pct / 100.0
    units = risk_amount / stop_distance
    notional = units * price if price else None
    return {
        "risk_amount": round(risk_amount, 2),
        "units": round(units, 4),
        "lots_std": round(units / 100.0, 4),  # 1 standard lot = 100 oz
        "notional": round(notional, 2) if notional else None,
        "disclaimer": "Position sizing reduces risk; it cannot eliminate losses.",
    }


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
@router.get("/news")
def news_endpoint(force: bool = False) -> dict[str, Any]:
    """Gold news headlines + lexicon sentiment (best-effort)."""
    return news.fetch_news(force=force)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@router.get("/settings")
def settings() -> dict[str, Any]:
    """Non-secret configuration for the Settings page."""
    return {
        "llm_provider": config.LLM_PROVIDER if config.llm_configured() else None,
        "llm_configured": config.llm_configured(),
        "llm_model": config.LLM_MODEL or None,
        "ml_horizon": config.ML_HORIZON,
        "buy_threshold": config.SIGNAL_BUY_THRESHOLD,
        "sell_threshold": config.SIGNAL_SELL_THRESHOLD,
        "spread_usd": config.SPREAD_USD,
        "commission_usd": config.COMMISSION_USD,
        "backtest_years": config.BACKTEST_YEARS,
        "disclaimer": ("Educational analysis only. Not financial advice. "
                       "Trading gold is risky and signals can be wrong. "
                       "Never risk money you can't afford to lose."),
    }


