"""Real performance statistics for backtest, paper and live signals.

Every number comes from recorded results: stored signals with an outcome are the
live/paper record; the walk-forward rule backtest supplies the simulated record.
Nothing is extrapolated or filled with placeholder numbers.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.concepts.base import session_of
from app.services import backtest as backtest_service
from app.services import db

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Signals can be wrong. "
    "Never risk money you can't afford to lose."
)
SESSION_LABELS = {
    "asian": "Asian", "london": "London", "new_york": "New York",
    "new_york_pm": "New York", "late": "Off-hours",
}


def session_for(timestamp: Any) -> str:
    """Human-readable UTC trading session for a signal timestamp."""
    return SESSION_LABELS.get(session_of(timestamp), "Off-hours")


def setup_for(payload: dict[str, Any]) -> str:
    """Derive a readable setup label from the signal's contributing concepts."""
    explanation = payload.get("explanation")
    if isinstance(explanation, str):
        import json
        try:
            explanation = json.loads(explanation)
        except (TypeError, ValueError):
            explanation = {}
    if not isinstance(explanation, dict):
        explanation = {}
    confluence = explanation.get("confluence") or {}
    agreeing = confluence.get("agreeing") if isinstance(confluence, dict) else None
    if agreeing:
        return str(agreeing[0]).replace("_", " ").title()
    concepts = payload.get("concepts") or []
    if concepts:
        first = concepts[0]
        if isinstance(first, dict):
            return str(first.get("type", "confluence")).replace("_", " ").title()
    grade = str(payload.get("grade") or "C")
    return f"{grade} confluence"


def annotate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Stamp setup/session/source onto a signal before it is persisted."""
    payload.setdefault("source", "live")
    payload.setdefault("setup", setup_for(payload))
    payload.setdefault("session", session_for(
        payload.get("signal_time") or payload.get("created_at")))
    return payload


def _metrics(results: list[float]) -> dict[str, Any]:
    """Win rate, profit factor, expectancy and drawdown from per-trade returns."""
    clean = [float(r) for r in results if isinstance(r, (int, float)) and np.isfinite(r)]
    if not clean:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0, "max_drawdown_pct": 0.0,
            "total_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        }
    wins = [r for r in clean if r > 0]
    losses = [r for r in clean if r <= 0]
    gross_profit = float(sum(wins))
    gross_loss = abs(float(sum(losses)))
    equity = 100.0 + np.cumsum(clean)
    peaks = np.maximum.accumulate(equity)
    drawdown = float(np.max((peaks - equity) / peaks * 100.0))
    return {
        "trades": len(clean),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(clean) * 100.0, 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (
            999.0 if gross_profit > 0 else 0.0),
        "expectancy": round(float(np.mean(clean)), 4),
        "max_drawdown_pct": round(drawdown, 2),
        "total_pct": round(float(sum(clean)), 2),
        "avg_win": round(gross_profit / len(wins), 4) if wins else 0.0,
        "avg_loss": round(float(sum(losses)) / len(losses), 4) if losses else 0.0,
    }


def _grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        result = row.get("result_pct")
        if result is None:
            continue
        key = str(row.get(field) or "unknown")
        groups.setdefault(key, []).append(float(result))
    return {key: _metrics(values) for key, values in sorted(groups.items())}


def _stored(source: str) -> dict[str, Any]:
    rows = db.list_signals(limit=2000)
    rows = [r for r in rows
            if str(r.get("source") or "live").lower() == source
            and r.get("result_pct") is not None]
    overall = _metrics([float(r["result_pct"]) for r in rows])
    return {
        **overall,
        "by_setup": _grouped(rows, "setup"),
        "by_session": _grouped(rows, "session"),
        "sample_note": ("Based on resolved stored signals only; open signals are excluded."
                        if rows else "No resolved signals recorded yet."),
    }


def build_results(years: int = 5) -> dict[str, Any]:
    """Side-by-side backtest / paper / live results with setup and session splits."""
    try:
        back = backtest_service.run_backtest(_frame(), years=years)
    except Exception as exc:  # noqa: BLE001 - missing data must not break the page
        logger.warning("backtest for stats page failed: %s", exc)
        back = {}
    backtest_block = {
        "trades": int(back.get("trades", 0)),
        "wins": int(back.get("by_side", {}).get("long", {}).get("wins", 0)
                    + back.get("by_side", {}).get("short", {}).get("wins", 0)),
        "losses": max(0, int(back.get("trades", 0)) - int(
            back.get("by_side", {}).get("long", {}).get("wins", 0)
            + back.get("by_side", {}).get("short", {}).get("wins", 0))),
        "win_rate": float(back.get("win_rate", 0.0)),
        "profit_factor": float(back.get("profit_factor", 0.0)),
        "expectancy": round(float(back.get("total_return_pct", 0.0)) /
                            max(int(back.get("trades", 1)), 1), 4),
        "max_drawdown_pct": float(back.get("max_drawdown_pct", 0.0)),
        "total_pct": float(back.get("total_return_pct", 0.0)),
        "by_setup": back.get("by_setup", {}),
        "by_session": back.get("by_session", {}),
        "sample_note": back.get("note", "Backtest unavailable."),
    }
    return {
        "years": years,
        "backtest": backtest_block,
        "paper": _stored("paper"),
        "live": _stored("live"),
        "disclaimer": DISCLAIMER,
    }


def _frame():
    from app import config
    from app.services import data
    return data.get_ohlcv(config.GOLD_TICKER, "1h").df
