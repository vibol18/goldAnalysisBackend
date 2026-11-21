"""Risk limits and automatic trade-journal synchronisation."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app import config
from app.services import db

logger = logging.getLogger(__name__)


def _utc_date(value: Any) -> Any:
    try:
        return datetime.fromisoformat(str(value)).date()
    except (ValueError, TypeError):
        return None


def daily_status() -> dict[str, Any]:
    """Realised today's result and whether the configured loss limit is breached."""
    rows = db.list_signals(limit=2000)
    today = datetime.now(timezone.utc).date()
    current = [r for r in rows
               if r.get("status") == "closed"
               and r.get("result_pct") is not None
               and _utc_date(r.get("closed_at")) == today]
    realised = sum(float(r["result_pct"]) for r in current)
    losses = sum(min(0.0, float(r["result_pct"])) for r in current)
    limit = float(config.DAILY_LOSS_LIMIT_PCT)
    breached = realised <= -limit or losses <= -limit
    near = (not breached) and realised <= -limit * 0.7
    if breached:
        warning = (f"Daily loss limit hit: {realised:.2f}% realised against a "
                   f"{limit:.1f}% limit. Stop trading today.")
    elif near:
        warning = f"Approaching daily loss limit: {realised:.2f}% of {limit:.1f}%."
    else:
        warning = ""
    return {
        "date": today.isoformat(),
        "realised_pct": round(realised, 3),
        "gross_loss_pct": round(losses, 3),
        "limit_pct": limit,
        "trades": len(current),
        "remaining_pct": round(-limit - realised, 3),
        "breached": breached,
        "near_limit": near,
        "warning": warning,
        "source": "closed signals recorded today (UTC)",
        "disclaimer": (
            "Educational analysis only. Not financial advice. Signals can be wrong. "
            "Never risk money you can't afford to lose."
        ),
    }


def sync_journal() -> dict[str, Any]:
    """Copy every resolved signal into the trade journal exactly once."""
    rows = [r for r in db.list_signals(limit=2000)
            if r.get("status") == "closed" and r.get("result_pct") is not None]
    created = sum(1 for row in rows if db.ensure_journal_entry(row))
    return {"scanned": len(rows), "created": created, "total": len(db.list_journal(2000))}
