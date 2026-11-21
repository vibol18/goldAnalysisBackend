"""Risk-limit and automatic journal tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config
from app.api import misc
from app.services import db, risk


def resolved_signal(signal_id: int, result_pct: float, days_ago: int = 0):
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "id": signal_id, "created_at": when.isoformat(), "closed_at": when.isoformat(),
        "timeframe": "1h", "action": "BUY", "grade": "A", "entry": 2400.0,
        "stop_loss": 2390.0, "take_profit2": 2420.0, "result_pct": result_pct,
        "setup": "Order Block", "session": "London", "status": "closed",
    }


def test_daily_loss_warning(monkeypatch):
    rows = [resolved_signal(1, -2.0), resolved_signal(2, -1.5),
            resolved_signal(3, 2.0, days_ago=2)]
    monkeypatch.setattr(db, "list_signals", lambda limit=200, offset=0: rows)
    monkeypatch.setattr(config, "DAILY_LOSS_LIMIT_PCT", 3.0)
    status = risk.daily_status()
    assert status["trades"] == 2
    assert status["realised_pct"] == -3.5
    assert status["breached"] is True
    assert "limit" in status["warning"].lower()


def test_auto_journal_imports_once(monkeypatch):
    stored = []
    monkeypatch.setattr(db, "list_signals", lambda limit=200, offset=0: [
        resolved_signal(1, 2.0), resolved_signal(2, -1.0, days_ago=1)])
    monkeypatch.setattr(db, "list_journal", lambda limit=200: stored)
    monkeypatch.setattr(db, "ensure_journal_entry", lambda row: stored.append(row) or True)
    out = risk.sync_journal()
    assert out["scanned"] == 2 and out["created"] == 2


def test_risk_endpoints(monkeypatch):
    monkeypatch.setattr(risk, "daily_status", lambda: {
        "warning": "test warning", "breached": True, "near_limit": False,
        "realised_pct": -3.5, "limit_pct": 3.0,
    })
    monkeypatch.setattr(risk, "sync_journal", lambda: {"created": 0})
    monkeypatch.setattr(db, "list_journal", lambda limit=200, user_id=None: [])
    app = FastAPI()
    app.include_router(misc.router)
    with TestClient(app) as client:
        daily = client.get("/api/risk/daily").json()
        journal = client.get("/api/journal").json()
    assert daily["breached"] is True
    assert journal["trades"] == []
    assert journal["disclaimer"]
