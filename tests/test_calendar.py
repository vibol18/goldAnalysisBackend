"""Tests for the editable economic calendar and event signal filter."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import concepts as concepts_api
from app.api import misc
from app.concepts.base import make_concept
from app.services import concept_context
from app.services.confluence import confluence
from app.intelligence import data_sources as ds


def test_normalise_event_validates_and_keeps_forecast_actual():
    out = concept_context.normalise_event({
        "title": "Nonfarm Payrolls", "time": "2026-01-02T13:30:00Z",
        "impact": "HIGH", "forecast": "180K", "actual": "200K", "country": "USD",
    })
    assert out["impact"] == "high"
    assert out["forecast"] == "180K" and out["actual"] == "200K"
    assert out["source"] == "user-edited"
    with pytest.raises(ValueError):
        concept_context.normalise_event({"title": "", "time": "2026-01-02T13:30:00Z"})
    with pytest.raises(ValueError):
        concept_context.normalise_event({"title": "CPI", "time": "not-a-date"})


def test_events_endpoint_persists_calendar(tmp_path, monkeypatch):
    monkeypatch.setattr(concept_context, "EVENTS_PATH", tmp_path / "events.json")
    app = FastAPI()
    app.include_router(concepts_api.router)
    now = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    with TestClient(app) as client:
        r = client.post("/api/events", json={"events": [
            {"title": "CPI", "time": now, "impact": "high", "forecast": "3.1%"},
        ]})
        assert r.status_code == 200, r.text
        assert r.json()["events"][0]["title"] == "CPI"
        assert client.get("/api/events").json()["events"][0]["forecast"] == "3.1%"


def test_high_impact_event_blocks_signal():
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(5)
    close = 2000 + np.linspace(0, 40, 200) + rng.normal(0, 2.0, 200)
    df = pd.DataFrame({
        "Open": close - 0.5, "High": close + 2, "Low": close - 2,
        "Close": close, "Volume": np.full(200, 1000.0),
    }, index=idx)
    base = [
        make_concept("trend", "bullish", str(idx[-1]), str(idx[-1]), None, None,
                     90, "active", "trend", "structure", {}),
        make_concept("mtf_bias", "bullish", str(idx[-1]), str(idx[-1]), None, None,
                     90, "active", "bias", "mtf", {"htf_bias": "bullish"}),
    ]
    assert confluence(base, df).action == "BUY"
    event = make_concept(
        "event_risk", "neutral", str(idx[-1]), str(idx[-1]), None, None,
        95, "active", "high-impact event", "macro",
        {"hours_away": 2.0, "event_name": "CPI"},
    )
    out = confluence(base + [event], df)
    assert out.action == "HOLD"
    assert any("event" in v for v in out.veto_reasons)


def test_calendar_endpoint_exposes_countdown_and_blocking(monkeypatch):
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    monkeypatch.setattr(ds, "fetch_calendar_events", lambda: [
        {"title": "FOMC", "time": soon, "impact": "high", "forecast": "5.50%"},
    ])
    app = FastAPI()
    app.include_router(misc.router)
    with TestClient(app) as client:
        r = client.get("/api/calendar")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["blocking_now"] is True
    assert body["blocking"]["name"] == "FOMC"
    assert body["countdown"]["available"] is True
    assert body["disclaimer"]
