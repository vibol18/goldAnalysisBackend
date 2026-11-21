"""API smoke tests for the concept endpoints (data layer is stubbed, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import concept_context, data


def frame(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    close = 2000 * np.cumprod(1 + rng.normal(0.0004, 0.006, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "Open": open_, "High": np.maximum(open_, close) * 1.004,
        "Low": np.minimum(open_, close) * 0.996, "Close": close,
        "Volume": rng.integers(500, 40000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))


@pytest.fixture()
def client(monkeypatch, tmp_path):
    gold = frame()

    def fake_get_ohlcv(ticker, timeframe, force=False):
        if ticker.startswith("XAU") and timeframe == "1h":
            return data.CacheEntry(df=gold, fetched_at=0.0, source="test")
        if ticker.startswith("XAU"):
            return data.CacheEntry(df=gold.resample("1D").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"}).dropna(),
                fetched_at=0.0, source="test")
        return data.CacheEntry(df=pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]),
            fetched_at=0.0, stale=True)

    monkeypatch.setattr(data, "get_ohlcv", fake_get_ohlcv)
    monkeypatch.setattr(data, "get_market_overview", lambda: {
        "gold": {}, "correlations": {"DXY": {"change_pct": -0.2}}, "meta": {}})
    monkeypatch.setattr(concept_context, "EVENTS_PATH", tmp_path / "events.json")
    with TestClient(app) as c:
        yield c


def test_concepts_endpoint(client):
    r = client.get("/api/concepts?timeframe=1h")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "XAU/USD (Gold)"
    assert body["count"] > 0
    assert body["concepts"][0]["type"]
    assert "disclaimer" in body


def test_confluence_endpoint(client):
    r = client.get("/api/confluence?timeframe=1h")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["confluence"]["action"] in ("BUY", "SELL", "HOLD")
    assert body["confluence"]["grade"] in ("A+", "A", "B", "C", "—")
    assert body["explanation"]["narrative"]


def test_learn_and_events(client):
    learn = client.get("/api/learn").json()
    assert learn["total"] >= 80
    assert any(c["type"] == "fvg" for c in learn["concepts"])
    assert client.get("/api/events").json()["events"] == []
    client.get("/api/auth/csrf")
    csrf = client.cookies.get("gs_csrf")
    saved = client.post("/api/events", json={"events": [
        {"time": "2030-01-01T13:30:00Z", "impact": "high", "name": "NFP"}]},
        headers={"X-CSRF-Token": csrf or ""})
    assert saved.status_code == 200, saved.text
    assert client.get("/api/events").json()["events"][0]["name"] == "NFP"


def test_weights_endpoint(client):
    body = client.get("/api/weights").json()
    assert body["weights"]["fvg"] > 0
    assert "category_caps" in body
