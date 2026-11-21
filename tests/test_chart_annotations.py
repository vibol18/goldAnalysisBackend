"""Chart annotation endpoint tests (all data/network layers are stubbed)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chart
from app.services import concept_context, data, db

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Signals can be wrong. "
    "Never risk money you can't afford to lose."
)

REQUIRED_KEYS = {
    "timeframe", "symbol", "annotations", "signal_markers",
    "total", "stale", "error", "disclaimer",
}

ANNOTATION_KEYS = {
    "id", "type", "category", "direction", "status", "label", "top", "bottom",
    "price", "start_time", "end_time", "extend_right", "strength", "notes", "meta",
}

MARKER_KEYS = {
    "id", "time", "action", "grade", "price", "status", "hit", "entry",
    "stop_loss", "take_profit1", "take_profit2", "take_profit3",
    "result_pct", "explanation",
}


def make_frame(n: int = 320) -> pd.DataFrame:
    """Deterministic OHLC frame with injected FVG, equal highs and a sweep."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    rng = np.random.default_rng(7)
    close = (2400 + 18 * np.sin(2 * np.pi * t / 24)
             + 7 * np.sin(2 * np.pi * t / 7) + 0.12 * t
             + rng.normal(0, 0.8, n))
    op = np.roll(close, 1)
    op[0] = close[0]
    hi = np.maximum(op, close) + 4 + 2 * np.abs(np.sin(t / 3))
    lo = np.minimum(op, close) - 4 - 2 * np.abs(np.cos(t / 4))
    frame = pd.DataFrame({
        "Open": op, "High": hi, "Low": lo, "Close": close,
        "Volume": rng.integers(500, 40000, n).astype(float),
    }, index=idx)

    col = {name: i for i, name in enumerate(frame.columns)}
    frame.iloc[118, col["High"]] = 2420.0
    frame.iloc[120, col["Low"]] = 2430.0
    frame.iloc[120, col["Close"]] = 2440.0
    frame.iloc[120, col["High"]] = 2445.0

    for bar in (60, 65):
        frame.iloc[bar, col["High"]] = 2500.0
        frame.iloc[bar, col["Close"]] = 2490.0
        frame.iloc[bar, col["Open"]] = 2485.0
        frame.iloc[bar, col["Low"]] = 2480.0
    for bar in (59, 61, 64, 66):
        frame.iloc[bar, col["High"]] = 2470.0

    frame.iloc[80, col["High"]] = 2520.0
    frame.iloc[80, col["Close"]] = 2485.0
    frame.iloc[80, col["Open"]] = 2495.0
    frame.iloc[80, col["Low"]] = 2480.0
    return frame


SAMPLE_SIGNAL = {
    "id": 42,
    "timeframe": "1h",
    "created_at": "2024-02-01T12:00:00+00:00",
    "signal_time": "2024-02-01T11:00:00+00:00",
    "action": "BUY",
    "grade": "A",
    "price": 2412.30,
    "status": "closed",
    "hit": "TP",
    "entry": 2410.0,
    "stop_loss": 2400.0,
    "take_profit1": 2420.0,
    "take_profit2": 2430.0,
    "result_pct": 0.83,
    "explanation": json.dumps({"summary": "Bullish confluence", "reasons": ["BOS", "FVG"]}),
}

OLD_SIGNAL = {
    "id": 7,
    "timeframe": "1h",
    "created_at": "2024-01-30T09:00:00+00:00",
    "action": "SELL",
    "price": 2380.0,
    "entry": 2380.0,
    "stop_loss": 2390.0,
    "take_profit1": 2360.0,
    "take_profit2": 2340.0,
    "explanation": "plain text explanation",
}


@pytest.fixture()
def client(monkeypatch):
    holder = {"df": make_frame()}

    def fake_get_ohlcv(ticker, timeframe, force=False):
        return data.CacheEntry(df=holder["df"], fetched_at=0.0, source="test")

    monkeypatch.setattr(data, "get_ohlcv", fake_get_ohlcv)
    monkeypatch.setattr(concept_context, "build_context", lambda timeframe: {})
    monkeypatch.setattr(db, "list_signals", lambda limit=200: [SAMPLE_SIGNAL, OLD_SIGNAL])

    app = FastAPI()
    app.include_router(chart.router)
    with TestClient(app) as c:
        yield c, holder


def test_endpoint_shape(client):
    c, _ = client
    r = c.get("/api/chart/annotations?timeframe=1h")
    assert r.status_code == 200, r.text
    body = r.json()
    assert REQUIRED_KEYS.issubset(body)
    assert body["timeframe"] == "1h"
    assert body["symbol"] == "XAU/USD (Gold)"
    assert body["disclaimer"] == DISCLAIMER
    assert isinstance(body["annotations"], list)
    assert isinstance(body["signal_markers"], list)
    assert body["total"] >= body["shown"] == len(body["annotations"])
    assert isinstance(body["stale"], bool)
    assert body["error"] is None


def test_annotation_fields_are_json_safe(client):
    c, _ = client
    body = c.get("/api/chart/annotations?timeframe=1h&show_all=true").json()
    assert body["annotations"]
    for annotation in body["annotations"]:
        assert ANNOTATION_KEYS.issubset(annotation)
        assert isinstance(annotation["id"], str) and annotation["id"]
        assert annotation["direction"] in ("bullish", "bearish", "neutral")
        assert annotation["status"] in ("active", "mitigated", "invalidated")
        assert isinstance(annotation["extend_right"], bool)
        assert isinstance(annotation["strength"], float)
        assert isinstance(annotation["meta"], dict)
        assert isinstance(annotation["label"], str)
        json.dumps(annotation["meta"], allow_nan=False)
        json.dumps(annotation, allow_nan=False)


def test_required_annotation_types(client):
    c, _ = client
    body = c.get("/api/chart/annotations?timeframe=1h&show_all=true").json()
    types = {a["type"] for a in body["annotations"]}
    for required in ("fvg_box", "eqh_line", "bos_line", "sweep_marker"):
        assert required in types, f"missing {required}"
    markers = body["signal_markers"]
    assert markers and all({"entry", "stop_loss", "take_profit1"} <= set(m) for m in markers)


def test_show_all_and_filters(client):
    c, _ = client
    default = c.get("/api/chart/annotations?timeframe=1h").json()
    everything = c.get("/api/chart/annotations?timeframe=1h&show_all=true").json()
    assert everything["total"] >= default["total"]
    assert len(everything["annotations"]) >= len(default["annotations"])

    filtered = c.get("/api/chart/annotations?timeframe=1h&show_all=true&types=fvg_box,bos_line").json()
    assert filtered["annotations"]
    assert {a["type"] for a in filtered["annotations"]} <= {"fvg_box", "bos_line"}

    by_category = c.get("/api/chart/annotations?timeframe=1h&show_all=true&category=smc").json()
    assert by_category["annotations"]
    assert {a["category"] for a in by_category["annotations"]} == {"smc"}


def test_signal_markers(client):
    c, _ = client
    markers = c.get("/api/chart/annotations?timeframe=1h").json()["signal_markers"]
    assert len(markers) == 2
    for marker in markers:
        assert MARKER_KEYS.issubset(marker)
        assert marker["action"] in ("BUY", "SELL", "HOLD")
        assert marker["status"] in ("open", "closed")
        json.dumps(marker, allow_nan=False)

    sample = next(m for m in markers if m["id"] == 42)
    assert sample["time"] == "2024-02-01T11:00:00+00:00"
    assert sample["grade"] == "A"
    assert sample["explanation"] == {"summary": "Bullish confluence", "reasons": ["BOS", "FVG"]}
    assert sample["take_profit3"] is None
    assert sample["result_pct"] == 0.83

    old = next(m for m in markers if m["id"] == 7)
    assert old["grade"] == ""
    assert old["take_profit3"] is None
    assert old["explanation"] == {"summary": "plain text explanation"}
    assert old["time"] == "2024-01-30T09:00:00+00:00"


def test_no_lookahead(client):
    c, holder = client
    before = c.get("/api/chart/annotations?timeframe=1h&show_all=true").json()["annotations"]
    original = holder["df"]
    # Use the last completed day boundary as the edit cutoff: every bar we
    # change lives strictly after it, so annotations stamped before it are
    # guaranteed to be derived only from unedited candles.
    cutoff = original.index[len(original) - 32].isoformat()

    edited = original.copy()
    col = {name: i for i, name in enumerate(edited.columns)}
    for pos in range(len(edited) - 30, len(edited)):
        for name in ("Open", "High", "Low", "Close"):
            edited.iloc[pos, col[name]] = float(edited.iloc[pos, col[name]]) + 1.0
    holder["df"] = edited

    after = c.get("/api/chart/annotations?timeframe=1h&show_all=true").json()["annotations"]
    after_by_id = {a["id"]: a for a in after}

    checked = 0
    for annotation in before:
        if annotation["end_time"] is None or annotation["end_time"] >= cutoff:
            continue
        match = after_by_id.get(annotation["id"])
        assert match is not None, f"past annotation disappeared: {annotation['id']}"
        for field in ("top", "bottom", "price", "start_time"):
            assert match[field] == annotation[field], (
                f"past annotation {annotation['id']} changed {field}: "
                f"{annotation[field]} -> {match[field]}"
            )
        checked += 1
    assert checked > 0


def test_empty_frame_returns_503(monkeypatch):
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    monkeypatch.setattr(data, "get_ohlcv",
                        lambda ticker, timeframe, force=False: data.CacheEntry(
                            df=empty, fetched_at=0.0, stale=True, error="no data"))
    app = FastAPI()
    app.include_router(chart.router)
    with TestClient(app) as c:
        r = c.get("/api/chart/annotations?timeframe=1h")
    assert r.status_code == 503


def test_clean_mode_caps_drawings_and_labels(client):
    c, _ = client
    clean = c.get("/api/chart/annotations?timeframe=1h").json()
    assert 15 <= clean["shown"] <= 25, clean["shown"]
    assert clean["total"] > clean["shown"]
    labels = [a["meta"]["axis_label"] for a in clean["annotations"]]
    assert sum(labels) <= 8
    labelled = [a for a in clean["annotations"] if a["meta"]["axis_label"]]
    prices = sorted((a["price"] or a["top"] or 0) for a in labelled)
    for left, right in zip(prices, prices[1:]):
        assert abs(right - left) > 0, "axis labels must not overlap"


def test_signal_markers_deduplicated(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(db, "list_signals", lambda limit=200: [SAMPLE_SIGNAL, dict(SAMPLE_SIGNAL)])
    markers = c.get("/api/chart/annotations?timeframe=1h").json()["signal_markers"]
    assert len(markers) == 1
