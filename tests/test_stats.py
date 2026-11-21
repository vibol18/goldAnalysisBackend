"""Tests for backtest/paper/live performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import misc
from app.services import db, stats


def test_metrics_are_derived_from_results_only():
    m = stats._metrics([2.0, -1.0, 3.0, -1.0])
    assert m["trades"] == 4
    assert m["wins"] == 2 and m["losses"] == 2
    assert m["win_rate"] == 50.0
    assert m["profit_factor"] == 2.5
    assert m["expectancy"] == 0.75
    assert m["max_drawdown_pct"] >= 0
    assert stats._metrics([])["trades"] == 0


def test_annotate_payload_stamps_source_setup_and_session():
    payload = {
        "action": "BUY", "grade": "A",
        "signal_time": "2024-01-01T03:00:00+00:00",
        "explanation": {"confluence": {"agreeing": ["order_block", "fvg"]}},
    }
    out = stats.annotate_payload(payload)
    assert out["source"] == "live"
    assert out["setup"] == "Order Block"
    assert out["session"] == "Asian"


def test_build_results_splits_by_source_setup_and_session(monkeypatch):
    rows = [
        {"result_pct": 2.0, "source": "live", "setup": "FVG", "session": "London"},
        {"result_pct": -1.0, "source": "live", "setup": "FVG", "session": "London"},
        {"result_pct": 3.0, "source": "paper", "setup": "BOS", "session": "New York"},
        {"result_pct": None, "source": "live", "setup": "FVG", "session": "London"},
    ]
    monkeypatch.setattr(db, "list_signals", lambda limit=200, offset=0: rows)
    idx = pd.date_range("2024-01-01", periods=300, freq="h")
    frame = pd.DataFrame({
        "Open": np.full(300, 10.0), "High": np.full(300, 11.0),
        "Low": np.full(300, 9.0), "Close": np.full(300, 10.0),
        "Volume": np.ones(300),
    }, index=idx)
    monkeypatch.setattr(stats, "_frame", lambda: frame)
    monkeypatch.setattr(stats.backtest_service, "run_backtest",
                        lambda df, years=None: {
                            "trades": 10, "win_rate": 60.0, "profit_factor": 1.8,
                            "total_return_pct": 20.0, "max_drawdown_pct": 5.0,
                            "by_side": {"long": {"wins": 6}, "short": {"wins": 0}},
                            "by_setup": {"Fvg": {"trades": 10}},
                            "by_session": {"london": {"trades": 10}},
                            "note": "test",
                        })
    out = stats.build_results(1)
    assert out["live"]["trades"] == 2
    assert out["live"]["win_rate"] == 50.0
    assert "FVG" in out["live"]["by_setup"]
    assert "London" in out["live"]["by_session"]
    assert out["paper"]["trades"] == 1
    assert out["backtest"]["trades"] == 10
    assert out["disclaimer"]


def test_stats_endpoint(monkeypatch):
    monkeypatch.setattr(stats, "build_results",
                        lambda years=5: {"years": years, "live": {}, "paper": {},
                                          "backtest": {}, "disclaimer": "d"})
    app = FastAPI()
    app.include_router(misc.router)
    with TestClient(app) as client:
        r = client.get("/api/stats/results?years=3")
    assert r.status_code == 200
    assert r.json()["years"] == 3
