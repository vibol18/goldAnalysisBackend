"""Non-repaint signal storage and explanation-contract tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import db
from app.services.confluence import confluence
from app.services.explanation import build_explanation
from app.services.signal_engine import closed_frame, generate_signal


def make_df(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    close = 2000 * np.cumprod(1 + rng.normal(0.0004, 0.006, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "Open": open_, "High": np.maximum(open_, close) * 1.004,
        "Low": np.minimum(open_, close) * 0.996, "Close": close,
        "Volume": rng.integers(500, 40000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))


def test_closed_frame_drops_forming_candle():
    df = make_df()
    closed = closed_frame(df)
    assert len(closed) == len(df) - 1
    assert closed.index[-1] == df.index[-2]


def test_signal_time_is_the_closed_candle_not_wall_clock():
    df = make_df()
    sig = generate_signal(closed_frame(df), "1h", train_model=False)
    expected = pd.Timestamp(df.index[-2]).isoformat()
    assert sig.signal_time == expected
    assert sig.to_dict()["signal_time"] == expected


def test_signal_immutable_when_future_candles_change():
    df = make_df()
    base = generate_signal(closed_frame(df), "1h", train_model=False)
    changed = df.copy()
    changed.iloc[-1:, changed.columns.get_loc("Close")] *= 1.3
    later = generate_signal(closed_frame(changed), "1h", train_model=False)
    assert base.signal_time == later.signal_time
    for field in ("action", "grade", "entry", "stop_loss", "take_profit1",
                  "take_profit2", "risk_reward"):
        assert getattr(base, field) == getattr(later, field), field


def test_storage_persists_signal_time_and_grade(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "signals.db")
    monkeypatch.setattr(db, "_conn", None)
    sig = generate_signal(closed_frame(make_df()), "1h", train_model=False)
    payload = sig.to_dict()
    payload["id"] = db.save_signal(payload)
    rows = db.list_signals(limit=1)
    assert rows[0]["signal_time"] == payload["signal_time"]
    assert rows[0]["grade"] == payload["grade"]
    assert isinstance(rows[0]["explanation"], dict)


def test_explanation_has_checklist_and_hold_reason():
    df = make_df()
    cf = confluence([], df)
    explanation = build_explanation([], cf, df, "1h")
    assert isinstance(explanation["checklist"], list)
    assert explanation["why_not_trading"]
    assert "No trade taken" in explanation["why_not_trading"]
