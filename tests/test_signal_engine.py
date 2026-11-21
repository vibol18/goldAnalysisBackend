"""Unit tests for the signal engine (scoring, pricing, outcomes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.indicators import add_all
from app.services.signal_engine import evaluate_outcome, generate_signal


def make_df(n: int = 400, seed: int = 3, drift: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.005, n)
    close = 2000 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1000, 50000, n).astype(float)},
        index=idx,
    )


def test_signal_valid_on_uptrend():
    sig = generate_signal(make_df(drift=0.002), "1D", train_model=False)
    assert sig.action in {"BUY", "SELL", "HOLD"}
    assert 0 <= sig.confidence <= 100
    assert sig.atr > 0
    if sig.action == "BUY":
        assert sig.stop_loss < sig.entry < sig.take_profit1 < sig.take_profit2
        assert sig.risk_reward > 0
    elif sig.action == "SELL":
        assert sig.take_profit2 < sig.take_profit1 < sig.entry < sig.stop_loss


def test_signal_explanation_has_reasons():
    sig = generate_signal(make_df(), "1D", train_model=False)
    exp = sig.explanation
    assert exp["summary"]
    assert isinstance(exp["reasons"], list)
    assert len(exp["components"]) >= 4
    for c in exp["components"]:
        assert -1.0001 <= c["score"] <= 1.0001


def test_downtrend_tends_bearish():
    """Persistent trends must score in the expected direction."""

    def monotonic(n: int, up: bool) -> pd.DataFrame:
        rng = np.random.default_rng(4)
        base = 2000 * (1.0015 if up else 0.9985) ** np.arange(n)
        close = base * (1 + rng.normal(0, 0.0002, n))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        return pd.DataFrame(
            {"Open": open_, "High": np.maximum(open_, close) * 1.002,
             "Low": np.minimum(open_, close) * 0.998, "Close": close,
             "Volume": rng.integers(1000, 50000, n).astype(float)},
            index=idx,
        )

    bear = generate_signal(monotonic(400, up=False), "1D", train_model=False)
    bull = generate_signal(monotonic(400, up=True), "1D", train_model=False)
    assert bear.rule_score < 0 < bull.rule_score
    assert bear.rule_score < bull.rule_score


def test_dxy_inverse_context():
    df = make_df(seed=5, drift=0.0)
    s_weak_dxy = generate_signal(df, "1D", dxy_change_pct=-0.8, train_model=False)
    s_strong_dxy = generate_signal(df, "1D", dxy_change_pct=+0.8, train_model=False)
    assert s_weak_dxy.rule_score > s_strong_dxy.rule_score


def test_evaluate_outcome_hit_tp():
    df = make_df(100, seed=1, drift=0.0001)
    ind = add_all(df)
    ts = ind.index[50]
    sig = {
        "created_at": ts.isoformat(),
        "action": "BUY",
        "entry": float(ind["Close"].iloc[50]),
        "stop_loss": float(ind["Low"].iloc[50]) - 1.0,
        "take_profit1": float(ind["Close"].iloc[50]) + 5,
        "take_profit2": float(ind["Close"].iloc[50]) + 10,
        "status": "open",
    }
    # force a TP2 touch right after signal time
    df2 = df.copy()
    df2.loc[df2.index[51], ["High", "Low", "Close"]] = [
        sig["entry"] + 50, sig["entry"] - 1, sig["entry"] + 40
    ]
    out = evaluate_outcome(sig, df2)
    assert out["status"] == "closed"
    assert out["hit"] == "TP"
    assert out["result_pct"] > 0


def test_evaluate_outcome_hit_sl():
    df = make_df(100, seed=2, drift=0.0001)
    ts = df.index[50]
    sig = {
        "created_at": ts.isoformat(),
        "action": "BUY",
        "entry": float(df["Close"].iloc[50]),
        "stop_loss": float(df["Close"].iloc[50]) - 10,
        "take_profit1": float(df["Close"].iloc[50]) + 20,
        "take_profit2": float(df["Close"].iloc[50]) + 40,
        "status": "open",
    }
    df2 = df.copy()
    df2.loc[df2.index[51], ["High", "Low", "Close"]] = [
        sig["entry"] + 1, sig["entry"] - 100, sig["entry"] - 50
    ]
    out = evaluate_outcome(sig, df2)
    assert out["status"] == "closed"
    assert out["hit"] == "SL"
    assert out["result_pct"] < 0


def test_evaluate_outcome_open_when_no_touch():
    df = make_df(100, seed=4, drift=0.0)
    ts = df.index[50]
    sig = {
        "created_at": ts.isoformat(),
        "action": "BUY",
        "entry": float(df["Close"].iloc[50]),
        "stop_loss": float(df["Close"].iloc[50]) - 1000,
        "take_profit1": float(df["Close"].iloc[50]) + 1000,
        "take_profit2": float(df["Close"].iloc[50]) + 2000,
        "status": "open",
    }
    out = evaluate_outcome(sig, df)
    assert out["status"] == "open"
    assert out["hit"] is None
