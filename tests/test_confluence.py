"""Tests for the weighted confluence engine."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.concepts import load_families
from app.concepts.base import DETECTORS, make_concept
from app.services.confluence import confluence, load_weights


def make_df(n: int = 200, start: float = 2000.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = start + np.linspace(0, 60, n) + np.sin(np.arange(n) / 7) * 4
    return pd.DataFrame({
        "Open": close - 0.5, "High": close + 2.5,
        "Low": close - 2.5, "Close": close,
        "Volume": np.full(n, 1000.0),
    }, index=idx)


def c(type_: str, direction: str, strength: float = 80,
      category: str = "smc", top: float | None = None,
      bottom: float | None = None, meta: dict | None = None,
      end: str | None = None) -> object:
    return make_concept(type_, direction, end or "2024-01-08 00:00:00",
                        end or "2024-01-08 00:00:00", top, bottom,
                        strength, "active", f"{type_} test", category, meta or {})


def test_weights_cover_all_detectors():
    load_families()
    weights = load_weights()
    missing = sorted(set(DETECTORS) - set(weights["weights"]))
    assert missing == [], f"no weight configured for: {missing}"
    assert set(weights["category_caps"]) >= {"structure", "smc", "classic",
                                             "price_action", "mtf", "macro"}


def test_bullish_and_bearish_concept_sets_flip_score():
    df = make_df()
    price = float(df["Close"].iloc[-1])
    bull = [
        c("trend", "bullish", 90, "structure"),
        c("bos", "bullish", 85, "structure"),
        c("fvg", "bullish", 80, "smc", price + 2, price - 2),
        c("order_block", "bullish", 85, "smc", price + 3, price - 3),
        c("liquidity_sweep", "bullish", 80, "smc"),
        c("mtf_bias", "bullish", 80, "mtf"),
    ]
    bear = [c(x.type, "bearish", x.strength, x.category, x.top, x.bottom) for x in bull]
    up = confluence(bull, df)
    down = confluence(bear, df)
    assert up.action == "BUY" and down.action == "SELL"
    assert up.score > 0 > down.score
    assert up.agreeing and down.agreeing
    assert up.grade in ("A+", "A", "B", "C")


def test_category_caps_prevent_one_family_dominating():
    df = make_df()
    many = [c("trend", "bullish", 100, "structure") for _ in range(50)]
    result = confluence(many, df)
    caps = load_weights()["category_caps"]
    assert result.categories["structure"] <= caps["structure"] + 1e-6
    assert abs(result.score) <= 1.0


def test_veto_for_choppy_regime_event_and_htf():
    df = make_df()
    base = [
        c("trend", "bullish", 95, "structure"),
        c("fvg", "bullish", 95, "smc", df["Close"].iloc[-1] + 2, df["Close"].iloc[-1] - 2),
        c("mtf_bias", "bullish", 90, "mtf"),
    ]
    choppy = c("market_regime", "neutral", 80, "classic",
               meta={"regime": "ranging"})
    event = c("event_risk", "neutral", 95, "macro")
    veto = c("htf_veto", "bearish", 80, "mtf")
    assert confluence(base + [choppy], df).action == "HOLD"
    out = confluence(base + [event], df)
    assert out.action == "HOLD" and any("event" in v for v in out.veto_reasons)
    out = confluence(base + [veto], df)
    assert out.action == "HOLD" and any("higher-timeframe" in v for v in out.veto_reasons)


def test_levels_and_targets_are_ordered():
    df = make_df()
    p = float(df["Close"].iloc[-1])
    now = str(df.index[-1])
    concepts = [
        c("trend", "bullish", 90, "structure", end=now),
        c("bos", "bullish", 90, "structure", end=now),
        c("liquidity_sweep", "bullish", 85, "smc", end=now),
        c("mtf_bias", "bullish", 85, "mtf", end=now),
        c("order_block", "bullish", 90, "smc", p + 3, p - 3, end=now),
        c("swing_low", "neutral", 80, "structure", p - 8, p - 8, end=now),
        c("equal_highs", "neutral", 70, "smc", p + 30, p + 30, end=now),
        c("pdh", "neutral", 70, "smc", p + 55, p + 55, end=now),
        c("pwh", "neutral", 70, "smc", p + 80, p + 80, end=now),
    ]
    out = confluence(concepts, df)
    assert out.action == "BUY"
    assert out.entry_zone is not None
    assert out.stop_loss is not None and out.stop_loss < out.entry_zone[0]
    assert out.take_profit1 is not None and out.take_profit2 is not None
    assert out.take_profit1 < out.take_profit2
    assert out.risk_reward > 0
    assert "close below" in out.invalidation


def test_result_is_json_serialisable():
    df = make_df()
    out = confluence([c("trend", "bullish", 80, "structure")], df, ml_probability=0.6)
    json.dumps(out.to_dict(), allow_nan=False)
    assert out.ml_probability == 0.6
