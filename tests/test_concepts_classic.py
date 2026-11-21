"""Unit tests for the strictly causal classic concept family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.concepts.classic as classic
from app.concepts import detect_all
from app.concepts.base import DETECTORS, DETECTOR_CATEGORIES, REQUIRED_FIELDS

ALL_TYPES = (
    "sr_zone", "supply_demand", "trendline", "channel", "breakout", "retest",
    "ema_cross", "price_vs_ema", "rsi_signal", "rsi_divergence", "macd_cross",
    "macd_divergence", "stochastic", "bollinger_squeeze", "bollinger_expansion",
    "atr_volatility", "adx_trend_strength", "fibonacci", "pivot_points",
    "vwap", "anchored_vwap", "volume_profile", "ichimoku", "market_regime",
)

GEOMETRY = ("type", "direction", "start_time", "end_time", "top", "bottom")


def frame_from_close(close: np.ndarray, seed: int = 11,
                     freq: str = "h") -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    rng = np.random.default_rng(seed)
    op = np.roll(close, 1)
    op[0] = close[0]
    hi = np.maximum(op, close) + np.abs(rng.normal(0.8, 0.3, len(close))) + 0.5
    lo = np.minimum(op, close) - np.abs(rng.normal(0.8, 0.3, len(close))) - 0.5
    vol = rng.integers(1000, 50000, len(close)).astype(float)
    return pd.DataFrame(
        {"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": vol},
        index=pd.date_range("2023-01-01", periods=len(close), freq=freq),
    )


def make_frame(n: int = 300, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.005, n)
    return frame_from_close(2000 * np.cumprod(1 + rets), seed=seed + 1)


def detect(types, df: pd.DataFrame) -> list[dict]:
    return [c.to_dict() for c in detect_all(df, types=types)]


def geom_keys(concepts: list[dict]) -> set[tuple]:
    keys = set()
    for c in concepts:
        top = None if c["top"] is None else round(float(c["top"]), 3)
        bottom = None if c["bottom"] is None else round(float(c["bottom"]), 3)
        keys.add((c["type"], c["direction"], c["start_time"], c["end_time"],
                  top, bottom))
    return keys


def test_all_detectors_registered_as_classic():
    for name in ALL_TYPES:
        assert name in DETECTORS, name
        assert DETECTOR_CATEGORIES[name] == "classic", name
        assert callable(DETECTORS[name])


def test_every_concept_has_required_fields():
    df = make_frame()
    concepts = detect(ALL_TYPES, df)
    assert concepts
    for c in concepts:
        for field in REQUIRED_FIELDS:
            assert field in c, (c["type"], field)
        assert c["direction"] in {"bullish", "bearish", "neutral"}
        assert c["status"] in {"active", "mitigated", "invalidated"}
        assert 0.0 <= c["strength"] <= 100.0
        assert c["category"] == "classic"


def test_ema_cross_fires_on_trend_flip_with_correct_direction():
    rng = np.random.default_rng(5)
    close = np.concatenate([
        np.linspace(2050, 1950, 80),
        np.linspace(1950, 2130, 100),
        np.linspace(2130, 1930, 100),
    ]) + rng.normal(0, 0.5, 280)
    res = detect(("ema_cross",), frame_from_close(close))
    golden = [c for c in res if c["direction"] == "bullish"
              and c["meta"]["pair"] in {"9/20", "20/50"}]
    death = [c for c in res if c["direction"] == "bearish"
             and c["meta"]["pair"] in {"9/20", "20/50"}]
    assert golden, "expected a golden cross after the up-turn"
    assert death, "expected a death cross after the down-turn"
    for c in golden + death:
        assert c["start_time"] == c["end_time"]


def test_price_vs_ema_direction():
    up = frame_from_close(np.linspace(1900, 2160, 300) +
                          np.random.default_rng(1).normal(0, 1, 300))
    down = frame_from_close(np.linspace(2160, 1900, 300) +
                            np.random.default_rng(2).normal(0, 1, 300))
    assert detect(("price_vs_ema",), up)[0]["direction"] == "bullish"
    assert detect(("price_vs_ema",), down)[0]["direction"] == "bearish"


def test_rsi_signal_overbought_and_oversold():
    up = frame_from_close(np.linspace(1880, 2120, 70))
    down = frame_from_close(np.linspace(2120, 1880, 70))
    ob = detect(("rsi_signal",), up)[0]
    os_ = detect(("rsi_signal",), down)[0]
    assert ob["meta"]["zone"] == "overbought" and ob["direction"] == "bearish"
    assert os_["meta"]["zone"] == "oversold" and os_["direction"] == "bullish"


def test_rsi_divergence_lower_low_with_higher_rsi_low():
    parts = [
        np.linspace(2050, 1990, 14),
        np.linspace(1990, 2060, 22),
        np.linspace(2060, 1975, 32),
        np.linspace(1975, 2045, 24),
    ]
    close = np.concatenate(parts) + np.random.default_rng(3).normal(0, 0.4, 92)
    res = detect(("rsi_divergence",), frame_from_close(close))
    bull = [c for c in res if c["direction"] == "bullish"]
    assert bull, "expected bullish RSI divergence on lower low / higher RSI low"
    assert bull[0]["meta"]["indicator_at_second"] > bull[0]["meta"]["indicator_at_first"]


def test_macd_cross_on_deterministic_frame():
    res = detect(("macd_cross",), make_frame())
    assert res
    assert all(c["meta"]["pair"] == "macd/signal" for c in res)


def test_stochastic_on_deterministic_frame():
    assert detect(("stochastic",), make_frame())


def test_bollinger_squeeze_and_expansion_on_deterministic_frame():
    df = make_frame()
    assert detect(("bollinger_squeeze",), df)
    assert detect(("bollinger_expansion",), df)


def test_adx_trend_strength_on_deterministic_frame():
    res = detect(("adx_trend_strength",), make_frame())
    assert res
    assert res[0]["meta"]["regime"] in {"trending", "ranging", "transitional"}


def test_market_regime_on_deterministic_frame():
    res = detect(("market_regime",), make_frame())
    assert res
    assert res[0]["meta"]["regime"] in {
        "trending_up", "trending_down", "ranging", "volatile",
    }
    assert res[0]["direction"] in {"bullish", "bearish", "neutral"}


def test_pivot_points_use_only_completed_daily_bars():
    n = 72
    rng = np.random.default_rng(9)
    close = 2000 + np.cumsum(rng.normal(0, 4, n))
    frame = frame_from_close(close, seed=9)
    full = detect(("pivot_points",), frame)
    assert full

    def levels(concepts: list[dict]) -> dict:
        return {c["meta"]["set"]: c["meta"]["levels"] for c in concepts}

    truncated = detect(("pivot_points",), frame.iloc[:60])
    assert levels(truncated) == levels(full)
    assert {c["start_time"] for c in truncated} == {c["start_time"] for c in full}

    edited = frame.copy()
    edited.iloc[48:, edited.columns.get_loc("Close")] += 500.0
    edited.iloc[48:, edited.columns.get_loc("High")] += 500.0
    edited.iloc[48:, edited.columns.get_loc("Low")] += 500.0
    assert levels(detect(("pivot_points",), edited)) == levels(full)
    assert {c["meta"]["set"] for c in full} == {"classic", "fibonacci", "camarilla"}


def test_causal_truncation_matches_full_history():
    df = make_frame(300)
    k = 200
    short = detect(ALL_TYPES, df.iloc[:k])
    full = detect(ALL_TYPES, df)
    cutoff = df.index[k - 1]
    before = [c for c in full
              if c["end_time"] is not None and pd.Timestamp(c["end_time"]) <= cutoff]

    missing = geom_keys(before) - geom_keys(short)
    assert not missing, f"concepts confirmed before {cutoff} vanished on truncation: {missing}"


def test_short_frames_are_handled_gracefully():
    for n in (0, 1, 3, 10):
        df = make_frame(max(n, 1)).iloc[:n]
        for fn in vars(classic).values():
            if callable(fn) and getattr(fn, "__name__", "").startswith("detect_"):
                assert isinstance(fn(df), list), fn.__name__
