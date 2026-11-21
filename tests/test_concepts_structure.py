"""Unit tests for the causal market-structure concept detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.concepts import detect_all, detect_all_dicts
from app.concepts.base import (
    DETECTORS,
    DETECTOR_CATEGORIES,
    REQUIRED_FIELDS,
    swing_points,
)
from app.concepts.structure import SWING_LOOKBACK, detect_swing_high

STRUCTURE_TYPES = (
    "swing_high", "swing_low", "trend", "bos", "choch", "mss",
    "internal_structure", "external_structure",
)


def _frame_from_closes(closes, wick: float = 1.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    op = np.concatenate([[closes[0]], closes[:-1]])
    hi = np.maximum(op, closes) + wick
    lo = np.minimum(op, closes) - wick
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"Open": op, "High": hi, "Low": lo, "Close": closes, "Volume": 1e3},
        index=idx,
    )


def _trend_frame(direction: int) -> pd.DataFrame:
    t = np.arange(60)
    closes = 2000 + direction * (2.0 * t + 15 * np.sin(2 * np.pi * t / 10))
    return _frame_from_closes(closes)


def _break_frame() -> pd.DataFrame:
    """Lower highs / lower lows, a bearish BOS, then a displacement CHoCH."""
    closes = [
        2060, 2055, 2050, 2045, 2040, 2045, 2050,
        2045, 2040, 2035, 2030, 2025, 2020, 2015, 2010,
        2015, 2020, 2015, 2010, 2005, 2000, 1995, 1990, 1985, 1980,
        1985, 2045,
    ]
    return _frame_from_closes(closes)


def _causal_frame(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 2000 + 30 * np.sin(2 * np.pi * t / 75) + 0.3 * t + rng.normal(0, 0.2, n)
    op = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(op, close) + 1.0 + rng.uniform(0, 0.3, n)
    lo = np.minimum(op, close) - 1.0 - rng.uniform(0, 0.3, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": 1e3},
        index=idx,
    )


def test_trend_direction_uptrend_and_downtrend():
    up = detect_all(_trend_frame(1), types=("trend",))
    down = detect_all(_trend_frame(-1), types=("trend",))
    assert up and down
    assert up[-1].direction == "bullish"
    assert down[-1].direction == "bearish"
    assert "higher highs" in up[-1].notes
    assert "lower lows" in down[-1].notes


def test_break_concepts_direction_levels_and_displacement():
    df = _break_frame()
    bos = detect_all(df, types=("bos",))
    choch = detect_all(df, types=("choch",))
    mss = detect_all(df, types=("mss",))

    bearish_bos = [c for c in bos if c.direction == "bearish"]
    assert bearish_bos
    lows = swing_points(df, lookback=3, confirm=3, kind="low")
    broken_low = max((p for p in lows if p["index"] < 20), key=lambda p: p["index"])
    assert all(c.top == pytest.approx(broken_low["price"]) for c in bearish_bos)
    assert all(c.bottom == pytest.approx(broken_low["price"]) for c in bearish_bos)
    assert all(c.top == c.bottom for c in bearish_bos)

    bullish_choch = [c for c in choch if c.direction == "bullish"]
    assert bullish_choch
    highs = swing_points(df, lookback=3, confirm=3, kind="high")
    broken_high = max((p for p in highs if p["index"] < 26), key=lambda p: p["index"])
    break_bar = df.index[26]
    assert bullish_choch[-1].end_time == break_bar
    assert bullish_choch[-1].top == pytest.approx(broken_high["price"])
    assert bullish_choch[-1].bottom == pytest.approx(broken_high["price"])

    bullish_mss = [c for c in mss if c.direction == "bullish"]
    assert bullish_mss
    latest = bullish_mss[-1]
    assert latest.end_time == break_bar
    assert latest.meta["displacement"] is True
    assert latest.top == pytest.approx(broken_high["price"])
    assert latest.bottom == pytest.approx(broken_high["price"])


def test_swing_pivots_are_confirmed_before_visibility():
    df = _causal_frame(90)
    concepts = detect_swing_high(df)
    assert concepts
    target = concepts[len(concepts) // 2]
    pivot_index = int(target.meta["pivot_index"])

    confirm_bar = df.index[pivot_index + SWING_LOOKBACK]
    assert pd.Timestamp(target.start_time) == confirm_bar
    assert pd.Timestamp(target.end_time) == confirm_bar
    assert pd.Timestamp(target.end_time) > pd.Timestamp(target.meta["pivot_time"])

    visible_before = {int(c.meta["pivot_index"]) for c in detect_swing_high(df[:pivot_index + SWING_LOOKBACK])}
    assert pivot_index not in visible_before


def test_causal_truncation_no_lookahead():
    df = _causal_frame()
    k = 220
    cutoff = df.index[k - 1]

    left = [c.to_dict() for c in detect_all(df[:k], types=STRUCTURE_TYPES)]
    full = detect_all(df, types=STRUCTURE_TYPES)
    right = [c.to_dict() for c in full if pd.Timestamp(c.end_time) <= cutoff]
    assert left == right

    edited = df.copy()
    col = edited.columns.get_loc("Close")
    edited.iloc[-5:, col] *= 1.4
    full_edited = detect_all(edited, types=STRUCTURE_TYPES)
    right_edited = [c.to_dict() for c in full_edited if pd.Timestamp(c.end_time) <= cutoff]
    assert right_edited == right


def test_every_dict_has_required_fields_and_valid_enums():
    out = detect_all_dicts(_causal_frame(120), types=STRUCTURE_TYPES)
    assert out
    for d in out:
        assert set(REQUIRED_FIELDS).issubset(d)
        assert d["direction"] in ("bullish", "bearish", "neutral")
        assert d["status"] in ("active", "mitigated", "invalidated")
        assert 0.0 <= d["strength"] <= 100.0
        assert isinstance(d["notes"], str)
        assert d["type"] in STRUCTURE_TYPES
        assert d["category"] == "structure"


def test_registration_caps_and_short_or_nan_frames():
    for name in STRUCTURE_TYPES:
        assert name in DETECTORS
        assert DETECTOR_CATEGORIES[name] == "structure"

    long = _causal_frame(300)
    for name in STRUCTURE_TYPES:
        cap = 40 if name in ("swing_high", "swing_low") else 10
        assert len(detect_all(long, types=(name,))) <= cap

    short = _frame_from_closes([2000.0, 2001.0])
    for name in STRUCTURE_TYPES:
        assert detect_all(short, types=(name,)) == []

    nan_frame = long.copy()
    nan_frame.iloc[10:15, nan_frame.columns.get_loc("Close")] = np.nan
    for name in STRUCTURE_TYPES:
        detect_all(nan_frame, types=(name,))
