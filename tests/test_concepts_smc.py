"""Unit tests for the smart-money-concept (SMC) detector family.

Frames are deterministic and hand-crafted; the app itself only ever uses real
market data.  The tests cover geometry, status transitions and — crucially —
strict causality (truncating a frame must not rewrite past concept geometry).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.concepts import detect_all, detect_all_dicts
from app.concepts.base import REQUIRED_FIELDS
from app.concepts.smc import (
    detect_equal_highs,
    detect_fvg,
    detect_killzone,
    detect_liquidity_sweep,
    detect_ote,
    detect_order_block,
    detect_premium_discount,
)


def _frame(rows: list[tuple[float, float, float, float]],
           start: str = "2024-01-01 00:00", freq: str = "h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


BULL_FVG_ROWS = [
    (100.0, 110.0, 99.0, 108.0),
    (108.0, 118.0, 107.0, 116.0),
    (116.0, 130.0, 121.0, 128.0),
]

BEAR_FVG_ROWS = [
    (130.0, 131.0, 120.0, 122.0),
    (122.0, 123.0, 112.0, 114.0),
    (114.0, 118.0, 105.0, 116.0),
]


# --------------------------------------------------------------------------- #
# FVG
# --------------------------------------------------------------------------- #
def test_bullish_fvg_geometry():
    concepts = detect_fvg(_frame(BULL_FVG_ROWS))
    assert len(concepts) == 1
    c = concepts[0]
    assert c.type == "fvg"
    assert c.direction == "bullish"
    assert c.top == pytest.approx(121.0)   # candle[i].Low
    assert c.bottom == pytest.approx(110.0)  # candle[i-2].High
    assert c.status == "active"
    assert c.category == "smc"


def test_bearish_fvg_geometry():
    concepts = detect_fvg(_frame(BEAR_FVG_ROWS))
    assert len(concepts) == 1
    c = concepts[0]
    assert c.direction == "bearish"
    assert c.top == pytest.approx(120.0)   # candle[i-2].Low
    assert c.bottom == pytest.approx(118.0)  # candle[i].High


def test_tiny_fvg_gap_below_atr_threshold_ignored():
    rows = [
        (100.0, 110.0, 99.0, 108.0),
        (108.0, 118.0, 107.0, 116.0),
        (116.0, 120.0, 110.5, 118.0),  # gap = 0.5 << 0.1 * ATR (~1.05)
    ]
    assert detect_fvg(_frame(rows)) == []


def test_fvg_mitigation_status_evolution():
    partial = _frame(BULL_FVG_ROWS + [
        (128.0, 130.0, 115.0, 116.0),  # wick into gap, close above bottom
    ])
    c = detect_fvg(partial)[0]
    assert c.status == "mitigated"

    through = _frame(BULL_FVG_ROWS + [
        (128.0, 130.0, 115.0, 116.0),
        (116.0, 117.0, 105.0, 106.0),  # closes through the whole gap
    ])
    c = detect_fvg(through)[0]
    assert c.status == "invalidated"


# --------------------------------------------------------------------------- #
# order block / liquidity sweep
# --------------------------------------------------------------------------- #
def test_bullish_order_block_from_bearish_candle():
    rows = [
        (100.0, 102.0, 99.0, 101.0),
        (101.0, 103.0, 100.0, 102.0),
        (102.0, 104.0, 101.0, 103.0),
        (103.0, 105.0, 102.0, 104.0),
        (104.0, 105.0, 98.0, 99.0),    # last bearish candle -> the order block
        (99.0, 117.0, 98.0, 115.0),    # strong bullish displacement, breaks high 105
        (115.0, 118.0, 113.0, 116.0),
        (116.0, 119.0, 114.0, 117.0),
    ]
    obs = detect_order_block(_frame(rows))
    bull = [o for o in obs if o.direction == "bullish"]
    assert bull, "expected a bullish order block"
    ob = bull[0]
    assert ob.top == pytest.approx(105.0)
    assert ob.bottom == pytest.approx(98.0)
    assert ob.end_time == pd.Timestamp("2024-01-01 05:00")
    assert ob.meta["ob_time"] == pd.Timestamp("2024-01-01 04:00")


def test_bearish_liquidity_sweep_above_equal_highs():
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 103.0, 100.0, 101.0),
        (101.0, 105.0, 101.0, 102.0),  # swing high 105
        (102.0, 103.0, 100.0, 101.0),
        (101.0, 104.0, 100.0, 102.0),
        (102.0, 105.0, 102.0, 103.0),  # equal swing high 105
        (103.0, 104.0, 101.0, 102.0),
        (102.0, 110.0, 100.0, 95.0),   # wick above 105, closes back below
        (95.0, 97.0, 93.0, 94.0),
    ]
    frame = _frame(rows)
    assert detect_equal_highs(frame), "equal highs must be detected first"
    sweeps = detect_liquidity_sweep(frame)
    bear = [s for s in sweeps if s.direction == "bearish"
            and abs(s.meta["swept_price"] - 105.0) < 1e-6]
    assert bear, "expected a bearish sweep of the 105 equal-high level"
    assert bear[0].top == pytest.approx(110.0)
    assert bear[0].bottom == pytest.approx(105.0)
    assert "105" in bear[0].notes


# --------------------------------------------------------------------------- #
# premium/discount, OTE, killzones
# --------------------------------------------------------------------------- #
def test_premium_discount_position_pct():
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 102.0, 99.0, 101.0),
        (101.0, 103.0, 101.0, 102.0),  # swing high 103
        (102.0, 102.5, 100.0, 101.0),
        (101.0, 102.0, 97.0, 98.0),    # swing low 97
        (98.0, 100.0, 98.0, 99.0),
        (99.0, 101.0, 98.5, 100.0),
        (100.0, 103.0, 99.0, 101.0),
        (101.0, 104.0, 100.0, 102.0),  # close = 102
    ]
    c = detect_premium_discount(_frame(rows))[0]
    assert c.meta["swing_low"] == pytest.approx(97.0)
    assert c.meta["swing_high"] == pytest.approx(103.0)
    assert c.top == pytest.approx(103.0)
    assert c.bottom == pytest.approx(97.0)
    assert c.meta["position_pct"] == pytest.approx((102 - 97) / 6 * 100)
    assert 0 <= c.meta["position_pct"] <= 100
    assert c.direction == "bearish"  # above equilibrium (100)


def test_ote_zone_is_62_to_79_percent_retracement():
    rows = [
        (200.0, 202.0, 199.0, 200.0),
        (200.0, 201.0, 198.0, 199.0),
        (199.0, 200.0, 190.0, 191.0),  # impulse low
        (191.0, 193.0, 191.0, 192.0),
        (192.0, 195.0, 192.0, 194.0),
        (194.0, 200.0, 194.0, 199.0),
        (199.0, 210.0, 199.0, 209.0),
        (209.0, 220.0, 208.0, 218.0),
        (218.0, 230.0, 217.0, 228.0),
        (228.0, 232.0, 226.0, 229.0),  # impulse high
        (229.0, 231.0, 225.0, 226.0),
    ]
    c = detect_ote(_frame(rows))[0]
    assert c.direction == "bullish"
    lo, hi = c.meta["impulse_low"], c.meta["impulse_high"]
    assert c.top == pytest.approx(hi - 0.62 * (hi - lo))
    assert c.bottom == pytest.approx(hi - 0.79 * (hi - lo))
    assert lo < c.bottom < c.top < hi


def test_killzone_labels_for_utc_timestamps():
    rows = [(100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i) for i in range(5)]
    idx = pd.to_datetime([
        "2024-01-01 00:00", "2024-01-01 06:00", "2024-01-01 07:00",
        "2024-01-01 12:00", "2024-01-01 20:00",
    ])
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    zones = [k.meta["zone"] for k in detect_killzone(frame)]
    assert zones == ["asian", "london", "new_york"]
    for k in detect_killzone(frame):
        assert k.meta["in_killzone"] is True


# --------------------------------------------------------------------------- #
# causality + schema
# --------------------------------------------------------------------------- #
def _noisy_frame(n: int = 220, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 4.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.5, 5.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 5.0, n)
    idx = pd.date_range("2024-03-01", periods=n, freq="h")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(100, 1000, n).astype(float)},
        index=idx,
    )


GEOMETRY = ("type", "direction", "top", "bottom", "start_time", "end_time")


def _pattern_frame(blocks: int = 6) -> pd.DataFrame:
    """Deterministic bearish-OB + bullish-displacement pattern blocks."""
    rows: list[tuple[float, float, float, float]] = []
    base = 1000.0
    for _ in range(blocks):
        rows += [
            (base, base + 2, base - 1, base + 1),
            (base + 1, base + 3, base, base + 2),
            (base + 2, base + 4, base + 1, base + 3),
            (base + 3, base + 5, base + 2, base + 4),
            (base + 4, base + 5, base - 4, base - 3),      # bearish OB
            (base - 3, base + 17, base - 5, base + 14),    # up-displacement
            (base + 14, base + 17, base + 12, base + 15),  # leaves an FVG
        ]
        base += 14
    idx = pd.date_range("2024-04-01", periods=len(rows), freq="h")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def test_causal_truncation_preserves_geometry():
    """Truncating at bar k must not change geometry of concepts ending <= k.

    Status for still-open zones may legitimately differ when future candles
    mitigate them, so only geometry fields are compared here.  Capped event
    types (e.g. liquidity_sweep) are excluded from this strict equality check:
    the per-type payload cap keeps the newest N detections, so a long frame
    legitimately drops old ones that a truncated frame still reports.
    """
    df = _pattern_frame()
    types = ("fvg", "displacement", "order_block")
    cutoff = len(df) - 3
    boundary = df.index[cutoff - 1]

    full = [c.to_dict() for c in detect_all(df, types=types)
            if pd.Timestamp(c.end_time) <= boundary]
    truncated = [c.to_dict() for c in detect_all(df.iloc[:cutoff], types=types)]

    def geometry(concepts):
        return sorted(tuple(c[f] for f in GEOMETRY) for c in concepts)

    assert geometry(full), "expected concepts in the full frame"
    assert geometry(full) == geometry(truncated)


def test_all_concepts_have_required_fields():
    frame = _noisy_frame(n=120, seed=3)
    dicts = detect_all_dicts(frame)
    assert dicts, "expected concepts from the full detector suite"
    for c in dicts:
        for field in REQUIRED_FIELDS:
            assert field in c, f"missing {field} in {c['type']}"
        assert c["direction"] in {"bullish", "bearish", "neutral"}
        assert c["status"] in {"active", "mitigated", "invalidated"}
        assert 0.0 <= c["strength"] <= 100.0
        assert isinstance(c["meta"], dict)


def test_empty_and_short_frames_do_not_raise():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    for fn in (detect_fvg, detect_order_block, detect_liquidity_sweep,
               detect_premium_discount, detect_ote, detect_killzone,
               detect_equal_highs):
        assert fn(empty) == []
