"""Tests for the extended-move / higher-timeframe signal-quality filter."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.concepts.base import make_concept
from app.services.confluence import confluence
from app.services.signal_quality import check_extended_move


def make_frame(n: int = 200, drift: float = 0.0, start: float = 2000.0) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rets = rng.normal(drift, 0.0015, n)
    close = start * np.cumprod(1.0 + rets)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) * 1.002,
        "Low": np.minimum(open_, close) * 0.998,
        "Close": close,
        "Volume": np.full(n, 1000.0),
    }, index=idx)


def concept(type_: str, direction: str, *, meta: dict | None = None,
            top: float | None = None, bottom: float | None = None,
            end: str | None = None) -> object:
    stamp = end or "2024-01-09 00:00:00"
    return make_concept(type_, direction, stamp, stamp, top, bottom,
                        85, "active", f"{type_} test", "smc", meta or {})


def test_sell_blocked_when_price_far_below_ema_and_oversold():
    df = make_frame(drift=-0.004)
    price = float(df["Close"].iloc[-1])
    concepts = [
        concept("trend", "bearish"),
        concept("bos", "bearish"),
        concept("fvg", "bearish", top=price + 3, bottom=price + 1),
        concept("mtf_bias", "bearish", meta={"htf_bias": "bearish"}),
    ]
    check = check_extended_move(concepts, df, "SELL")
    assert check.blocked
    assert any("ATR below EMA" in r for r in check.reasons)
    assert any("oversold" in r for r in check.reasons)

    result = confluence(concepts, df)
    assert result.action == "HOLD"
    assert result.grade == "—"
    assert any("extended move" in r or "oversold" in r for r in result.veto_reasons)
    assert result.quality_check and result.quality_check["blocked"]


def test_buy_blocked_at_higher_timeframe_resistance():
    df = make_frame(drift=0.0002)
    price = float(df["Close"].iloc[-1])
    concepts = [
        concept("trend", "bullish"),
        concept("bos", "bullish"),
        concept("mtf_bias", "bullish", meta={"htf_bias": "bullish"}),
        concept("htf_level", "bearish", meta={
            "kind": "resistance", "timeframe": "1d", "level": price + 0.4,
            "distance_atr": 0.2,
        }),
    ]
    check = check_extended_move(concepts, df, "BUY")
    assert check.blocked
    assert any("resistance" in r for r in check.reasons)
    assert confluence(concepts, df).action == "HOLD"


def test_higher_timeframe_disagreement_blocks_signal():
    df = make_frame(drift=0.001)
    concepts = [
        concept("trend", "bullish"),
        concept("bos", "bullish"),
        concept("fvg", "bullish"),
        concept("mtf_bias", "bearish", meta={"htf_bias": "bearish"}),
    ]
    check = check_extended_move(concepts, df, "BUY")
    assert check.blocked
    assert any("HTF agreement" in r for r in check.reasons)


def test_extended_mirror_allows_sell_in_uptrend_and_buy_in_downtrend():
    df_up = make_frame(drift=0.001, start=1980.0)
    df_down = make_frame(drift=-0.001, start=2020.0)
    bull = [concept("mtf_bias", "bullish", meta={"htf_bias": "bullish"})]
    bear = [concept("mtf_bias", "bearish", meta={"htf_bias": "bearish"})]
    assert not check_extended_move(bear, df_up, "SELL").blocked
    assert not check_extended_move(bull, df_down, "BUY").blocked


def test_quality_check_is_json_serialisable():
    df = make_frame()
    check = check_extended_move([], df, "BUY")
    json.dumps(check.to_dict(), allow_nan=False)
    assert check.metrics["atr"] > 0
