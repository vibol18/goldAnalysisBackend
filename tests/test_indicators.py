"""Unit tests for technical indicators (causality + correctness)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.indicators import (
    add_all,
    add_atr,
    add_bollinger,
    add_ema,
    add_macd,
    add_rsi,
    add_support_resistance,
)


def make_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.005, n)
    close = 2000 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1000, 50000, n).astype(float)},
        index=idx,
    )


def test_ema_is_causal():
    """Changing the future must not alter past EMA values."""
    df = make_df()
    a = add_ema(df)
    df2 = df.copy()
    df2.loc[df2.index[-1], "Close"] *= 1.5
    b = add_ema(df2)
    # all but the last few rows must match (EMA has long memory via recursion,
    # so compare only rows unaffected by the final print: shift window back)
    head = slice(None, -1)
    pd.testing.assert_series_equal(
        a["ema50"].iloc[head], b["ema50"].iloc[head],
        check_names=False,
    )


def test_rsi_bounds_and_length():
    df = make_df()
    out = add_rsi(df)
    valid = out["rsi"].dropna()
    assert len(valid) > 100
    assert valid.between(0, 100).all()


def test_macd_cross_flags_are_boolean():
    df = add_macd(make_df())
    assert df["macd_cross_up"].dtype == bool
    assert df["macd_cross_down"].dtype == bool
    # a bar cannot be both crosses at once
    assert not (df["macd_cross_up"] & df["macd_cross_down"]).any()


def test_bollinger_ordering():
    df = add_bollinger(make_df())
    v = df.dropna(subset=["bb_high", "bb_low"])
    assert (v["bb_high"] >= v["bb_mid"]).all()
    assert (v["bb_mid"] >= v["bb_low"]).all()
    assert df["bb_pct"].dropna().between(0, 1).all()


def test_atr_positive():
    df = add_atr(make_df())
    # ta returns 0 during the warmup window, real values afterwards
    assert (df["atr"].iloc[20:].dropna() > 0).all()


def test_support_resistance_causal():
    """S/R at time t must not change when future bars are modified."""
    df = make_df(300)
    out1 = add_support_resistance(df, window=5)
    df2 = df.copy()
    df2.iloc[-3:, df2.columns.get_loc("Close")] *= 1.2
    df2.iloc[-3:, df2.columns.get_loc("High")] *= 1.2
    df2.iloc[-3:, df2.columns.get_loc("Low")] *= 1.2
    out2 = add_support_resistance(df2, window=5)
    n = len(df) - 20  # ignore warmup / tail boundary
    pd.testing.assert_series_equal(
        out1["support"].iloc[:n], out2["support"].iloc[:n],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out1["resistance"].iloc[:n], out2["resistance"].iloc[:n],
        check_names=False,
    )


def test_add_all_columns_present():
    df = add_all(make_df())
    for col in ("ema20", "ema50", "ema200", "rsi", "macd", "atr",
                "bb_pct", "support", "resistance"):
        assert col in df.columns
