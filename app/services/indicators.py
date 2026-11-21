"""Technical indicators built on the `ta` library plus support/resistance.

Every function accepts an OHLCV DataFrame and returns the same frame with
extra columns. All computations are causal (use only past/present data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def add_ema(df: pd.DataFrame, periods: tuple[int, ...] = (20, 50, 200)) -> pd.DataFrame:
    """Add EMA columns (ema20, ema50, ema200) using close price."""
    out = df.copy()
    for p in periods:
        out[f"ema{p}"] = ta.trend.ema_indicator(out["Close"], window=p)
    return out


def add_bollinger(df: pd.DataFrame, window: int = 20, dev: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands (bb_high, bb_mid, bb_low, bb_pct)."""
    out = df.copy()
    bb = ta.volatility.BollingerBands(out["Close"], window=window, window_dev=dev)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_low"] = bb.bollinger_lband()
    span = (out["bb_high"] - out["bb_low"]).replace(0, np.nan)
    out["bb_pct"] = ((out["Close"] - out["bb_low"]) / span).clip(0, 1)
    return out


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add RSI(14) and its 1-bar rate-of-change (slope)."""
    out = df.copy()
    out["rsi"] = ta.momentum.rsi(out["Close"], window=window)
    out["rsi_slope"] = out["rsi"].diff()
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
             signal: int = 9) -> pd.DataFrame:
    """Add MACD line, signal line, histogram and cross flags."""
    out = df.copy()
    macd = ta.trend.MACD(out["Close"], window_slow=slow, window_fast=fast,
                         window_sign=signal)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    prev = out["macd"].shift(1)
    prev_sig = out["macd_signal"].shift(1)
    out["macd_cross_up"] = (prev <= prev_sig) & (out["macd"] > out["macd_signal"])
    out["macd_cross_down"] = (prev >= prev_sig) & (out["macd"] < out["macd_signal"])
    return out


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add ATR(14) and a normalized ATR percent."""
    out = df.copy()
    atr = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"],
                                         window=window)
    out["atr"] = atr.average_true_range()
    out["atr_pct"] = out["atr"] / out["Close"] * 100.0
    return out


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume SMA for context."""
    out = df.copy()
    out["vol_sma20"] = out["Volume"].rolling(20).mean()
    return out


def add_support_resistance(df: pd.DataFrame, window: int = 10,
                           max_levels: int = 4) -> pd.DataFrame:
    """Flag swing-high resistance / swing-low support pivots (causal).

    A pivot high at index i requires ``2*window+1`` candles (window on each
    side), so the flag at i only depends on candles up to i+window. For live
    signalling we therefore shift flags back by ``window`` so that at time t
    we only ever see pivots whose right side is already in the past.
    """
    out = df.copy()
    high, low = out["High"], out["Low"]
    piv_h = pd.Series(False, index=out.index)
    piv_l = pd.Series(False, index=out.index)
    for i in range(window, len(out) - window):
        segment_h = high.iloc[i - window: i + window + 1]
        segment_l = low.iloc[i - window: i + window + 1]
        if high.iloc[i] == segment_h.max() and (segment_h < high.iloc[i]).sum() >= 1:
            piv_h.iloc[i] = True
        if low.iloc[i] == segment_l.min() and (segment_l > low.iloc[i]).sum() >= 1:
            piv_l.iloc[i] = True
    # only reveal a pivot after its right window has elapsed
    out["pivot_high"] = piv_h.shift(window).eq(True)
    out["pivot_low"] = piv_l.shift(window).eq(True)
    out["pivot_high_price"] = high.where(piv_h).shift(window)
    out["pivot_low_price"] = low.where(piv_l).shift(window)

    # rolling nearest levels using only revealed pivots
    levels_high: list[float] = []
    levels_low: list[float] = []
    res = np.full(len(out), np.nan)
    sup = np.full(len(out), np.nan)
    for i in range(len(out)):
        if out["pivot_high"].iloc[i]:
            levels_high.append(float(high.iloc[i]))
            levels_high = levels_high[-30:]
        if out["pivot_low"].iloc[i]:
            levels_low.append(float(low.iloc[i]))
            levels_low = levels_low[-30:]
        price = float(close_i) if (close_i := out["Close"].iloc[i]) else float(close_i)
        above = [x for x in levels_high if x > price]
        below = [x for x in levels_low if x < price]
        res[i] = min(above) if above else np.nan
        sup[i] = max(below) if below else np.nan
    out["resistance"] = res
    out["support"] = sup
    return out


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full indicator suite used by the signal engine."""
    out = add_ema(df)
    out = add_bollinger(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_atr(out)
    out = add_volume(out)
    out = add_support_resistance(out)
    # convenience: distance from nearest S/R in ATR units
    atr_safe = out["atr"].replace(0, np.nan)
    out["dist_to_res_atr"] = (out["resistance"] - out["Close"]) / atr_safe
    out["dist_to_sup_atr"] = (out["Close"] - out["support"]) / atr_safe
    return out
