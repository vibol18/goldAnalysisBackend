"""Causal volatility models: realized vol, ATR regime, GARCH filter, Hurst.

Every function uses only information available at or before each bar.  GARCH
parameters are fixed (RiskMetrics-style) rather than fitted on the full sample,
so parameter estimation cannot peek at the future.  `fit_garch_parameters` is
provided for offline reporting only and is never used in features or signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.concepts.base import atr as atr_series


def log_returns(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    close = close.where(close > 0)
    return np.log(close).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def realized_vol(df: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.Series:
    ret = log_returns(df)
    vol = ret.rolling(window, min_periods=max(5, window // 2)).std()
    return vol * np.sqrt(252) if annualize else vol


def atr_percentile(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """Current ATR expressed as a percentile of its trailing distribution."""
    value = atr_series(df, 14)
    return value.rolling(lookback, min_periods=30).rank(pct=True)


def garch_filter(returns: pd.Series, omega: float = 1e-7, alpha: float = 0.06,
                 beta: float = 0.92, horizon: int = 1) -> pd.Series:
    """Recursive GARCH(1,1) conditional variance using fixed causal parameters.

    Returns annualised conditional volatility.  The recursion at bar i uses only
    the return and variance at i-1, so it never repaints.
    """
    r = pd.Series(returns, dtype=float).fillna(0.0).clip(-0.5, 0.5)
    var = float(np.var(r[:20])) if len(r) > 20 else (float(r.iloc[0] ** 2) if len(r) else 1e-5)
    var = max(var, 1e-8)
    out = np.empty(len(r), dtype=float)
    for i, value in enumerate(r.to_numpy()):
        out[i] = np.sqrt(max(var * 252.0, 1e-10)) * (1.0 if horizon == 1 else np.sqrt(horizon))
        var = omega + alpha * value ** 2 + beta * var
    return pd.Series(out, index=r.index, name="garch_vol")


def fit_garch_parameters(returns: pd.Series) -> dict[str, float]:
    """Offline MLE-style parameter estimate for reporting (not used in signals)."""
    r = pd.Series(returns, dtype=float).fillna(0.0).to_numpy()
    if len(r) < 100:
        return {"omega": 1e-7, "alpha": 0.06, "beta": 0.92, "note": "insufficient data; defaults"}
    base = float(np.var(r[: max(50, len(r) // 2)])) or 1e-6
    grid = np.linspace(0.02, 0.15, 14)
    best = (np.inf, 0.06, 0.92)
    for a in grid:
        b = 0.97 - a
        var = base
        neg_log = 0.0
        for value in r:
            v = max(1e-8, base * (1 - a - b) + a * value ** 2 + b * var)
            neg_log += 0.5 * (np.log(v) + value ** 2 / v)
            var = v
        if neg_log < best[0]:
            best = (neg_log, float(a), float(b))
    return {"omega": float(base * (1 - best[1] - best[2])), "alpha": best[1],
            "beta": best[2], "note": "offline estimate only; not used in features"}


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Variance-of-differences Hurst estimate: <0.5 mean-reverting, >0.5 trending."""
    x = pd.Series(series, dtype=float).dropna()
    if len(x) < max_lag * 3:
        return 0.5
    lags = np.arange(2, min(max_lag, len(x) // 3) + 1)
    scales = []
    for lag in lags:
        diff = x.diff(lag).dropna()
        scales.append(float(diff.std()) if len(diff) > 5 else np.nan)
    scales = np.asarray(scales)
    valid = np.isfinite(scales) & (scales > 0)
    if valid.sum() < 3:
        return 0.5
    slope = np.polyfit(np.log(lags[valid]), np.log(scales[valid]), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(series: pd.Series, window: int = 120, max_lag: int = 15) -> pd.Series:
    values = pd.Series(series, dtype=float)
    out = pd.Series(np.nan, index=values.index, name="hurst")
    for i in range(window, len(values) + 1):
        out.iloc[i - 1] = hurst_exponent(values.iloc[i - window:i], max_lag=max_lag)
    return out


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time volatility feature block for the ML pipeline."""
    close = pd.to_numeric(df["Close"], errors="coerce")
    price = close.replace(0, np.nan)
    a = atr_series(df, 14)
    level = price.where(price > 0)
    atr_pct = (a / level * 100.0).replace([np.inf, -np.inf], np.nan)
    pctile = atr_percentile(df)
    rv20 = realized_vol(df, 20)
    rv60 = realized_vol(df, 60)
    garch = garch_filter(log_returns(df))
    ret = log_returns(df)
    out = pd.DataFrame(index=df.index)
    out["vol_atr_pct"] = atr_pct
    out["vol_atr_percentile"] = pctile
    out["vol_realized_20"] = rv20
    out["vol_realized_60"] = rv60
    out["vol_garch_forecast"] = garch
    out["vol_realized_ratio"] = (rv20 / rv60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["vol_expected_daily_range"] = a
    out["vol_autocorr_1"] = ret.rolling(60, min_periods=30).corr(ret.shift(1))
    out["vol_hurst"] = rolling_hurst(close, window=120, max_lag=12).shift(1)
    regime = pd.Series("normal", index=df.index, dtype=object)
    regime[pctile < 0.3] = "low"
    regime[pctile > 0.7] = "high"
    regime[rv20 > rv20.rolling(120, min_periods=60).quantile(0.85)] = "high"
    out["vol_regime_code"] = regime.map({"low": 0.0, "normal": 1.0, "high": 2.0}).astype(float)
    out["vol_hurst_class_code"] = np.where(out["vol_hurst"] > 0.55, 1.0,
                                    np.where(out["vol_hurst"] < 0.45, -1.0, 0.0))
    return out.replace([np.inf, -np.inf], np.nan)
