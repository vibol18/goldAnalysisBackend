"""Causal Gaussian regime filter with per-regime expectancy and trade gating.

Uses forward filtering only (never Viterbi smoothing), so a regime label at bar i
depends exclusively on data up to bar i.  Emission parameters are estimated on a
fixed warmup segment; no future data is used to fit them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from app.intelligence.volatility import atr_percentile, log_returns, realized_vol

REGIMES = ("trend_up", "trend_down", "range", "high_vol_shock")
DEFAULT_WEIGHTS = {"trend_up": 1.0, "trend_down": 1.0, "range": 0.5, "high_vol_shock": 0.0}


def _features(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["Close"], errors="coerce").where(lambda s: s > 0)
    ret = log_returns(df)
    r20 = ret.rolling(20, min_periods=10).sum()
    rv = realized_vol(df, 20)
    rv_z = ((rv - rv.rolling(120, min_periods=60).mean()) /
            rv.rolling(120, min_periods=60).std().replace(0, np.nan))
    ap = atr_percentile(df) * 2 - 1
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    slope = ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 1000).clip(-10, 10)
    out = pd.DataFrame({
        "momentum": r20.clip(-0.3, 0.3),
        "trend": slope.clip(-10, 10),
        "vol": rv_z.clip(-4, 4),
        "range": ap,
    })
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _label_states(means: np.ndarray, covs: np.ndarray) -> dict[int, str]:
    """Assign semantic names to latent states using their fitted statistics."""
    n = means.shape[0]
    vol_rank = np.argsort(np.abs(means[:, 2]))
    shock = int(vol_rank[-1])
    remaining = [i for i in range(n) if i != shock]
    remaining.sort(key=lambda i: means[i, 1])
    labels: dict[int, str] = {}
    if len(remaining) >= 3:
        labels[remaining[0]] = "trend_down"
        labels[remaining[-1]] = "trend_up"
        for i in remaining[1:-1]:
            labels[i] = "range"
    else:
        for i in remaining:
            labels[i] = "trend_up" if means[i, 1] >= 0 else "trend_down"
    labels[shock] = "high_vol_shock"
    if "range" not in labels.values():
        labels[remaining[len(remaining) // 2]] = "range"
    return labels


def detect_regime(df: pd.DataFrame, warmup: int = 200, n_states: int = 4) -> pd.DataFrame:
    """Return causal regime probabilities and labels aligned to `df`."""
    x = _features(df)
    n = len(x)
    out = pd.DataFrame(index=df.index)
    if n < 60:
        out["regime"] = "range"
        out["regime_confidence"] = 0.0
        for name in REGIMES:
            out[f"p_{name}"] = 1.0 if name == "range" else 0.0
        return out

    train_n = min(warmup, max(60, n // 2))
    train = x.iloc[:train_n].to_numpy()
    if len(np.unique(train.round(4), axis=0)) < n_states:
        out["regime"] = "range"
        out["regime_confidence"] = 0.0
        for name in REGIMES:
            out[f"p_{name}"] = 1.0 if name == "range" else 0.0
        return out
    gmm = GaussianMixture(n_components=n_states, covariance_type="diag",
                          reg_covar=1e-4, random_state=42, n_init=2, max_iter=150)
    gmm.fit(train)
    means, covs = gmm.means_, gmm.covariances_
    labels = _label_states(means, covs)

    stay = 0.96
    switch = (1.0 - stay) / max(1, n_states - 1)
    transition = np.full((n_states, n_states), switch)
    np.fill_diagonal(transition, stay)

    values = x.to_numpy()
    log_emission = np.empty((n, n_states))
    for k in range(n_states):
        variance = np.maximum(covs[k], 1e-4)
        log_emission[:, k] = -0.5 * np.sum(
            ((values - means[k]) ** 2) / variance + np.log(2 * np.pi * variance), axis=1
        )
    log_trans = np.log(transition + 1e-12)
    alpha = np.full(n_states, 1.0 / n_states)
    filtered = np.empty((n, n_states))
    for t in range(n):
        prior = alpha @ log_trans
        logits = prior + log_emission[t]
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        filtered[t] = weights
        alpha = weights

    names = [labels[k] for k in range(n_states)]
    for name in REGIMES:
        out[f"p_{name}"] = filtered[:, [i for i, nm in enumerate(names) if nm == name]].sum(axis=1) \
            if any(nm == name for nm in names) else 0.0
    out["regime"] = [names[int(i)] for i in np.argmax(filtered, axis=1)]
    out["regime_confidence"] = filtered.max(axis=1)
    out["regime_entropy"] = -np.sum(np.where(filtered > 0, filtered * np.log(filtered + 1e-12), 0), axis=1)
    return out


def regime_expectancy(regimes: pd.Series, forward_returns: pd.Series) -> dict[str, dict[str, float]]:
    """Historical expectancy per regime (measurement only, not a feature)."""
    r = pd.Series(forward_returns, dtype=float)
    out: dict[str, dict[str, float]] = {}
    for name in REGIMES:
        mask = regimes == name
        subset = r[mask].dropna()
        out[name] = {
            "samples": int(len(subset)),
            "expectancy_pct": round(float(subset.mean() * 100), 4) if len(subset) else None,
            "win_rate": round(float((subset > 0).mean() * 100), 2) if len(subset) else None,
        }
    return out


def regime_trade_weights(expectancy: dict[str, dict[str, float]],
                         reference_pct: float = 0.2) -> dict[str, float]:
    """Zero weight for non-positive expectancy; otherwise 0.5-1.0 by edge size."""
    weights: dict[str, float] = {}
    for name in REGIMES:
        exp = expectancy.get(name, {}).get("expectancy_pct")
        if exp is None or expectancy[name]["samples"] < 20:
            weights[name] = DEFAULT_WEIGHTS[name]
        elif exp <= 0:
            weights[name] = 0.0
        else:
            weights[name] = float(np.clip(0.5 + exp / max(2 * reference_pct, 1e-6), 0.5, 1.0))
    return weights


def apply_regime_gate(signal: pd.Series, regimes: pd.Series,
                      weights: dict[str, float]) -> pd.Series:
    """Set signal to HOLD (0) in regimes with no proven edge."""
    sig = pd.Series(signal, index=regimes.index, dtype=float).fillna(0.0)
    mult = regimes.map(weights).fillna(0.0)
    return (sig * mult).where(mult > 0, 0.0)
