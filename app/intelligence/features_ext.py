"""Extended, strictly causal feature pipeline.

Layers: existing indicator/concept features + volatility features + forward-filtered
regime probabilities + advanced concept states + point-in-time macro/positioning.
Macro series are as-of merged with explicit release lags, never shifted backwards.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.concepts import detect_all, load_families
from app.concepts.advanced import ADVANCED_TYPES
from app.intelligence import data_sources as ds
from app.intelligence import regime as rg
from app.intelligence import volatility as vol
from app.ml.features import build_features, build_label

FEATURE_COLUMNS_EXT: list[str] = []
RECENCY_BARS = 24


def advanced_concept_features(df: pd.DataFrame, concepts: list[Any] | None = None) -> pd.DataFrame:
    """Signed, decayed state of each advanced concept at every bar."""
    load_families()
    if concepts is None:
        concepts = detect_all(df, unlimited=True)
    concepts = [c for c in concepts if c.category == "advanced" and c.direction != "neutral"]
    index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True).tz_localize(None))
    out = pd.DataFrame(index=df.index)
    if not concepts:
        return out

    events: list[tuple[int, Any]] = []
    for c in concepts:
        try:
            ts = pd.Timestamp(c.end_time)
            if ts.tz is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            pos = index.searchsorted(ts, side="right") - 1
        except (TypeError, ValueError):
            continue
        if pos >= 0:
            events.append((pos, c))
    events.sort(key=lambda x: x[0])

    types = sorted(ADVANCED_TYPES)
    for t in types:
        out[f"adv_{t}_dir"] = 0.0
        out[f"adv_{t}_decay"] = 0.0
        out[f"adv_{t}_strength"] = 0.0
    state: dict[str, tuple[int, Any]] = {}
    cursor = 0
    for i in range(len(index)):
        while cursor < len(events) and events[cursor][0] <= i:
            pos, c = events[cursor]
            state[c.type] = (pos, c)
            cursor += 1
        for t, (pos, c) in state.items():
            age = max(0, i - pos)
            decay = 0.5 ** (age / RECENCY_BARS)
            sign = 1.0 if c.direction == "bullish" else -1.0
            out.iat[i, out.columns.get_loc(f"adv_{t}_dir")] = sign * decay
            out.iat[i, out.columns.get_loc(f"adv_{t}_decay")] = decay
            out.iat[i, out.columns.get_loc(f"adv_{t}_strength")] = float(c.strength) / 100.0 * decay
    return out


def load_macro_frames(include_network: bool = True) -> dict[str, pd.DataFrame | None]:
    """Fetch (cached) real macro frames; returns N/A-safe dictionary."""
    frames: dict[str, pd.DataFrame | None] = {
        "real_yield": None, "breakeven": None, "cot": None, "corr": None,
    }
    if not include_network:
        return frames
    try:
        frames["real_yield"] = ds.fred_series("DFII10").frame
    except Exception:  # noqa: BLE001
        pass
    try:
        frames["breakeven"] = ds.fred_series("T10YIE").frame
    except Exception:  # noqa: BLE001
        pass
    try:
        frames["cot"] = ds.cot_gold().frame
    except Exception:  # noqa: BLE001
        pass
    try:
        frames["corr"] = ds.intermarket_correlation_frame()
    except Exception:  # noqa: BLE001
        pass
    return frames


def macro_features(df: pd.DataFrame, frames: dict[str, pd.DataFrame | None] | None = None,
                   include_network: bool = False) -> pd.DataFrame:
    """Point-in-time macro features using as-of values with release lags."""
    if frames is None:
        frames = load_macro_frames(include_network=include_network)
    index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True).tz_localize(None))
    out = pd.DataFrame(index=df.index)

    def column(name: str, values: list[float]) -> None:
        out[name] = values

    real = frames.get("real_yield")
    breakeven = frames.get("breakeven")
    cot = frames.get("cot")
    corr = frames.get("corr")
    ry_vals, ry_chg = [], []
    be_vals = []
    cot_net, cot_pct, cot_chg = [], [], []
    corr_cols = list(corr.columns) if corr is not None and not corr.empty else []
    corr_vals: dict[str, list[float]] = {c: [] for c in corr_cols}

    for ts in index:
        value, _ = ds.as_of_value(real, ts, lag_days=1)
        ry_vals.append(value if value is not None else np.nan)
        value, prev_date = ds.as_of_value(real, ts - pd.Timedelta(days=20), lag_days=1)
        ry_chg.append((ry_vals[-1] - value) if value is not None and pd.notna(ry_vals[-1]) else np.nan)
        value, _ = ds.as_of_value(breakeven, ts, lag_days=1)
        be_vals.append(value if value is not None else np.nan)
        value, _ = ds.as_of_value(cot, ts, lag_days=3)
        cot_net.append(value if value is not None else np.nan)
        pct_frame = cot[["pct_3y"]] if cot is not None and "pct_3y" in cot else None
        value, _ = ds.as_of_value(pct_frame, ts, lag_days=3)
        cot_pct.append(value if value is not None else np.nan)
        chg_frame = cot[["weekly_change"]] if cot is not None and "weekly_change" in cot else None
        value, _ = ds.as_of_value(chg_frame, ts, lag_days=3)
        cot_chg.append(value if value is not None else np.nan)
        for c in corr_cols:
            value, _ = ds.as_of_value(corr[[c]], ts, lag_days=1)
            corr_vals[c].append(value if value is not None else np.nan)

    column("macro_real_yield", ry_vals)
    column("macro_real_yield_change", ry_chg)
    column("macro_breakeven", be_vals)
    column("macro_cot_net", cot_net)
    column("macro_cot_percentile", cot_pct)
    column("macro_cot_change", cot_chg)
    for c in corr_cols:
        column(f"macro_{c}", corr_vals[c])
    return out.replace([np.inf, -np.inf], np.nan)


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal regime probabilities and confidence."""
    labels = rg.detect_regime(df)
    out = labels.copy()
    out["regime_code"] = out["regime"].map({
        "trend_up": 1.0, "trend_down": -1.0, "range": 0.0, "high_vol_shock": -2.0,
    }).fillna(0.0)
    out = out.drop(columns=["regime"])
    return out.astype(float)


def build_extended_features(df: pd.DataFrame, macro_frames: dict[str, pd.DataFrame | None] | None = None,
                            include_base: bool = True, include_network: bool = False) -> pd.DataFrame:
    """Full causal feature matrix. Missing macro data becomes 0 (documented N/A)."""
    base = build_features(df) if include_base else pd.DataFrame(index=df.index)
    blocks = [
        base,
        vol.volatility_features(df),
        regime_features(df),
        advanced_concept_features(df),
        macro_features(df, frames=macro_frames, include_network=include_network),
    ]
    out = pd.concat(blocks, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    global FEATURE_COLUMNS_EXT
    FEATURE_COLUMNS_EXT = list(out.columns)
    return out


def make_extended_supervised(df: pd.DataFrame, horizon: int | None = None,
                             macro_frames: dict[str, pd.DataFrame | None] | None = None,
                             include_network: bool = False):
    """Return (X, y) where y is a future-return label, never a feature."""
    X = build_extended_features(df, macro_frames=macro_frames, include_network=include_network)
    y = build_label(df, horizon=horizon)
    aligned = pd.concat([X, y.rename("label")], axis=1).dropna(subset=["label"])
    return aligned.drop(columns=["label"]), aligned["label"]
