"""Feature engineering for the XGBoost direction model.

CRITICAL NO-LOOKAHEAD RULE: every feature at row ``i`` is computed from data
at or before ``i`` (indicators are causal; returns are shifted back). Labels
use future prices but are ONLY used as training targets, never as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import config

FEATURE_COLUMNS: list[str] = []


def concept_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features derived from detected concepts (no future information).

    Concepts are applied at their confirmation timestamp, then their state is
    carried forward. Distance features are recomputed per bar from the ATR at
    that bar, so editing a future candle cannot change an earlier row.
    """
    from app.concepts import detect_all

    out = pd.DataFrame(index=df.index)
    for col in (
        "concept_fvg_dist_atr", "concept_ob_dist_atr", "concept_in_discount",
        "concept_swept_liquidity", "concept_killzone", "concept_regime",
        "concept_bos", "concept_choch", "concept_mtf_bias",
    ):
        out[col] = 0.0
    try:
        concepts = detect_all(df, unlimited=True)
    except Exception:  # noqa: BLE001 - concept features are optional
        return out

    index_set = set(df.index)
    events: list[tuple[pd.Timestamp, Any]] = []
    for c in concepts:
        t = pd.Timestamp(c.end_time or c.start_time)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        if t in index_set:
            events.append((t, c))
    events.sort(key=lambda pair: pair[0])

    index = df.index
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    from app.concepts.base import in_killzone

    state: dict[str, Any] = {}
    state_bar: dict[str, int] = {}
    e = 0
    fvg_dist = np.zeros(len(index))
    ob_dist = np.zeros(len(index))
    discount = np.zeros(len(index))
    swept = np.zeros(len(index))
    killzone = np.zeros(len(index))
    regime = np.zeros(len(index))
    bos = np.zeros(len(index))
    choch = np.zeros(len(index))
    mtf_bias = np.zeros(len(index))

    for i, ts in enumerate(index):
        while e < len(events) and events[e][0] <= ts:
            c = events[e][1]
            state[c.type] = c
            state_bar[c.type] = i
            e += 1
        price = float(close.iloc[i])
        a = float(atr_s.iloc[i]) if np.isfinite(atr_s.iloc[i]) else max(price * 0.01, 1e-6)

        def zone_dist(concept: Any | None) -> float:
            if concept is None or concept.top is None or concept.bottom is None:
                return 0.0
            # Never use the detector's *final* status here: it may have been
            # invalidated by future bars. Geometry is stamped at confirmation.
            mid = (float(concept.top) + float(concept.bottom)) / 2.0
            value = (price - mid) / max(a, 1e-6)
            value = float(np.clip(value, -10.0, 10.0))
            return value if concept.direction == "bullish" else -value

        fvg_dist[i] = zone_dist(state.get("fvg"))
        ob_dist[i] = zone_dist(state.get("order_block"))
        pd_concept = state.get("premium_discount")
        if pd_concept is not None:
            discount[i] = (1.0 if pd_concept.direction == "bullish" else
                           -1.0 if pd_concept.direction == "bearish" else 0.0)
        sweep = state.get("liquidity_sweep")
        if sweep is not None:
            age = i - state_bar.get("liquidity_sweep", i)
            if age <= 8:
                swept[i] = 1.0 if sweep.direction == "bullish" else -1.0
        killzone[i] = 1.0 if in_killzone(ts) else 0.0
        market_regime = state.get("market_regime")
        if market_regime is not None:
            marker = str((market_regime.meta or {}).get("regime", ""))
            regime[i] = 1.0 if marker == "trending_up" else (
                -1.0 if marker == "trending_down" else 0.0)
        for key, target in (("bos", bos), ("choch", choch), ("mtf_bias", mtf_bias)):
            c = state.get(key)
            if c is not None:
                target[i] = 1.0 if c.direction == "bullish" else (
                    -1.0 if c.direction == "bearish" else 0.0)

    out["concept_fvg_dist_atr"] = fvg_dist
    out["concept_ob_dist_atr"] = ob_dist
    out["concept_in_discount"] = discount
    out["concept_swept_liquidity"] = swept
    out["concept_killzone"] = killzone
    out["concept_regime"] = regime
    out["concept_bos"] = bos
    out["concept_choch"] = choch
    out["concept_mtf_bias"] = mtf_bias
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to ``df`` index using only past data."""
    from app.services.indicators import add_all

    ind = add_all(df)
    out = pd.DataFrame(index=ind.index)
    close = ind["Close"]

    # lagged returns (1,2,3,5,10 bars) — shift(+) keeps causality
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_{lag}"] = close.pct_change(lag)

    # indicator levels (causal by construction)
    out["rsi"] = ind["rsi"] / 100.0
    out["rsi_slope"] = ind["rsi_slope"] / 10.0
    out["macd_norm"] = (ind["macd"] - ind["macd_signal"]) / ind["atr"].replace(0, np.nan)
    out["macd_hist_norm"] = ind["macd_hist"] / ind["atr"].replace(0, np.nan)
    out["bb_pct"] = ind["bb_pct"]
    out["bb_width"] = (ind["bb_high"] - ind["bb_low"]) / close

    # EMA relationships (normalized by ATR so they are scale-free)
    atr = ind["atr"].replace(0, np.nan)
    out["close_vs_ema20"] = (close - ind["ema20"]) / atr
    out["close_vs_ema50"] = (close - ind["ema50"]) / atr
    out["ema20_vs_ema50"] = (ind["ema20"] - ind["ema50"]) / atr
    out["ema50_vs_ema200"] = (ind["ema50"] - ind["ema200"]) / atr

    # volatility & structure
    out["atr_pct"] = ind["atr_pct"] / 100.0
    out["dist_to_res"] = ind["dist_to_res_atr"].clip(-10, 10)
    out["dist_to_sup"] = ind["dist_to_sup_atr"].clip(-10, 10)
    out["range_pos"] = ((close - ind["Low"]) /
                        (ind["High"] - ind["Low"]).replace(0, np.nan)).clip(0, 1)

    # volume context
    vol = ind["Volume"].replace(0, np.nan)
    out["vol_ratio"] = (vol / ind["vol_sma20"]).clip(0, 5)

    # day-of-week seasonality (causal calendar feature)
    out["dow"] = ind.index.dayofweek / 6.0

    out = pd.concat([out, concept_features(df)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in out.columns]
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = cols
    return out[cols]


def build_label(df: pd.DataFrame, horizon: int | None = None) -> pd.Series:
    """Binary label: 1 if close is higher ``horizon`` bars later.

    Labels are future-looking by design and must never enter the feature set.
    """
    h = horizon or config.ML_HORIZON
    future = df["Close"].shift(-h)
    return (future > df["Close"]).astype("float64").where(future.notna())


def make_supervised(df: pd.DataFrame, horizon: int | None = None
                    ) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with rows where the label is unknown removed.

    Feature NaNs are forward-filled (causal: uses only past values) before
    dropping early warmup rows — never bfilled, which would leak the future.
    """
    X = build_features(df).ffill()
    y = build_label(df, horizon)
    mask = y.notna() & X.notna().all(axis=1)
    return X[mask], y[mask]
