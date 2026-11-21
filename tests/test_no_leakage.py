"""No-lookahead / data-leakage tests for the ML pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.features import build_features, build_label, make_supervised
from app.ml.model import walk_forward_validate


def make_df(n: int = 900, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.006, n)
    close = 1800 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1000, 90000, n).astype(float)},
        index=idx,
    )


def test_features_causal_future_edit_invariance():
    """Editing future bars must not change features on earlier bars."""
    df = make_df()
    f1 = build_features(df)
    df2 = df.copy()
    # hammer the last 10 bars
    df2.iloc[-10:, df2.columns.get_loc("Close")] *= 1.3
    df2.iloc[-10:, df2.columns.get_loc("High")] *= 1.3
    df2.iloc[-10:, df2.columns.get_loc("Low")] *= 1.3
    f2 = build_features(df2)
    n = len(df) - 60  # leave room for rolling windows at the tail
    pd.testing.assert_frame_equal(
        f1.iloc[:n], f2.iloc[:n], check_exact=False, rtol=1e-10,
    )


def test_label_uses_future_close_only():
    df = make_df(200)
    y = build_label(df, horizon=5)
    close = df["Close"]
    # label[i] == 1 iff close[i+5] > close[i]
    for i in (50, 100, 150):
        expected = 1.0 if close.iloc[i + 5] > close.iloc[i] else 0.0
        assert float(y.iloc[i]) == expected
    # last `horizon` labels are NaN (unknown future)
    assert y.iloc[-1] != y.iloc[-1]


def test_no_feature_columns_contain_label_info():
    """Features must not include future-derived columns by name."""
    X = build_features(make_df(300))
    for col in X.columns:
        assert "future" not in col.lower()
        assert "label" not in col.lower()
        assert "target" not in col.lower()


def test_walk_forward_is_out_of_sample():
    """Each fold's predictions come from a model trained only on earlier rows."""
    df = make_df(1200)
    res = walk_forward_validate(df, n_folds=4, min_train_bars=500)
    assert len(res.oof_predictions) > 100
    # first min_train_bars rows must NOT have predictions (never in-sample)
    X, _ = make_supervised(df)
    first_pred_idx = res.oof_predictions.index[0]
    train_boundary = X.index[500]
    assert first_pred_idx >= train_boundary
    assert res.folds, "expected fold statistics"
    for fold in res.folds:
        assert fold["test_rows"] > 0


def test_walk_forward_prediction_count_matches_labels():
    df = make_df(1000)
    res = walk_forward_validate(df, n_folds=3, min_train_bars=400)
    assert len(res.oof_predictions) == len(res.oof_labels)
    assert res.accuracy >= 0.0  # sanity: property computes
    _ = res.brier
