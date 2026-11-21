"""XGBoost direction model with strict walk-forward validation.

No leakage guarantees:
  * features are causal (see features.build_features);
  * walk-forward uses expanding windows — a model trained at fold k only
    sees rows strictly before the rows it predicts;
  * a final production model may be fit on ALL historical data because at
    inference time all of that data is in the past.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.ml.features import build_features, make_supervised

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Out-of-sample predictions produced by walk-forward validation."""

    oof_predictions: pd.Series  # index = bar time, value = P(price up)
    oof_labels: pd.Series
    folds: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if len(self.oof_predictions) == 0:
            return 0.0
        pred = (self.oof_predictions >= 0.5).astype(int)
        return float((pred == self.oof_labels.astype(int)).mean())

    @property
    def brier(self) -> float:
        if len(self.oof_predictions) == 0:
            return float("nan")
        p = self.oof_predictions.to_numpy(dtype=float)
        y = self.oof_labels.to_numpy(dtype=float)
        return float(np.mean((p - y) ** 2))


def _make_model(seed: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )


def walk_forward_validate(df: pd.DataFrame, n_folds: int = 5,
                          min_train_bars: int = 500,
                          horizon: int | None = None) -> WalkForwardResult:
    """Expanding-window walk-forward CV; returns strictly out-of-sample probs."""
    X, y = make_supervised(df, horizon)
    if len(X) < min_train_bars + 50:
        raise ValueError(
            f"not enough data for walk-forward ({len(X)} rows < {min_train_bars + 50})"
        )

    folds = np.linspace(min_train_bars, len(X), n_folds + 1).astype(int)
    oof_pred = pd.Series(np.nan, index=X.index, name="ml_prob")
    fold_stats: list[dict[str, Any]] = []

    for k in range(n_folds):
        train_end = folds[k]
        test_end = folds[k + 1]
        if test_end - train_end < 20 or train_end < 200:
            continue
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        if y_tr.nunique() < 2:
            continue
        model = _make_model(seed=42 + k)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        oof_pred.iloc[train_end:test_end] = proba
        acc = float(((proba >= 0.5).astype(int) == y_te.astype(int).to_numpy()).mean())
        fold_stats.append({
            "fold": k + 1,
            "train_rows": int(train_end),
            "test_rows": int(test_end - train_end),
            "accuracy": round(acc, 4),
        })
        logger.info("walk-forward fold %s acc=%.3f", k + 1, acc)

    valid = oof_pred.notna()
    return WalkForwardResult(
        oof_predictions=oof_pred[valid],
        oof_labels=y[oof_pred.index[valid]] if valid.any() else pd.Series(dtype=float),
        folds=fold_stats,
    )


class DirectionModel:
    """Trained (lazily, cached) XGBoost model + live probability helper."""

    def __init__(self) -> None:
        self._model: Any = None
        self._trained_at: float = 0.0
        self._feature_cols: list[str] = []
        self._lock = threading.Lock()
        self._last_wf: WalkForwardResult | None = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def walk_forward(self) -> WalkForwardResult | None:
        return self._last_wf

    def train(self, df: pd.DataFrame, force: bool = False,
              n_folds: int = 5) -> WalkForwardResult | None:
        """Train production model on all rows; run walk-forward for honesty metrics."""
        with self._lock:
            if self._model is not None and not force and (time.time() - self._trained_at < 3600):
                return self._last_wf
            X, y = make_supervised(df)
            if len(X) < 400 or y.nunique() < 2:
                logger.warning("insufficient data to train model (%s rows)", len(X))
                return None
            try:
                self._last_wf = walk_forward_validate(df, n_folds=n_folds)
            except Exception as exc:  # noqa: BLE001
                logger.warning("walk-forward failed: %s", exc)
                self._last_wf = None
            model = _make_model()
            model.fit(X, y)
            self._model = model
            self._feature_cols = list(X.columns)
            self._trained_at = time.time()
            from app.services import db

            brier = self._last_wf.brier if self._last_wf is not None else None
            db.set_model_metadata("direction-model", {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "training_rows": len(X),
                "feature_columns": self._feature_cols,
                "horizon": config.ML_HORIZON,
                "walk_forward_folds": len(self._last_wf.folds)
                if self._last_wf is not None else 0,
                "walk_forward_accuracy": self._last_wf.accuracy
                if self._last_wf is not None else None,
                "walk_forward_brier": brier if brier is not None and np.isfinite(brier)
                else None,
            })
            return self._last_wf

    def predict_proba(self, df: pd.DataFrame) -> float | None:
        """P(close higher in N bars) for the most recent bar, or None."""
        if self._model is None:
            return None
        X = build_features(df)
        if X.empty:
            return None
        last = X.iloc[[-1]].fillna(0.0)
        try:
            return float(self._model.predict_proba(last)[0][1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("predict failed: %s", exc)
            return None


# module-level singleton so signals reuse a trained model
MODEL = DirectionModel()
