"""Learned weights with honest baselines, calibration, ensemble, meta-labeling
and TreeSHAP explanations.

All data splits are strictly time-ordered.  A gradient-boosted model is kept
only when it beats a logistic-regression baseline out of sample.  Probabilities
are calibrated on a *separate* validation slice, so a stated 70% confidence is
grounded in historical frequencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

UNCALIBRATED = "none"


@dataclass
class ModelBundle:
    kind: str
    model: Any
    scaler: StandardScaler | None = None
    calibrator: Any = None
    calibration_method: str = UNCALIBRATED
    ensemble: list[Any] = field(default_factory=list)
    meta_model: Any = None
    meta_scaler: StandardScaler | None = None
    feature_names: list[str] = field(default_factory=list)
    oos_metrics: dict[str, Any] = field(default_factory=dict)
    importance: list[dict[str, Any]] = field(default_factory=list)
    trained_at: str | None = None

    def _features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = X.to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return values

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> float:
        values = self._features(X)
        if self.kind == "logistic":
            raw = float(self.model.predict_proba(values[-1:])[0][1])
        else:
            raw = float(self.model.predict_proba(values[-1:])[0][1])
        return self.apply_calibration(raw)

    def predict_many(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = self._features(X)
        raw = self.model.predict_proba(values)[:, 1]
        return np.array([self.apply_calibration(float(p)) for p in raw])

    def apply_calibration(self, raw: float) -> float:
        if self.calibrator is None:
            return float(np.clip(raw, 0.0, 1.0))
        if self.calibration_method == "isotonic":
            return float(np.clip(self.calibrator.predict([raw])[0], 0.0, 1.0))
        logit = np.log(max(raw, 1e-6) / max(1 - raw, 1e-6))
        return float(np.clip(self.calibrator.predict_proba([[logit]])[0][1], 0.0, 1.0))

    def ensemble_prediction(self, X: pd.DataFrame | np.ndarray) -> dict[str, float | None]:
        if not self.ensemble:
            return {"mean": None, "disagreement": None}
        values = self._features(X)[-1:]
        preds = [float(m.predict_proba(values)[0][1]) for m in self.ensemble]
        return {
            "mean": round(float(np.mean(preds)), 4),
            "disagreement": round(float(np.std(preds)), 4),
        }

    def meta_probability(self, X: pd.DataFrame | np.ndarray) -> float | None:
        if self.meta_model is None:
            return None
        values = X.to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if self.meta_scaler is not None:
            values = self.meta_scaler.transform(values)
        return float(self.meta_model.predict_proba(values[-1:])[0][1])

    def shap_factors(self, row: pd.DataFrame | np.ndarray, top_n: int = 6) -> list[dict[str, Any]]:
        """TreeSHAP contributions (XGBoost) or linear contributions (baseline)."""
        values = self._features(row)
        names = self.feature_names or [f"f{i}" for i in range(values.shape[1])]
        if self.kind == "xgb":
            try:
                import xgboost as xgb
                contrib = self.model.get_booster().predict(
                    xgb.DMatrix(values[-1:]), pred_contribs=True
                )[0][:-1]
                items = sorted(zip(names, contrib), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
                return [{"factor": str(k), "impact": round(float(v), 4),
                         "direction": "up" if v > 0 else "down"} for k, v in items]
            except Exception:  # noqa: BLE001 - fall back to gain importance
                pass
        gains = {g["factor"]: g.get("gain", 0.0) for g in self.importance}
        items = sorted(gains.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        return [{"factor": k, "impact": round(float(v), 4),
                 "direction": "up" if v > 0 else "down"} for k, v in items]


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "auc": None, "log_loss": None, "brier": None, "accuracy": None, "samples": int(len(y_true)),
    }
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return out
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    out["auc"] = round(float(roc_auc_score(y_true, clipped)), 4)
    out["log_loss"] = round(float(log_loss(y_true, clipped)), 4)
    out["brier"] = round(float(brier_score_loss(y_true, clipped)), 4)
    out["accuracy"] = round(float(((clipped >= 0.5).astype(int) == y_true).mean()), 4)
    return out


def _xgb(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=220, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
        eval_metric="logloss", random_state=seed, n_jobs=2, tree_method="hist",
    )


def _time_splits(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = int(n * 0.6), int(n * 0.8)
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def _add_calibration(bundle: ModelBundle, raw_valid: np.ndarray, y_valid: np.ndarray,
                     raw_test: np.ndarray, y_test: np.ndarray) -> None:
    """Fit isotonic (large sample) or Platt (small sample) on validation only."""
    if len(np.unique(y_valid)) < 2 or len(y_valid) < 40:
        bundle.calibration_method = UNCALIBRATED
        bundle.oos_metrics["calibrated"] = _metrics(y_test, raw_test)
        return
    if len(y_valid) >= 300:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_valid, y_valid)
        bundle.calibrator, bundle.calibration_method = iso, "isotonic"
    else:
        logit = np.log(np.clip(raw_valid, 1e-6, 1 - 1e-6) /
                       np.clip(1 - raw_valid, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        platt.fit(logit, y_valid)
        bundle.calibrator, bundle.calibration_method = platt, "platt"
    bundle.oos_metrics["calibrated"] = _metrics(
        y_test, np.array([bundle.apply_calibration(float(p)) for p in raw_test])
    )


def _permutation_importance(bundle: ModelBundle, X: pd.DataFrame, y: pd.Series,
                            top_n: int = 20) -> list[dict[str, Any]]:
    if len(X) < 30:
        return []
    base = bundle.predict_many(X)
    base_auc = roc_auc_score(y, base) if len(np.unique(y)) > 1 else 0.5
    rows = []
    rng = np.random.default_rng(42)
    for col in X.columns:
        shuffled = X.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        auc = roc_auc_score(y, bundle.predict_many(shuffled)) if len(np.unique(y)) > 1 else 0.5
        rows.append({"factor": str(col), "gain": round(float(base_auc - auc), 5)})
    return sorted(rows, key=lambda r: abs(r["gain"]), reverse=True)[:top_n]


def fit_model_bundle(X: pd.DataFrame, y: pd.Series,
                     meta_X: pd.DataFrame | None = None,
                     meta_y: pd.Series | None = None,
                     min_rows: int = 200) -> ModelBundle | None:
    """Train, compare, calibrate, ensemble and report. None if data is insufficient."""
    frame = pd.concat([X, y.rename("label")], axis=1).dropna()
    if len(frame) < min_rows or frame["label"].nunique() < 2:
        return None
    X, y = frame[frame.columns.drop("label")], frame["label"].astype(int)
    tr, va, te = _time_splits(len(X))
    if len(te) < 30 or y.iloc[tr].nunique() < 2:
        return None

    scaler = StandardScaler().fit(X.iloc[tr].to_numpy())
    Xs = pd.DataFrame(scaler.transform(X.to_numpy()), columns=X.columns, index=X.index)
    yv = y.to_numpy()

    baseline = LogisticRegression(max_iter=2000, C=0.5).fit(Xs.iloc[tr].to_numpy(), yv[tr])
    boost = _xgb().fit(Xs.iloc[tr], yv[tr])

    base_oos = _metrics(yv[np.r_[va, te]], baseline.predict_proba(Xs.iloc[np.r_[va, te]])[:, 1])
    boost_oos = _metrics(yv[np.r_[va, te]], boost.predict_proba(Xs.iloc[np.r_[va, te]])[:, 1])
    keep_boost = (boost_oos["auc"] or 0) > (base_oos["auc"] or 0) + 0.005
    chosen = boost if keep_boost else baseline

    bundle = ModelBundle(
        kind="xgb" if keep_boost else "logistic",
        model=chosen, scaler=scaler, feature_names=list(X.columns),
    )
    bundle.oos_metrics = {
        "baseline": base_oos, "boosted": boost_oos, "kept_model": bundle.kind,
        "selection_rule": "XGBoost kept only if OOS AUC beats logistic by >0.005",
    }

    raw_valid = chosen.predict_proba(Xs.iloc[va])[:, 1]
    raw_test = chosen.predict_proba(Xs.iloc[te])[:, 1]
    _add_calibration(bundle, raw_valid, yv[va], raw_test, yv[te])

    bundle.ensemble = [
        _xgb(seed=7).fit(Xs.iloc[np.r_[tr, va]], yv[np.r_[tr, va]]),
        ExtraTreesClassifier(n_estimators=250, min_samples_leaf=5, max_features=0.7,
                             random_state=11, n_jobs=2).fit(Xs.iloc[np.r_[tr, va]], yv[np.r_[tr, va]]),
    ]
    ens_test = np.mean([m.predict_proba(Xs.iloc[te])[:, 1] for m in bundle.ensemble], axis=0)
    bundle.oos_metrics["ensemble"] = _metrics(yv[te], ens_test)

    if meta_X is not None and meta_y is not None and len(meta_X) >= 80:
        joined = pd.concat([meta_X, meta_y.rename("label")], axis=1).dropna()
        if len(joined) > 80 and joined["label"].nunique() > 1:
            mx, my = joined[joined.columns.drop("label")], joined["label"].astype(int)
            mt, mv = _time_splits(len(mx))[0], np.r_[_time_splits(len(mx))[1], _time_splits(len(mx))[2]]
            if my.iloc[mv].nunique() > 1:
                ms = StandardScaler().fit(mx.iloc[mt].to_numpy())
                bundle.meta_scaler = ms
                bundle.meta_model = LogisticRegression(max_iter=1500, C=0.3).fit(
                    ms.transform(mx.iloc[mt].to_numpy()), my.iloc[mt].to_numpy()
                )
                bundle.oos_metrics["meta_labeler"] = _metrics(
                    my.iloc[mv].to_numpy(),
                    bundle.meta_model.predict_proba(ms.transform(mx.iloc[mv].to_numpy()))[:, 1],
                )

    bundle.importance = _permutation_importance(bundle, X.iloc[te], y.iloc[te])
    return bundle


def proposed_trades(df: pd.DataFrame, signal: pd.Series,
                    features: pd.DataFrame | None = None,
                    sl_atr: float = 1.5, tp_atr: float = 3.0,
                    max_hold: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Rule proposals with causal features and forward outcome labels.

    The outcome (did TP beat SL within max_hold?) is a *label* measured on later
    bars; it is never available as a feature at the proposal bar.
    """
    from app.concepts.base import atr as atr_series

    a = atr_series(df, 14).to_numpy(float)
    high, low, close = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
    sig = pd.Series(signal, index=df.index).fillna(0)
    rows, labels = [], []
    for i in range(len(df) - max_hold - 1):
        direction = 0 if sig.iloc[i] == 0 else (1 if sig.iloc[i] > 0 else -1)
        if direction == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        entry = close[i]
        sl = entry - direction * sl_atr * a[i]
        tp = entry + direction * tp_atr * a[i]
        hit = 0
        for j in range(i + 1, i + 1 + max_hold):
            if direction > 0 and low[j] <= sl:
                hit = 0
                break
            if direction < 0 and high[j] >= sl:
                hit = 0
                break
            if direction > 0 and high[j] >= tp:
                hit = 1
                break
            if direction < 0 and low[j] <= tp:
                hit = 1
                break
        row: dict[str, Any] = {
            "signal_dir": float(direction),
            "entry_vs_atr": float((close[i] - float(np.mean(close[max(0, i - 10):i]))) / a[i]),
            "atr_pct": float(a[i] / close[i] * 100),
            "range_atr": float((high[i] - low[i]) / a[i]),
        }
        if features is not None:
            base = features.iloc[i]
            row["confluence"] = float(base.get("concept_mtf_bias", 0.0))
            for key in ("vol_atr_percentile", "p_trend_up", "p_trend_down",
                        "p_range", "p_high_vol_shock", "regime_code"):
                if key in features.columns:
                    row[key] = float(base[key])
        rows.append(row)
        labels.append(hit)
    return pd.DataFrame(rows), pd.Series(labels, name="meta_label")


def calibration_chart(y: pd.Series, prob: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    """Reliability diagram points: predicted vs observed frequency per bin."""
    if len(y) < 20:
        return []
    try:
        observed, predicted = calibration_curve(y.astype(int), np.clip(prob, 1e-6, 1 - 1e-6),
                                                n_bins=min(bins, max(4, len(y) // 20)))
    except ValueError:
        return []
    return [{"predicted": round(float(p), 3), "observed": round(float(o), 3)}
            for p, o in zip(predicted, observed)]
