"""Endpoints for advanced intelligence (real macro data, models and validation)."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from app import config
from app.intelligence import data_sources as ds
from app.services import data as market_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Signals can be wrong. "
    "Never risk money you can't afford to lose."
)


@router.get("/macro")
def macro(force: bool = False) -> dict[str, Any]:
    """Real yields, breakeven, intermarket, COT, calendar and news with provenance."""
    try:
        return ds.macro_snapshot(force=force)
    except Exception as exc:  # noqa: BLE001
        logger.exception("macro snapshot failed")
        raise HTTPException(503, f"macro data unavailable: {exc}") from exc


@router.get("/series/{name}")
def series(name: str) -> dict[str, Any]:
    """A single real macro series, or N/A when unavailable."""
    loaders = {
        "real_yield": lambda: ds.fred_series(config.REAL_YIELD_SERIES).to_dict(include_frame=True),
        "breakeven": lambda: ds.fred_series(config.BREAKEVEN_SERIES).to_dict(include_frame=True),
        "cot": lambda: ds.cot_gold().to_dict(include_frame=True),
        "gold_silver": ds.gold_silver_ratio,
        "intermarket": ds.intermarket_snapshot,
        "calendar": ds.calendar_countdown,
        "news": ds.news_intelligence,
    }
    loader = loaders.get(name)
    if loader is None:
        raise HTTPException(404, f"unknown series: {name}")
    try:
        return loader()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"{name} unavailable: {exc}") from exc


@router.get("/volatility")
def volatility(timeframe: str = "1D") -> dict[str, Any]:
    """ATR percentile, realized/GARCH vol, regime and Hurst (causal)."""
    from app.intelligence import volatility as vol
    try:
        frame = market_data.get_ohlcv(config.GOLD_TICKER, timeframe).df
        if frame is None or frame.empty:
            return {"available": False, "reason": "gold data unavailable", "disclaimer": DISCLAIMER}
        features = vol.volatility_features(frame)
        last = features.iloc[-1]
        numeric = {k: (round(float(v), 4) if pd.notna(v) else None)
                   for k, v in last.items() if isinstance(v, (int, float, np.floating))}
        return {
            "available": True,
            "timeframe": timeframe,
            "source": market_data.get_ohlcv(config.GOLD_TICKER, timeframe).source,
            "updated_at": market_data.get_ohlcv(config.GOLD_TICKER, timeframe).last_updated,
            "current": {**numeric,
                        "vol_regime": {0: "low", 1: "normal", 2: "high"}.get(
                            int(last.get("vol_regime_code", 1)), "normal"),
                        "vol_hurst_class": {1: "trending", 0: "random", -1: "mean_reverting"}.get(
                            int(last.get("vol_hurst_class_code", 0)), "random")},
            "garch_parameters": vol.fit_garch_parameters(vol.log_returns(frame)),
            "disclaimer": DISCLAIMER,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"volatility unavailable: {exc}") from exc


@router.get("/regime")
def regime(timeframe: str = "1D") -> dict[str, Any]:
    """Causal forward-filtered regime with expectancy and trade weights."""
    from app.intelligence import regime as rg
    try:
        frame = market_data.get_ohlcv(config.GOLD_TICKER, timeframe).df
        if frame is None or frame.empty:
            return {"available": False, "reason": "gold data unavailable", "disclaimer": DISCLAIMER}
        labels = rg.detect_regime(frame)
        fwd = frame["Close"].pct_change().shift(-1)
        expectancy = rg.regime_expectancy(labels["regime"], fwd)
        weights = rg.regime_trade_weights(expectancy)
        last = labels.iloc[-1]
        return {
            "available": True,
            "timeframe": timeframe,
            "current": str(last["regime"]),
            "probabilities": {k: round(float(last[k]), 4) for k in labels.columns if k.startswith("p_")},
            "confidence": round(float(last["regime_confidence"]), 4),
            "expectancy": expectancy,
            "trade_weights": weights,
            "method": "forward-filtered Gaussian HMM (no lookahead, no Viterbi)",
            "disclaimer": DISCLAIMER,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"regime unavailable: {exc}") from exc


_MODEL_CACHE: dict[str, tuple[float, Any]] = {}


@router.get("/model")
def model_report(timeframe: str = "1D", refresh: bool = False) -> dict[str, Any]:
    """OOS model comparison, calibration, SHAP factors and uncertainty."""
    from app.intelligence import features_ext as fe
    from app.intelligence import models as ml
    from app.intelligence import regime as rg
    try:
        import time as _time
        from app.services.indicators import add_all
        now = _time.time()
        cached = _MODEL_CACHE.get(timeframe)
        if cached and not refresh and now - cached[0] < 3600:
            return cached[1]
        frame = market_data.get_ohlcv(config.GOLD_TICKER, timeframe).df
        if frame is None or len(frame) < 300:
            return {"available": False, "reason": "insufficient real history", "disclaimer": DISCLAIMER}
        X, y = fe.make_extended_supervised(frame, include_network=True)
        ind = add_all(frame)
        rule_signal = (ind["ema20"] > ind["ema50"]).astype(float)
        meta_X, meta_y = ml.proposed_trades(frame, rule_signal, features=X)
        bundle = ml.fit_model_bundle(X, y, meta_X=meta_X, meta_y=meta_y)
        if bundle is None:
            return {"available": False, "reason": "insufficient supervised rows", "disclaimer": DISCLAIMER}
        probability = bundle.predict_proba(X)
        ensemble = bundle.ensemble_prediction(X)
        labels = rg.detect_regime(frame)
        fwd = frame["Close"].pct_change().shift(-1)
        weights = rg.regime_trade_weights(rg.regime_expectancy(labels["regime"], fwd))
        regime_name = str(labels["regime"].iloc[-1])
        payload = {
            "available": True,
            "timeframe": timeframe,
            "kind": bundle.kind,
            "calibration_method": bundle.calibration_method,
            "calibrated_probability": round(probability, 4),
            "ensemble": ensemble,
            "meta_probability": bundle.meta_probability(meta_X) if not meta_X.empty else None,
            "shap_factors": bundle.shap_factors(X.iloc[[-1]]),
            "oos_metrics": bundle.oos_metrics,
            "calibration_chart": ml.calibration_chart(
                y.iloc[int(len(y) * 0.8):], bundle.predict_many(X.iloc[int(len(X) * 0.8):])
            ),
            "feature_importance": bundle.importance[:20],
            "current_regime": regime_name,
            "regime_weight": weights.get(regime_name, 0.0),
            "source": market_data.get_ohlcv(config.GOLD_TICKER, timeframe).source,
            "updated_at": market_data.get_ohlcv(config.GOLD_TICKER, timeframe).last_updated,
            "disclaimer": DISCLAIMER,
        }
        _MODEL_CACHE[timeframe] = (now, payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.exception("model report failed")
        raise HTTPException(503, f"model unavailable: {exc}") from exc


@router.get("/data-health")
def data_health() -> dict[str, Any]:
    """Provenance and freshness of every macro input (N/A when missing)."""
    snapshot = ds.macro_snapshot()
    gold = market_data.get_ohlcv(config.GOLD_TICKER, "1D")
    rows = []
    for key in ("real_yield", "breakeven_inflation", "intermarket", "cot", "calendar", "news"):
        item = snapshot.get(key, {})
        rows.append({
            "input": key,
            "source": item.get("source") or "N/A",
            "updated_at": item.get("updated_at") or "N/A",
            "available": bool(item.get("available")),
            "stale": bool(item.get("stale")),
        })
    rows.append({
        "input": "gold_ohlcv",
        "source": gold.source or "N/A",
        "updated_at": gold.last_updated,
        "available": not gold.df.empty,
        "stale": gold.stale,
    })
    return {"inputs": rows, "generated_at": snapshot["generated_at"], "disclaimer": DISCLAIMER}
