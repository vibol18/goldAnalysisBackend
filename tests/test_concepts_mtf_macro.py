"""Unit tests for the causal MTF and macro concept detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.concepts import detect_all, load_families
from app.concepts.base import (
    DETECTORS,
    DETECTOR_CATEGORIES,
    REQUIRED_FIELDS,
    serialise,
)
from app.concepts.macro import (
    detect_dxy_correlation,
    detect_event_risk,
    detect_macro_regime,
    detect_news_sentiment,
    detect_real_yield_proxy,
    detect_risk_regime,
    detect_us10y_correlation,
)
from app.concepts.mtf import (
    detect_htf_level,
    detect_htf_veto,
    detect_ltf_entry_confirmation,
    detect_mtf_bias,
)

MTF_NAMES = ("mtf_bias", "htf_veto", "htf_level", "ltf_entry_confirmation")
MACRO_NAMES = (
    "dxy_correlation", "us10y_correlation", "real_yield_proxy",
    "risk_regime", "news_sentiment", "event_risk", "macro_regime",
)


def _ohlc(n: int, start: float, drift: float, freq: str,
          t0: str = "2026-01-01", wick: float = 0.4) -> pd.DataFrame:
    close = start + drift * np.arange(n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range(t0, periods=n, freq=freq)
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) + wick,
        "Low": np.minimum(open_, close) - wick,
        "Close": close,
        "Volume": np.full(n, 1000.0),
    }, index=index)


def _series(n: int, start: float, step: float, freq: str, t0: str) -> pd.DataFrame:
    index = pd.date_range(t0, periods=n, freq=freq)
    return pd.DataFrame({"Close": start + step * np.arange(n)}, index=index)


def _ltf_with_displacement(n: int = 20, t0: str = "2026-02-01 00:00") -> pd.DataFrame:
    index = pd.date_range(t0, periods=n, freq="15min")
    close = 1970.0 + 0.2 * np.arange(n)
    open_ = close.copy()
    open_[-1] = close[-2]
    close[-1] = close[-2] + 12.0
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) + 0.2,
        "Low": np.minimum(open_, close) - 0.2,
        "Close": close,
        "Volume": np.full(n, 500.0),
    }, index=index)


def _context() -> dict:
    base = _ohlc(30, 1980.0, 1.5, "1h", "2026-02-01")
    return {
        "htf_1d": _ohlc(40, 2100.0, -8.0, "1D", "2026-01-01"),
        "htf_4h": _ohlc(30, 2040.0, -3.0, "4h", "2026-01-25"),
        "ltf_15m": _ltf_with_displacement(),
        "dxy_df": _series(20, 104.0, -0.2, "1h", "2026-02-01"),
        "us10y_df": _series(20, 4.4, 0.02, "1h", "2026-02-01"),
        "cpi_yoy": 3.0,
        "spx_df": _series(20, 5000.0, -8.0, "1h", "2026-02-01"),
        "news": {"sentiment": 0.6, "label": "dovish Fed"},
        "events": [
            {"time": "2026-02-02 12:00:00", "impact": "high", "name": "CPI"},
            {"time": "2026-03-01 12:00:00", "impact": "medium", "name": "FOMC"},
        ],
    }


def test_all_detectors_registered_with_correct_categories():
    load_families()
    for name in MTF_NAMES:
        assert name in DETECTORS
        assert DETECTOR_CATEGORIES[name] == "mtf"
    for name in MACRO_NAMES:
        assert name in DETECTORS
        assert DETECTOR_CATEGORIES[name] == "macro"


def test_mtf_bias_top_down_bearish_and_htf_veto():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    daily = _ohlc(40, 2100.0, -8.0, "1D", "2026-01-01")
    four_h = _ohlc(30, 2040.0, -3.0, "4h", "2026-01-25")

    bias = detect_mtf_bias(base, htf_1d=daily, htf_4h=four_h)
    assert len(bias) == 1
    assert bias[0].direction == "bearish"
    assert bias[0].meta["timeframes"]["1d"]["direction"] == "bearish"
    assert bias[0].meta["biases"]["1h"] == "bullish"
    assert bias[0].meta["bias"] == "bearish"

    vetoes = detect_htf_veto(base, htf_1d=daily, htf_4h=four_h)
    assert len(vetoes) == 1
    assert vetoes[0].direction == "bearish"
    assert "veto" in vetoes[0].notes.lower()
    assert vetoes[0].meta["vetoed_direction"] == "bullish"


def test_htf_future_bars_do_not_leak():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    daily = _ohlc(40, 2100.0, -8.0, "1D", "2026-01-01")
    future = _ohlc(15, 2000.0, 10.0, "1D", "2026-02-10")
    leaked = pd.concat([daily, future])

    original = detect_mtf_bias(base, htf_1d=daily)[0].to_dict()
    with_future = detect_mtf_bias(base, htf_1d=leaked)[0].to_dict()
    assert original == with_future

    veto_original = detect_htf_veto(base, htf_1d=daily)[0].to_dict()
    veto_future = detect_htf_veto(base, htf_1d=leaked)[0].to_dict()
    assert veto_original == veto_future

    level_original = detect_htf_level(base, htf_1d=daily)[0].to_dict()
    level_future = detect_htf_level(base, htf_1d=leaked)[0].to_dict()
    assert level_original == level_future


def test_htf_level_and_ltf_confirmation():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    daily = _ohlc(40, 2100.0, -8.0, "1D", "2026-01-01")
    levels = detect_htf_level(base, htf_1d=daily)
    assert len(levels) == 1
    assert levels[0].meta["timeframe"] == "1d"
    assert levels[0].meta["kind"] in ("support", "resistance")
    assert levels[0].top >= levels[0].bottom

    ltf = _ltf_with_displacement(t0="2026-02-09 00:00")

    confirmations = detect_ltf_entry_confirmation(base, ltf_15m=ltf)
    assert len(confirmations) == 1
    assert confirmations[0].direction == "bullish"
    assert confirmations[0].meta["confirm_type"] == "displacement"


def test_core_frame_truncation_causality():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    cut = 12
    visible = base.iloc[:cut]
    dxy_visible = _series(cut, 104.0, -0.2, "1h", "2026-02-09 00:00")
    dxy_future = _series(6, 101.0, 3.0, "1h", "2026-02-09 12:00")
    dxy_all = pd.concat([dxy_visible, dxy_future])

    asof = detect_dxy_correlation(visible, dxy_df=dxy_visible)[0].to_dict()
    with_future = detect_dxy_correlation(visible, dxy_df=dxy_all)[0].to_dict()
    assert asof == with_future

    edited = base.copy()
    edited.iloc[cut:, edited.columns.get_loc("Close")] *= 2.0
    before = detect_mtf_bias(visible, htf_1d=_ohlc(40, 2100.0, -8.0, "1D"))[0].to_dict()
    after = detect_mtf_bias(edited.iloc[:cut], htf_1d=_ohlc(40, 2100.0, -8.0, "1D"))[0].to_dict()
    assert before == after


def test_macro_correlations():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    rising_dxy = _series(12, 100.0, 1.5, "1h", "2026-02-09 00:00")
    falling_dxy = _series(12, 115.0, -1.5, "1h", "2026-02-09 00:00")
    assert detect_dxy_correlation(base, dxy_df=rising_dxy)[0].direction == "bearish"
    assert detect_dxy_correlation(base, dxy_df=falling_dxy)[0].direction == "bullish"

    rising_yield = _series(12, 4.0, 0.03, "1h", "2026-02-09 00:00")
    falling_yield = _series(12, 4.4, -0.03, "1h", "2026-02-09 00:00")
    assert detect_us10y_correlation(base, us10y_df=rising_yield)[0].direction == "bearish"
    assert detect_us10y_correlation(base, us10y_df=falling_yield)[0].direction == "bullish"
    assert detect_real_yield_proxy(base, us10y_df=falling_yield, cpi_yoy=3.0)[0].direction == "bullish"
    assert detect_real_yield_proxy(base, us10y_df=rising_yield, cpi_yoy=3.0)[0].direction == "bearish"


def test_macro_risk_news_and_regime():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    selloff = _series(12, 5000.0, -30.0, "1h", "2026-02-09 00:00")
    rally = _series(12, 4700.0, 30.0, "1h", "2026-02-09 00:00")
    assert detect_risk_regime(base, spx_df=selloff)[0].direction == "bullish"
    assert detect_risk_regime(base, spx_df=rally)[0].direction == "bearish"

    bullish_news = detect_news_sentiment(base, news={"sentiment": 0.7, "label": "dovish"})
    bearish_news = detect_news_sentiment(base, news={"sentiment": -0.6, "label": "hawkish"})
    assert bullish_news[0].direction == "bullish"
    assert "dovish" in bullish_news[0].notes
    assert bearish_news[0].direction == "bearish"

    regime = detect_macro_regime(
        base,
        dxy_df=_series(12, 100.0, -1.0, "1h", "2026-02-09 00:00"),
        us10y_df=_series(12, 4.4, -0.02, "1h", "2026-02-09 00:00"),
        spx_df=selloff,
        news={"sentiment": 0.5, "label": "safe haven"},
    )
    assert len(regime) == 1
    assert regime[0].direction == "bullish"
    assert set(regime[0].meta["components"]) == {
        "dxy_correlation", "us10y_correlation", "risk_regime", "news_sentiment",
    }


def test_event_risk_window_and_veto_semantics():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    now = pd.Timestamp(base.index[-1])
    near = [{"time": (now + pd.Timedelta(hours=12)).isoformat(),
             "impact": "high", "name": "US CPI"}]
    far = [{"time": (now + pd.Timedelta(hours=48)).isoformat(),
            "impact": "high", "name": "US CPI"}]
    medium = [{"time": (now + pd.Timedelta(hours=1)).isoformat(),
               "impact": "medium", "name": "PMI"}]

    soon = detect_event_risk(base, events=near)
    assert len(soon) == 1
    assert soon[0].direction == "neutral"
    assert "US CPI" in soon[0].notes
    assert soon[0].meta["hours_away"] == pytest.approx(12.0)
    assert detect_event_risk(base, events=far) == []
    assert detect_event_risk(base, events=medium) == []


def test_missing_frames_and_short_frames_are_graceful():
    base = _ohlc(24, 1980.0, 1.5, "1h", "2026-02-09 00:00")
    assert detect_htf_veto(base) == []
    assert detect_htf_level(base) == []
    assert detect_ltf_entry_confirmation(base) == []
    assert detect_dxy_correlation(base) == []
    assert detect_us10y_correlation(base) == []
    assert detect_real_yield_proxy(base, us10y_df=_series(12, 4.0, 0.01, "1h", "2026-02-09")) == []
    assert detect_risk_regime(base) == []
    assert detect_news_sentiment(base) == []
    assert detect_event_risk(base) == []
    assert detect_macro_regime(base) == []

    tiny = _ohlc(3, 2000.0, 0.5, "1h", "2026-02-09 00:00")
    for detector in (detect_mtf_bias, detect_htf_veto, detect_htf_level,
                     detect_ltf_entry_confirmation, detect_macro_regime):
        result = detector(tiny)
        assert isinstance(result, list)

    with_nan = base.copy()
    with_nan.iloc[-1, with_nan.columns.get_loc("Close")] = np.nan
    assert isinstance(detect_mtf_bias(with_nan), list)
    assert isinstance(detect_news_sentiment(with_nan, news={"sentiment": 0.5, "label": "x"}), list)


def test_detect_all_contract_fields_and_categories():
    base = _ohlc(30, 1980.0, 1.5, "1h", "2026-02-01")
    results = serialise(detect_all(base, context=_context(), families=("mtf", "macro")))
    assert results
    for item in results:
        for field in REQUIRED_FIELDS:
            assert field in item, f"missing {field} on {item.get('type')}"
        assert item["direction"] in ("bullish", "bearish", "neutral")
        assert item["status"] in ("active", "mitigated", "invalidated")
        assert 0.0 <= item["strength"] <= 100.0
        assert item["category"] in ("mtf", "macro")

    emitted = {item["type"] for item in results}
    assert set(MTF_NAMES + MACRO_NAMES) <= emitted
