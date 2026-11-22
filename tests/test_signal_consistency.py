"""Consistency tests for R:R, factor accounting and HOLD explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.concepts.base import make_concept
from app.services.confluence import confluence
from app.services.explanation import build_explanation
from app.services.risk_reward import rr_text, risk_reward_ratios


def test_rr_from_entry_sl_each_target():
    ratios = risk_reward_ratios(3604.30, 3582.73, 3625.87, 3647.43)
    assert abs(ratios["risk_reward_tp1"] - 1.0) < 0.01
    assert abs(ratios["risk_reward_tp2"] - 2.0) < 0.01
    assert rr_text(ratios) == "1:1.0 / 1:2.0"


def test_rr_zero_risk_never_invents_a_ratio():
    ratios = risk_reward_ratios(3604.30, 3604.30, 3625.87, 3647.43)
    assert ratios["risk_reward_tp1"] == 0.0
    assert rr_text(ratios) == "N/A"


def _frame(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = 3600 + np.linspace(0, 10, n)
    return pd.DataFrame({
        "Open": close - 0.2, "High": close + 0.5,
        "Low": close - 0.5, "Close": close, "Volume": np.full(n, 100.0),
    }, index=idx)


def test_factor_count_uses_every_detected_factor():
    price = float(_frame()["Close"].iloc[-1])
    concepts = [
        make_concept("trend", "bullish", "2026-01-01 00:00", "2026-01-06 00:00",
                     None, None, 80, "active", "bullish trend", "structure"),
        make_concept("fvg", "bullish", "2026-01-05 00:00", "2026-01-06 00:00",
                     price + 2, price - 2, 80, "active", "bullish fvg", "smc"),
        make_concept("rsi_signal", "bearish", "2026-01-05 00:00", "2026-01-06 00:00",
                     None, None, 60, "active", "overbought", "classic"),
    ]
    result = confluence(concepts, _frame())
    explanation = build_explanation(concepts, result, _frame(), "1h")
    assert "0 of 0" not in explanation["narrative"]
    assert f"{len(result.factors)} factors" in explanation["narrative"]
    assert len(explanation["checklist"]) == len(result.factors)
    assert all("direction" in item for item in explanation["checklist"])


def test_hold_explanation_has_no_trade_levels():
    result = confluence([], _frame())
    explanation = build_explanation([], result, _frame(), "1h")
    assert result.action == "HOLD"
    plan = explanation["trade_plan"]
    assert plan["stop_loss"] is None
    assert plan["take_profit1"] is None
    assert plan["take_profit2"] is None
    assert plan["risk_reward"] == 0.0
    assert "No tradeable setup" in explanation["narrative"]
