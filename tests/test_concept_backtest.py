"""Tests for concept-level and full confluence backtesting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.concept_backtest import full_confluence_backtest, per_concept_performance


def make_df(n: int = 600, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.008, n)
    close = 2000 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": open_, "High": np.maximum(open_, close) * 1.005,
        "Low": np.minimum(open_, close) * 0.995, "Close": close,
        "Volume": rng.integers(1000, 50000, n).astype(float),
    }, index=idx)


def test_per_concept_performance_reports_all_directional_types():
    rows = per_concept_performance(make_df())
    assert rows
    names = {r["type"] for r in rows}
    assert len(names) >= 8
    assert {"fvg", "trend", "bos"} <= names
    for row in rows:
        assert row["trades"] > 0
        assert 0.0 <= row["win_rate"] <= 100.0
        assert row["weight"] >= 0
        assert row["category"] in {"structure", "smc", "classic",
                                   "price_action", "mtf", "macro", "advanced"}


def test_full_backtest_returns_metrics_and_uses_costs():
    result = full_confluence_backtest(make_df(), years=None)
    for key in ("trades", "win_rate", "profit_factor", "expectancy",
                "max_drawdown_pct", "sharpe", "total_return_pct", "final_equity"):
        assert key in result
    assert result["assumptions"]["spread_usd_per_oz"] > 0
    assert result["concepts_used"] > 50
    assert all("entry_time" in t and "pnl" in t for t in result["trade_list"])


def test_backtest_is_causal_when_future_changes():
    df = make_df()
    base = full_confluence_backtest(df, years=None)
    changed = df.copy()
    changed.iloc[-40:, changed.columns.get_loc("Close")] *= 1.25
    changed.iloc[-40:, changed.columns.get_loc("High")] *= 1.25
    changed.iloc[-40:, changed.columns.get_loc("Low")] *= 1.25
    later = full_confluence_backtest(changed, years=None)
    cutoff = str(df.index[-60])
    early_base = [t for t in base["trade_list"] if t["entry_time"] < cutoff]
    early_later = [t for t in later["trade_list"] if t["entry_time"] < cutoff]
    assert early_base == early_later
