import pandas as pd

from app.services.tradingview_compare import (
    align_and_compare,
    detect_timeframe,
    parse_tradingview_csv,
)


def test_parse_and_align_tradingview_csv():
    csv = b"""time,open,high,low,close
2026-11-20T22:00:00Z,2000,2005,1998,2003
2026-11-20T22:15:00Z,2003,2007,2001,2006
"""
    reference = parse_tradingview_csv(csv)
    assert len(reference) == 2
    assert detect_timeframe(reference.index) == "15m"

    app = reference.copy()
    app.loc[:, "open"] += 0.10
    app.loc[:, "high"] += 0.10
    app.loc[:, "low"] += 0.10
    app.loc[:, "close"] += 0.10
    result = align_and_compare(reference, app)
    assert result["matched"] == 2
    assert abs(result["metrics"]["close"]["median"] - 0.10) < 0.001
    assert result["within_20cents"] == 100.0


def test_comparison_reports_no_exact_matches():
    reference = pd.DataFrame(
        {"open": [2000], "high": [2001], "low": [1999], "close": [2000]},
        index=pd.DatetimeIndex(["2026-11-20 22:00"], tz="UTC"),
    )
    app = reference.copy()
    app.index = app.index + pd.Timedelta(hours=1)
    result = align_and_compare(reference, app)
    assert result["matched"] == 0
    assert "No exact candle-open matches" in result["diagnosis"]["likely_cause"]
