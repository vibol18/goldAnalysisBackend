from datetime import timedelta

import pandas as pd

from app.services.candle_aggregation import (
    aggregate_ohlcv,
    clean_ohlcv,
    daily_anchor_before,
)
from app.services.signal_engine import closed_frame


def base_frame(start: str, periods: int, freq: str = "1min") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    closes = [2000.0 + i * 0.1 for i in range(periods)]
    return pd.DataFrame({
        "Open": [v - 0.2 for v in closes],
        "High": [v + 0.5 for v in closes],
        "Low": [v - 0.5 for v in closes],
        "Close": closes,
        "Volume": [10.0] * periods,
    }, index=index)


def test_m1_to_15m_and_1h_uses_clean_ohlc():
    frame = base_frame("2026-01-05 22:00", 180)
    bars = aggregate_ohlcv(frame, "15m")
    assert len(bars) == 12
    assert bars.iloc[0]["Open"] == round(frame.iloc[0]["Open"], 2)
    assert bars.iloc[0]["Close"] == round(frame.iloc[14]["Close"], 2)

    hourly = aggregate_ohlcv(frame, "1h")
    assert len(hourly) == 3
    assert hourly.iloc[0]["High"] == frame.iloc[:60]["High"].max()


def test_ny_close_anchor_changes_by_dst():
    winter = daily_anchor_before(pd.Timestamp("2026-01-06 21:30", tz="UTC"))
    summer = daily_anchor_before(pd.Timestamp("2026-07-06 20:30", tz="UTC"))
    assert winter.isoformat() == "2026-01-05T22:00:00+00:00"
    assert summer.isoformat() == "2026-07-05T21:00:00+00:00"


def test_daily_and_4h_aggregation_has_no_flat_weekend_fill():
    frame = base_frame("2026-11-20 22:00", 2880)
    daily = aggregate_ohlcv(frame, "1D")
    four_hour = aggregate_ohlcv(frame, "4h")
    assert len(daily) < 3
    assert len(four_hour) <= 12
    assert (four_hour["High"] >= four_hour["Low"]).all()
    assert not daily.duplicated().any()
    assert not four_hour.duplicated().any()


def test_duplicate_and_negative_values_are_removed():
    frame = base_frame("2026-01-05 22:00", 5)
    duplicate = pd.concat([frame, frame.iloc[[2]]])
    duplicate.loc[duplicate.index[-1], "Close"] = -1
    cleaned = clean_ohlcv(duplicate)
    assert len(cleaned) == 4
    assert (cleaned[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_reverting_price_spike_is_removed():
    frame = base_frame("2026-01-05 22:00", 30)
    frame.iloc[15, frame.columns.get_loc("Close")] = 2200.0
    frame.iloc[15, frame.columns.get_loc("High")] = 2201.0
    cleaned = clean_ohlcv(frame)
    assert len(cleaned) == 29


def test_closed_frame_never_changes_last_closed_candle():
    frame = base_frame("2026-01-05 22:00", 20)
    closed = closed_frame(frame)
    assert len(closed) == 19
    last_closed = closed.iloc[-1].copy()
    frame.iloc[-1] = frame.iloc[-1] + 100
    assert closed.iloc[-1].equals(last_closed)
