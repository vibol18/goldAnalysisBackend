"""Tests for the causal price-action concept detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.concepts import detect_all
from app.concepts.base import DETECTOR_CATEGORIES, REQUIRED_FIELDS as FIELDS
import app.concepts.price_action as pa  # noqa: F401 - registers the family

PRICE_ACTION_TYPES = tuple(
    name for name, cat in DETECTOR_CATEGORIES.items() if cat == "price_action"
)


def frame(candles: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(candles), freq="h")
    return pd.DataFrame(
        candles, columns=["Open", "High", "Low", "Close"], index=idx
    ).assign(Volume=100.0)


def detect(type_: str, candles, zones=None):
    context = {"zones": zones} if zones is not None else {}
    return detect_all(frame(candles), context=context, types=(type_,))


def zone(top, bottom, kind="level"):
    return {"top": top, "bottom": bottom, "kind": kind}


def test_all_detectors_registered():
    expected = {
        "engulfing", "pin_bar", "doji", "morning_star", "evening_star",
        "inside_bar", "outside_bar", "three_soldiers", "three_crows",
        "double_top", "double_bottom", "head_shoulders",
        "inverse_head_shoulders", "triangle", "wedge", "flag",
        "range_breakout",
    }
    assert expected.issubset(set(PRICE_ACTION_TYPES))
    for name in PRICE_ACTION_TYPES:
        assert DETECTOR_CATEGORIES[name] == "price_action"


def test_bullish_engulfing():
    candles = [(2000 + 2 * t, 2001 + 2 * t, 1999 + 2 * t, 2001 + 2 * t)
               for t in range(20)]
    candles += [(2045, 2046, 2037, 2038), (2035, 2062, 2034, 2060)]
    candles += [(2062 + 2 * t, 2064 + 2 * t, 2061 + 2 * t, 2063 + 2 * t)
                for t in range(5)]
    found = detect("engulfing", candles, [zone(2062, 2034, "support")])
    assert found
    c = found[-1]
    assert c.direction == "bullish"
    assert c.strength > 0
    assert c.meta["context"]["source"] == "zone"
    assert c.end_time == frame(candles).index[21]


def test_hammer():
    candles = [(2100 - 2 * t, 2101 - 2 * t, 2099 - 2 * t, 2099 - 2 * t)
               for t in range(20)]
    candles += [(2058, 2059, 2051, 2052), (2052, 2057, 2036, 2056)]
    candles += [(2056 + 2 * t, 2058 + t, 2055 + t, 2057 + t) for t in range(5)]
    found = detect("pin_bar", candles, [zone(2058, 2036, "support")])
    assert found
    c = found[-1]
    assert c.direction == "bullish"
    assert c.strength > 0
    assert c.meta["pattern"] == "hammer"


def test_shooting_star():
    candles = [(1980 + 2 * t, 1981 + 2 * t, 1979 + 2 * t, 1981 + 2 * t)
               for t in range(20)]
    candles += [(1998, 2005, 1997, 2004), (2004, 2020, 1999.5, 2000)]
    candles += [(2000 - t, 2001 - t, 1999 - t, 1999 - t) for t in range(5)]
    found = detect("pin_bar", candles, [zone(2020, 1999.5, "resistance")])
    assert found
    c = found[-1]
    assert c.direction == "bearish"
    assert c.strength > 0
    assert c.meta["pattern"] == "shooting star"


def test_doji():
    candles = [(2000 + t, 2001 + t, 1999 + t, 2000 + t) for t in range(20)]
    candles += [(2000, 2003, 1997, 2000.2)]
    found = detect("doji", candles, [zone(2003, 1997, "support")])
    assert found
    c = found[-1]
    assert c.direction == "neutral"
    assert c.strength > 0


def test_inside_bar():
    candles = [(2000 + t, 2001 + t, 1999 + t, 2000 + t) for t in range(20)]
    candles += [(2000, 2012, 1998, 2010), (2003, 2010, 2001, 2007)]
    found = detect("inside_bar", candles, [zone(2012, 1998, "support")])
    assert found
    c = found[-1]
    assert c.direction == "bullish"
    assert c.strength > 0


def test_morning_star():
    candles = [(2100 - t, 2101 - t, 2099 - t, 2100 - t) for t in range(20)]
    candles += [(2050, 2051, 2029, 2030), (2025, 2027, 2022, 2024),
                (2028, 2052, 2027, 2050)]
    candles += [(2050 + t, 2052 + t, 2049 + t, 2051 + t) for t in range(3)]
    found = detect("morning_star", candles, [zone(2052, 2022, "support")])
    assert found
    c = found[-1]
    assert c.direction == "bullish"
    assert c.strength > 0


def test_three_soldiers():
    candles = [(2000 + t, 2001 + t, 1999 + t, 2000 + t) for t in range(20)]
    candles += [(2000, 2011, 1999, 2010), (2009, 2021, 2008, 2020),
                (2019, 2033, 2018, 2032)]
    candles += [(2032 + t, 2034 + t, 2031 + t, 2033 + t) for t in range(3)]
    found = detect("three_soldiers", candles, [zone(2033, 1999, "support")])
    assert found
    c = found[-1]
    assert c.direction == "bullish"
    assert c.strength > 0


def test_engulfing_without_context_is_ignored():
    candles = [(2000 + 2 * t, 2001 + 2 * t, 1999 + 2 * t, 2001 + 2 * t)
               for t in range(20)]
    candles += [(2045, 2046, 2037, 2038), (2035, 2062, 2034, 2060)]
    candles += [(2062 + 2 * t, 2064 + 2 * t, 2061 + 2 * t, 2063 + 2 * t)
                for t in range(5)]
    assert detect("engulfing", candles) == []


def test_context_fallback_uses_swing_level():
    candles = [(2010 - 3 * t, 2011 - 3 * t, 2009 - 3 * t, 2008 - 3 * t)
               for t in range(6)]
    candles += [
        (1991, 1992, 1984, 1985), (1985, 1986, 1980, 1981),
        (1981, 1983, 1978, 1982), (1982, 1989, 1981, 1988),
        (1988, 1995, 1987, 1994), (1994, 1997, 1993, 1996),
        (1996, 1997, 1992, 1993), (1993, 1994, 1988, 1989),
        (1989, 1990, 1985, 1986), (1986, 1987, 1983, 1984),
        (1984, 1988, 1974, 1986),
    ]
    candles += [(1986 + t, 1989 + t, 1985 + t, 1988 + t) for t in range(4)]
    found = detect("pin_bar", candles)
    assert found
    c = found[-1]
    assert c.meta["pattern"] == "hammer"
    assert c.meta["context"]["source"] == "swing"
    assert c.strength > 0


DOUBLE_BOTTOM_CANDLES = [
    (2000, 2001, 1999, 2001), (2001, 2002, 2000, 2002),
    (2002, 2003, 2001, 2003), (2003, 2004, 1998, 1999),
    (1999, 2000, 1990, 1991), (1991, 1992, 1983, 1984),
    (1984, 1985, 1980, 1981), (1981, 1984, 1980, 1983),
    (1983, 1990, 1982, 1989), (1989, 1996, 1988, 1995),
    (1995, 2002, 1994, 2001), (2001, 2008, 2000, 2007),
    (2007, 2014, 2006, 2013), (2013, 2020, 2012, 2019),
    (2019, 2019, 2012, 2013), (2013, 2014, 2005, 2006),
    (2006, 2007, 1998, 1999), (1999, 2000, 1983, 1984),
    (1984, 1985, 1980, 1981), (1981, 1984, 1980, 1983),
    (1983, 1992, 1982, 1991), (1991, 2000, 1990, 1999),
    (1999, 2008, 1998, 2007), (2007, 2015, 2006, 2014),
    (2014, 2020, 2013, 2019), (2019, 2035, 2018, 2034),
    (2034, 2037, 2033, 2036), (2036, 2038, 2035, 2037),
]

HEAD_SHOULDERS_CANDLES = [
    (1995, 1996, 1994, 1996), (1996, 1997, 1995, 1997),
    (1997, 1998, 1996, 1998), (1998, 2002, 1997, 2001),
    (2001, 2006, 2000, 2005), (2005, 2010, 2004, 2009),
    (2009, 2011, 2008, 2010), (2010, 2010, 2004, 2005),
    (2005, 2006, 1998, 1999), (1999, 2000, 1995, 1996),
    (1996, 1997, 1993, 1994), (1994, 1998, 1993, 1997),
    (1997, 2003, 1996, 2002), (2002, 2009, 2001, 2008),
    (2008, 2018, 2007, 2017), (2017, 2030, 2016, 2029),
    (2029, 2029, 2018, 2019), (2019, 2020, 2008, 2009),
    (2009, 2010, 1998, 1999), (1999, 2000, 1993, 1994),
    (1994, 1998, 1993, 1997), (1997, 2003, 1996, 2002),
    (2002, 2009, 2001, 2008), (2008, 2011, 2007, 2010),
    (2010, 2010, 2002, 2003), (2003, 2004, 1997, 1998),
    (1998, 1999, 1992, 1997), (1993, 1994, 1990, 1994),
    (1989, 1992, 1985, 1986), (1986, 1988, 1984, 1985),
]


def test_double_bottom_measured_target():
    found = detect("double_bottom", DOUBLE_BOTTOM_CANDLES)
    assert found
    c = found[-1]
    assert c.direction == "bullish"
    assert c.meta["target"] == pytest.approx(2060.0)
    assert "2060" in c.notes
    assert c.bottom == pytest.approx(1980.0)


def test_head_shoulders_measured_target():
    found = detect("head_shoulders", HEAD_SHOULDERS_CANDLES)
    assert found
    c = found[-1]
    assert c.direction == "bearish"
    assert c.meta["target"] == pytest.approx(1956.0)
    assert "1956" in c.notes
    assert c.top == pytest.approx(2030.0)


def test_causal_under_truncation():
    df = frame(DOUBLE_BOTTOM_CANDLES)
    k = 26
    cutoff = df.index[k - 1]

    def geometry(concepts):
        return {
            (c.type, str(c.start_time), str(c.end_time)):
                (c.direction, c.top, c.bottom, round(c.strength, 6))
            for c in concepts if pd.Timestamp(c.end_time) <= cutoff
        }

    full = detect_all(df, types=PRICE_ACTION_TYPES)
    truncated = detect_all(df.iloc[:k], types=PRICE_ACTION_TYPES)
    assert geometry(full) == geometry(truncated)
    assert any(c.type == "double_bottom" for c in truncated)


def test_required_fields_and_limits():
    df = frame(DOUBLE_BOTTOM_CANDLES + HEAD_SHOULDERS_CANDLES)
    results = detect_all(df, types=PRICE_ACTION_TYPES)
    assert results
    counts: dict[str, int] = {}
    for concept in results:
        data = concept.to_dict()
        for field in FIELDS:
            assert field in data
        assert data["category"] == "price_action"
        assert data["direction"] in {"bullish", "bearish", "neutral"}
        assert data["status"] in {"active", "mitigated", "invalidated"}
        assert 0 <= data["strength"] <= 100
        counts[data["type"]] = counts.get(data["type"], 0) + 1
    assert all(count <= 12 for count in counts.values())


def test_short_and_nan_frames_do_not_raise():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    assert detect_all(empty, types=PRICE_ACTION_TYPES) == []
    short = frame([(1.0, 1.1, 0.9, 1.0)] * 3)
    assert isinstance(detect_all(short, types=PRICE_ACTION_TYPES), list)
    with_nan = frame([(2000 + t, 2002 + t, 1998 + t, 2001 + t)
                      for t in range(40)])
    with_nan.loc[with_nan.index[10], "High"] = np.nan
    assert isinstance(detect_all(with_nan, types=PRICE_ACTION_TYPES), list)
