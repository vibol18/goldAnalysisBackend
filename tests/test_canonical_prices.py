"""Guard rails for the single canonical XAUUSD price service."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app import config
from app.services import price_service

BACKEND = Path(__file__).resolve().parents[1] / "app"
ALLOWED = {"services/data.py", "services/price_service.py", "services/providers.py"}


def _frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=50, freq="h", tz="UTC")
    return pd.DataFrame({
        "Open": np.linspace(2000, 2010, 50),
        "High": np.linspace(2001, 2011, 50),
        "Low": np.linspace(1999, 2009, 50),
        "Close": np.linspace(2000.5, 2010.5, 50),
        "Volume": np.full(50, 10.0),
    }, index=idx)


def test_hash_series_is_stable_and_sensitive():
    frame = _frame()
    first = price_service.hash_series(frame)
    assert first == price_service.hash_series(frame)
    frame.iloc[-1, frame.columns.get_loc("Close")] += 1.0
    assert first != price_service.hash_series(frame)


def test_canonical_entry_metadata(monkeypatch):
    from app.services import data
    entry = data.CacheEntry(df=_frame(), fetched_at=1.0, source="twelvedata")
    monkeypatch.setattr(data, "get_ohlcv", lambda *a, **kw: entry)
    price_service.reset()
    result = price_service.get_series("1h")
    assert result.meta["canonical"] is True
    assert result.meta["series_source"] == "twelvedata"
    assert len(result.meta["dataset_hash"]) == 40


def test_source_change_records_boundary(monkeypatch):
    from app.services import data
    entry = data.CacheEntry(df=_frame(), fetched_at=1.0, source="twelvedata")
    monkeypatch.setattr(data, "get_ohlcv", lambda *a, **kw: entry)
    price_service.reset()
    price_service.get_series("1h")
    entry.source = "mt5"
    price_service.get_series("1h")
    boundaries = price_service.series_meta("1h")["boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["from"] == "twelvedata"
    assert boundaries[0]["to"] == "mt5"


def test_no_module_fetches_gold_prices_directly():
    """Only the canonical service and provider internals may fetch gold prices."""
    violations: list[str] = []
    for path in BACKEND.rglob("*.py"):
        relative = path.relative_to(BACKEND).as_posix()
        if relative in ALLOWED or "tests" in relative:
            continue
        text = path.read_text(encoding="utf-8")
        if "get_ohlcv(config.GOLD_TICKER" in text or "get_ohlcv(tk," in text:
            violations.append(relative)
    assert violations == [], f"direct price fetches: {violations}"
