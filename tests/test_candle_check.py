from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd

from app.api import market
from app.services import data
from app.services.providers import YFinanceProvider


def test_candle_check_returns_last_50_with_utc_and_local_times(monkeypatch):
    index = pd.date_range("2026-11-20", periods=55, freq="1h", tz="UTC")
    frame = pd.DataFrame({
        "Open": [2000.0 + i for i in range(55)],
        "High": [2001.0 + i for i in range(55)],
        "Low": [1999.0 + i for i in range(55)],
        "Close": [2000.5 + i for i in range(55)],
        "Volume": [100.0 + i for i in range(55)],
    }, index=index)
    monkeypatch.setattr(
        market.data,
        "get_ohlcv",
        lambda ticker, timeframe: data.CacheEntry(
            df=frame,
            fetched_at=0,
            source="mt5",
            meta={
                "broker": "FOREX.com",
                "server": "TEST",
                "symbol": "XAUUSD",
                "digits": 2,
                "timezone_offset_seconds": 7200,
                "approximate": False,
                "higher_timeframe_build": "MT5-provided",
            },
        ),
    )

    app = FastAPI()
    app.include_router(market.router)
    with TestClient(app) as client:
        response = client.get("/api/market/candle-check?timeframe=1h")

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["source"] == "mt5"
    assert payload["meta"]["price_side"] == "bid"
    assert payload["meta"]["status"] == "LIVE"
    assert len(payload["candles"]) == 50
    assert payload["candles"][-1]["utc"].endswith("+00:00")


def test_yfinance_fallback_never_uses_futures(monkeypatch):
    provider = YFinanceProvider()
    calls = []

    def fake_getter(ticker, timeframe):
        calls.append(ticker)
        raise RuntimeError("stop")

    monkeypatch.setattr(provider, "_download", fake_getter)
    monkeypatch.setattr(provider, "_chart_api", fake_getter)

    try:
        provider.fetch("XAUUSD=X", "1h")
    except RuntimeError:
        pass

    assert calls == ["XAUUSD=X", "XAUUSD=X"]
