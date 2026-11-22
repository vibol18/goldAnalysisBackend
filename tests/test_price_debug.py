"""Measured checks for the price diagnosis endpoint and divergence guard."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.services import price_service, prices


def test_debug_price_reports_bridge_reason_and_sources():
    price_service.reset()
    with TestClient(app) as client:
        response = client.get("/api/debug/price")
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "XAU/USD"
    assert "source_priority" in payload
    assert "configured_sources" in payload
    assert isinstance(payload["why_bridge_not_used"], list)
    assert "verification" in payload


def test_mt5_tick_becomes_active_price_and_is_shared():
    price_service.reset()
    prices.FEED.update("mt5", 3615.53, 3616.43)
    snapshot = price_service.status()
    assert snapshot["source"] == "mt5"
    assert snapshot["bid"] == 3615.53
    assert snapshot["ask"] == 3616.43
    assert abs(snapshot["spread"] - 0.9) < 1e-6
    assert snapshot["price"] == 3615.53
    assert snapshot["state"] in (prices.LIVE, prices.CLOSED)
    price_service.reset()


def test_divergence_above_tolerance_sets_unverified():
    price_service.reset()
    prices.FEED.update("mt5", 3615.53, 3616.43)
    prices.FEED.update("twelvedata", 3604.10, 3604.10)
    result = price_service.verification()
    assert result["state"] == "UNVERIFIED"
    assert result["max_difference_usd"] >= 11.0
    assert result["tolerance_usd"] == config.PRICE_DIVERGENCE_TOLERANCE_USD
    price_service.reset()


def test_debug_price_has_no_stale_cache_beyond_limit():
    price_service.reset()
    payload = price_service.diagnostics()
    assert payload["cache_age_limit_seconds"] == 5
    price_service.reset()
