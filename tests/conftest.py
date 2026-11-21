"""Pytest path setup: make `app` importable from backend/ root."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(autouse=True)
def disable_telegram(monkeypatch: pytest.MonkeyPatch):
    """Never start real Telegram polling/webhooks during the test suite."""
    from app import config
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(config, "TELEGRAM_WEBHOOK_SECRET", "")
    yield
