from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.services import alerts, db


def _assert_persistence_round_trip(monkeypatch, database_url: str) -> None:
    db.close_db()
    monkeypatch.setattr(db, "engine", db.create_database_engine(database_url))
    monkeypatch.setattr(db, "_initialized", False)
    monkeypatch.setattr(alerts, "_schema_ready", False)
    db.init_db()

    marker = uuid4().hex
    signal = {
        "timeframe": "1h",
        "action": "BUY",
        "confidence": 80,
        "final_score": 0.7,
        "rule_score": 0.6,
        "ml_probability": None,
        "entry": 2000,
        "stop_loss": 1900,
        "take_profit1": 2100,
        "take_profit2": 2200,
        "risk_reward": 2,
        "atr": 12,
        "price": 2000,
        "explanation": {"summary": marker},
        "signal_time": marker,
    }

    try:
        signal_id = db.save_signal(signal)
        assert db.save_signal(signal) == signal_id
        assert [row["id"] for row in db.list_signals()] == [signal_id]

        db.set_app_setting(marker, "calendar")
        assert db.get_app_setting(marker) == "calendar"
        db.set_model_metadata(marker, {"rows": 12})
        assert db.get_model_metadata(marker) == {"rows": 12}

        alerts.save_settings(
            {"timeframes": ["1h"], "min_grade": "A", "channels": {}}, 17,
        )
        assert alerts.get_settings(17)["timeframes"] == ["1h"]
        db.save_journal_entry({"timeframe": "1h", "notes": marker}, 17)
        db.save_chat("user", marker, 17)
        assert db.list_journal(user_id=17)[0]["notes"] == marker
        assert db.list_chat(user_id=17)[0]["content"] == marker
    finally:
        db.close_db()


def test_persistence_round_trip_on_sqlite(monkeypatch, tmp_path) -> None:
    _assert_persistence_round_trip(
        monkeypatch, f"sqlite:///{tmp_path / 'portable.db'}",
    )


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="set TEST_POSTGRES_URL to run PostgreSQL integration coverage",
)
def test_persistence_round_trip_on_postgres(monkeypatch) -> None:
    _assert_persistence_round_trip(monkeypatch, os.environ["TEST_POSTGRES_URL"])
