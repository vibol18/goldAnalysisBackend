from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.services import data


def test_health_reports_cached_runtime_status(monkeypatch) -> None:
    monkeypatch.setattr(config, "ENV", "development")
    monkeypatch.setattr(config, "DATABASE_TYPE", "sqlite")
    monkeypatch.setattr(config, "APP_VERSION", "test-version")
    monkeypatch.setattr(
        data,
        "health_snapshot",
        lambda: {"status": "live", "last_updated": "2026-01-01T00:00:00+00:00"},
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
        "database": "sqlite",
        "data_source_status": "live",
        "last_data_update": "2026-01-01T00:00:00+00:00",
        "version": "test-version",
    }
