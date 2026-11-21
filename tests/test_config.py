from pathlib import Path

from app.config import normalize_database_url, parse_cors_origins


def test_postgres_url_is_normalized_and_render_ssl_is_required() -> None:
    url = normalize_database_url("postgres://user:pass@db.example.render.com/goldsight")

    assert url == (
        "postgresql://user:pass@db.example.render.com/goldsight?sslmode=require"
    )


def test_postgres_url_preserves_explicit_ssl_mode() -> None:
    url = normalize_database_url(
        "postgres://user:pass@db.example.render.com/goldsight?sslmode=verify-full"
    )

    assert url.endswith("?sslmode=verify-full")


def test_empty_database_url_uses_local_sqlite_file(tmp_path: Path) -> None:
    url = normalize_database_url("", tmp_path / "goldsight.db")

    assert url == "sqlite:///" + (tmp_path / "goldsight.db").resolve().as_posix()


def test_cors_origins_are_trimmed_and_empty_values_removed() -> None:
    assert parse_cors_origins(
        " https://frontend.example.com/ , , https://admin.example.com "
    ) == ("https://frontend.example.com", "https://admin.example.com")


def test_production_cors_defaults_to_no_origins() -> None:
    assert parse_cors_origins("", production=True) == ()
