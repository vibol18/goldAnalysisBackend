"""Authentication, session, RBAC and IDOR tests."""
from __future__ import annotations

import pyotp
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config
from app.api import auth as auth_api
from app.api import misc
from app.services import auth, db

STRONG = "Str0ngPassw0rd!2026"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'auth.db'}"
    monkeypatch.setattr(config, "DATABASE_URL", url)
    monkeypatch.setattr(db, "engine", db.create_database_engine(url))
    monkeypatch.setattr(db, "_initialized", False)
    monkeypatch.setattr(config, "PRODUCTION", False)
    db.init_db()
    app = FastAPI()
    app.include_router(auth_api.router)
    app.include_router(misc.router)
    with TestClient(app) as c:
        yield c


def signup(client, email="user@example.com", password=STRONG, username="user"):
    return client.post("/api/auth/signup", json={
        "email": email, "username": username, "password": password})


def test_password_policy_and_hashing():
    auth.validate_password(STRONG)
    with pytest.raises(ValueError):
        auth.validate_password("short1")
    with pytest.raises(ValueError):
        auth.validate_password("password123")
    hashed = auth.hash_password(STRONG)
    assert hashed.startswith("$argon2id$")
    assert auth.verify_password(hashed, STRONG)
    assert not auth.verify_password(hashed, "wrong")


def test_signup_login_me_and_cookies(client):
    assert signup(client).status_code == 201
    response = client.post("/api/auth/login", json={
        "email": "user@example.com", "password": STRONG})
    assert response.status_code == 200, response.text
    assert client.cookies.get("gs_access")
    assert client.cookies.get("gs_refresh")
    assert client.cookies.get("gs_csrf")
    assert client.get("/api/auth/me").json()["user"]["email"] == "user@example.com"


def test_generic_login_error_does_not_reveal_account(client):
    signup(client)
    unknown = client.post("/api/auth/login", json={
        "email": "nobody@example.com", "password": "whatever"})
    wrong = client.post("/api/auth/login", json={
        "email": "user@example.com", "password": "whatever"})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json() == {"detail": "Invalid email or password"}


def test_refresh_rotates_and_reuse_revokes_family(client):
    signup(client)
    client.post("/api/auth/login", json={"email": "user@example.com", "password": STRONG})
    old = client.cookies.get("gs_refresh")
    assert old
    rotated = client.post("/api/auth/refresh", json={"refresh_token": old})
    assert rotated.status_code == 200, rotated.text
    new = client.cookies.get("gs_refresh")
    assert new != old
    reuse = client.post("/api/auth/refresh", json={"refresh_token": old})
    assert reuse.status_code == 401


def test_logout_and_logout_all_revoke_sessions(client):
    signup(client)
    client.post("/api/auth/login", json={"email": "user@example.com", "password": STRONG})
    out = client.post("/api/auth/logout")
    assert out.status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_lockout_after_repeated_failures(client):
    signup(client)
    for _ in range(5):
        client.post("/api/auth/login", json={
            "email": "user@example.com", "password": "wrong-password"})
    assert client.post("/api/auth/login", json={
        "email": "user@example.com", "password": STRONG}).status_code == 423


def test_2fa_setup_enable_and_required_at_login(client):
    signup(client)
    client.post("/api/auth/login", json={"email": "user@example.com", "password": STRONG})
    secret = client.post("/api/auth/2fa/setup").json()["secret"]
    code = pyotp.TOTP(secret).now()
    assert client.post("/api/auth/2fa/enable", json={"code": code}).status_code == 200
    client.post("/api/auth/logout")
    missing = client.post("/api/auth/login", json={
        "email": "user@example.com", "password": STRONG})
    assert missing.status_code == 401
    ok = client.post("/api/auth/login", json={
        "email": "user@example.com", "password": STRONG,
        "totp": pyotp.TOTP(secret).now()})
    assert ok.status_code == 200, ok.text


def test_password_change_revokes_all_sessions(client):
    signup(client)
    client.post("/api/auth/login", json={"email": "user@example.com", "password": STRONG})
    changed = client.post("/api/auth/password", json={
        "current_password": STRONG, "new_password": "An0therStrong!2026"})
    assert changed.status_code == 200, changed.text
    assert client.get("/api/auth/me").status_code == 401


def test_idor_blocked_between_users(client):
    signup(client, "a@example.com", username="usera")
    client.post("/api/auth/login", json={"email": "a@example.com", "password": STRONG})
    own = client.post("/api/journal", json={"action": "BUY", "result_pct": 1.0})
    assert own.status_code == 200
    journal_id = own.json()["id"]
    client.post("/api/auth/logout")

    signup(client, "b@example.com", username="userb")
    client.post("/api/auth/login", json={"email": "b@example.com", "password": STRONG})
    assert client.get("/api/journal").json()["trades"] == []
    assert client.delete(f"/api/journal/{journal_id}").json()["deleted"] is False


def test_admin_requires_totp(client):
    signup(client, "admin@example.com", username="admin")
    with db._lock:
        db.get_conn().execute("UPDATE users SET role='admin' WHERE email='admin@example.com'")
        db.get_conn().commit()
    login = client.post("/api/auth/login", json={
        "email": "admin@example.com", "password": STRONG})
    assert login.status_code == 401
    assert client.get("/api/auth/admin/users").status_code == 401


def test_csrf_required_for_state_change():
    from app.main import app
    with TestClient(app) as c:
        assert c.post("/api/journal", json={"action": "BUY"}).status_code == 403
        c.get("/api/auth/csrf")
        assert c.post("/api/journal", json={"action": "BUY"},
                      headers={"X-CSRF-Token": c.cookies.get("gs_csrf")}).status_code == 200
