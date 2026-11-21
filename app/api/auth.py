"""Authentication endpoints: signup, login, sessions, 2FA, reset, admin."""
from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Any

import pyotp
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app import config
from app.services import auth, db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignupIn(StrictModel):
    email: EmailStr
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=12, max_length=256)

    @field_validator("password")
    @classmethod
    def strong(cls, value: str) -> str:
        auth.validate_password(value)
        return value


class LoginIn(StrictModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)
    totp: str | None = Field(default=None, max_length=12)


class RefreshIn(StrictModel):
    refresh_token: str | None = None


class PasswordIn(StrictModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=256)


class CodeIn(StrictModel):
    code: str = Field(min_length=6, max_length=8)


class TOTPIn(StrictModel):
    code: str = Field(min_length=6, max_length=8)
    password: str


class ForgotIn(StrictModel):
    email: EmailStr


class ResetIn(StrictModel):
    token: str
    new_password: str = Field(min_length=12, max_length=256)


class VerifyIn(StrictModel):
    token: str


def _set_cookies(response: Response, tokens: dict[str, str]) -> None:
    secure = config.PRODUCTION
    response.set_cookie(auth.ACCESS_COOKIE, tokens["access_token"], httponly=True,
                        secure=secure, samesite="lax", path="/",
                        max_age=int(config.ACCESS_TOKEN_MINUTES) * 60)
    response.set_cookie(auth.REFRESH_COOKIE, tokens["refresh_token"], httponly=True,
                        secure=secure, samesite="lax", path="/",
                        max_age=int(config.REFRESH_TOKEN_DAYS) * 86400)
    response.set_cookie(auth.CSRF_COOKIE, tokens["csrf_token"], httponly=False,
                        secure=secure, samesite="lax", path="/",
                        max_age=int(config.REFRESH_TOKEN_DAYS) * 86400)


def _clear_cookies(response: Response) -> None:
    for name in (auth.ACCESS_COOKIE, auth.REFRESH_COOKIE, auth.CSRF_COOKIE):
        response.delete_cookie(name, path="/")


@router.get("/csrf")
def csrf_cookie(response: Response) -> dict[str, str]:
    """Issue the double-submit CSRF cookie. Safe and unauthenticated."""
    token = secrets.token_urlsafe(32)
    response.set_cookie(auth.CSRF_COOKIE, token, httponly=False,
                        secure=config.PRODUCTION, samesite="lax", path="/")
    return {"status": "ok"}


@router.post("/signup", status_code=201)
def signup(body: SignupIn, request: Request, response: Response) -> dict[str, Any]:
    auth.check_lockout(body.email, auth.client_ip(request))
    user = auth.create_user(body.email, body.username, body.password)
    token = auth.create_email_token(user["id"], "verify")
    auth.audit(user["id"], "signup", "account created", auth.client_ip(request))
    out: dict[str, Any] = {"user": user, "status": "created"}
    if not config.PRODUCTION:
        out["dev_verification_token"] = token
    return out


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response) -> dict[str, Any]:
    ip = auth.client_ip(request)
    auth.check_lockout(str(body.email), ip)
    record = auth._get_auth_record(str(body.email))
    valid = bool(record and auth.verify_password(record["password_hash"], body.password))
    need_totp = bool(valid and (record["totp_secret"] or record["role"] == "admin"))
    if need_totp:
        if not body.totp or not record["totp_secret"] or not pyotp.TOTP(
                record["totp_secret"]).verify(body.totp, valid_window=1):
            valid = False
    if not valid or not record or not record["is_active"]:
        auth.record_login_attempt(str(body.email), ip, False)
        auth.audit(record["id"] if record else None, "login_failed",
                   "invalid credentials", ip)
        raise HTTPException(401, "Invalid email or password")
    auth.record_login_attempt(str(body.email), ip, True)
    tokens = auth.create_session(auth._row_to_user(record), ip,
                                 request.headers.get("user-agent", ""))
    _set_cookies(response, tokens)
    auth.audit(record["id"], "login", "session created", ip)
    return {"user": auth._row_to_user(record), "status": "ok"}


@router.post("/refresh")
def refresh(body: RefreshIn, request: Request, response: Response) -> dict[str, str]:
    token = body.refresh_token or request.cookies.get(auth.REFRESH_COOKIE)
    if not token:
        raise HTTPException(401, "No refresh token")
    tokens = auth.rotate_refresh(token, auth.client_ip(request),
                                 request.headers.get("user-agent", ""))
    _set_cookies(response, tokens)
    return {"status": "refreshed"}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    token = request.cookies.get(auth.ACCESS_COOKIE)
    if token:
        auth.revoke_session(token)
    _clear_cookies(response)
    return {"status": "logged_out"}


@router.post("/logout-all")
def logout_all(request: Request, response: Response,
               user: dict = Depends(auth.current_user)) -> dict[str, str]:
    auth.revoke_all_sessions(user["id"])
    auth.audit(user["id"], "logout_all", "all sessions revoked", auth.client_ip(request))
    _clear_cookies(response)
    return {"status": "logged_out"}


@router.get("/me")
def me(user: dict = Depends(auth.current_user)) -> dict[str, Any]:
    return {"user": user}


@router.post("/password")
def password_change(body: PasswordIn, request: Request, response: Response,
                    user: dict = Depends(auth.current_user)) -> dict[str, str]:
    with db._lock:
        row = db.get_conn().execute(
            "SELECT password_hash FROM users WHERE id=?", (user["id"],)
        ).fetchone()
    if not row or not auth.verify_password(row["password_hash"], body.current_password):
        raise HTTPException(401, "Invalid email or password")
    auth.change_password(user, body.new_password)
    _clear_cookies(response)
    return {"status": "password_changed"}


@router.post("/2fa/setup")
def totp_setup(user: dict = Depends(auth.current_user)) -> dict[str, str]:
    secret = pyotp.random_base32()
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE email_tokens SET used=1 WHERE user_id=? AND kind='totp_setup'",
                     (user["id"],))
        conn.execute(
            """INSERT INTO email_tokens (user_id, token_hash, kind, expires_at, created_at)
               VALUES (?,?, 'totp_setup', ?, ?)""",
            (user["id"], auth._hash_token(secret),
             auth._iso(auth._now() + timedelta(minutes=15)),
             auth._iso(auth._now())),
        )
        conn.commit()
    uri = pyotp.TOTP(secret).provisioning_uri(name=user["email"], issuer_name="GoldSight")
    return {"secret": secret, "uri": uri}


@router.post("/2fa/enable")
def totp_enable(body: CodeIn, request: Request,
                user: dict = Depends(auth.current_user)) -> dict[str, str]:
    with db._lock:
        conn = db.get_conn()
        row = conn.execute(
            """SELECT token_hash, id FROM email_tokens
               WHERE user_id=? AND kind='totp_setup' AND used=0
               ORDER BY id DESC LIMIT 1""", (user["id"],),
        ).fetchone()
    if row is None or not pyotp.TOTP(row["token_hash"]).verify(body.code, valid_window=1):
        raise HTTPException(400, "Invalid verification code")
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE users SET totp_secret=? WHERE id=?",
                     (row["token_hash"], user["id"]))
        conn.execute("UPDATE email_tokens SET used=1 WHERE id=?", (row["id"],))
        conn.commit()
    updated = auth.get_user_by_id(user["id"]) or {}
    auth.audit(user["id"], "2fa_enabled", "TOTP enabled", auth.client_ip(request))
    return {"status": "enabled", "user": updated}


@router.post("/2fa/disable")
def totp_disable(body: TOTPIn, request: Request,
                 user: dict = Depends(auth.current_user)) -> dict[str, str]:
    with db._lock:
        row = db.get_conn().execute(
            "SELECT totp_secret, password_hash FROM users WHERE id=?", (user["id"],)
        ).fetchone()
    if not row or not row["totp_secret"] or not pyotp.TOTP(
            row["totp_secret"]).verify(body.code, valid_window=1):
        raise HTTPException(400, "Invalid verification code")
    if not auth.verify_password(row["password_hash"], body.password):
        raise HTTPException(401, "Invalid email or password")
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE users SET totp_secret=NULL WHERE id=?", (user["id"],))
        conn.commit()
    auth.audit(user["id"], "2fa_disabled", "TOTP disabled", auth.client_ip(request))
    return {"status": "disabled"}


@router.post("/verify-email")
def verify_email(body: VerifyIn) -> dict[str, str]:
    user_id = auth.consume_email_token(body.token, "verify")
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE users SET email_verified=1 WHERE id=?", (user_id,))
        conn.commit()
    return {"status": "verified"}


@router.post("/forgot")
def forgot(body: ForgotIn, request: Request) -> dict[str, str]:
    user = auth.get_user_by_email(str(body.email))
    if not user:
        return {"status": "accepted"}
    token = auth.create_email_token(user["id"], "reset")
    auth.audit(user["id"], "password_reset_requested", "", auth.client_ip(request))
    out = {"status": "accepted"}
    if not config.PRODUCTION:
        out["dev_reset_token"] = token
    return out


@router.post("/reset")
def reset(body: ResetIn, request: Request) -> dict[str, str]:
    user_id = auth.consume_email_token(body.token, "reset")
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(400, "Invalid token")
    auth.change_password(user, body.new_password)
    auth.audit(user_id, "password_reset", "completed", auth.client_ip(request))
    return {"status": "reset"}


@router.get("/admin/users")
def admin_users(admin: dict = Depends(auth.admin_user)) -> dict[str, Any]:
    with db._lock:
        rows = db.get_conn().execute(
            "SELECT id, email, username, role, is_active, email_verified, created_at FROM users"
        ).fetchall()
    return {"users": [dict(r) for r in rows]}


@router.post("/admin/users/{user_id}/role")
def admin_set_role(user_id: int, role: str,
                   admin: dict = Depends(auth.admin_user)) -> dict[str, str]:
    if role not in {"user", "admin"}:
        raise HTTPException(400, "Invalid role")
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
        conn.commit()
    auth.audit(admin["id"], "role_changed", f"user {user_id} -> {role}")
    return {"status": "updated"}
