"""Authentication, sessions, tokens, RBAC and abuse controls."""
from __future__ import annotations

import hashlib
import logging
import re
import secrets
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import HTTPException, Request
from sqlalchemy.exc import IntegrityError

from app import config
from app.services import db

logger = logging.getLogger(__name__)

ACCESS_COOKIE = "gs_access"
REFRESH_COOKIE = "gs_refresh"
CSRF_COOKIE = "gs_csrf"
ALGORITHM = "HS256"
SECRET = config.JWT_SECRET or secrets.token_urlsafe(48)
MIN_PASSWORD_LENGTH = 12
COMMON_PASSWORDS = {
    "password", "password123", "123456789", "qwerty123", "letmein123",
    "welcome123", "admin12345", "gold123456", "trading123", "iloveyou123",
    "abc1234567", "111111111111", "000000000000", "qwertyuiop", "passw0rd123",
}
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2,
                         hash_len=32, salt_len=16)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored: str, password: str) -> bool:
    try:
        return _hasher.verify(stored, password)
    except (VerifyMismatchError, InvalidHashError, ValueError):
        return False


def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if password.lower() in COMMON_PASSWORDS:
        raise ValueError("Password is too common")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise ValueError("Password must contain letters and numbers")


def validate_email(email: str) -> str:
    clean = email.strip().lower()
    if not EMAIL_RE.match(clean) or len(clean) > 254:
        raise ValueError("Invalid email address")
    return clean


def _row_to_user(row: Any) -> dict[str, Any]:
    user = dict(row)
    user.pop("password_hash", None)
    user["totp_enabled"] = bool(user.pop("totp_secret", None))
    user["is_active"] = bool(user.get("is_active"))
    user["email_verified"] = bool(user.get("email_verified"))
    return user


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with db._lock:
        row = db.get_conn().execute(
            "SELECT * FROM users WHERE email=?", (validate_email(email),)
        ).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with db._lock:
        row = db.get_conn().execute(
            "SELECT * FROM users WHERE id=?", (user_id,)
        ).fetchone()
    return _row_to_user(row) if row else None


def _get_auth_record(email: str) -> Mapping[str, Any] | None:
    with db._lock:
        return db.get_conn().execute(
            "SELECT * FROM users WHERE email=?", (validate_email(email),)
        ).fetchone()


def create_user(email: str, username: str, password: str,
                role: str = "user") -> dict[str, Any]:
    clean_email = validate_email(email)
    validate_password(password)
    if role not in {"user", "admin"}:
        raise ValueError("Invalid role")
    if not username or len(username.strip()) > 64:
        raise ValueError("Invalid username")
    now = _iso(_now())
    with db._lock:
        conn = db.get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO users
                   (email, username, password_hash, role, is_active, email_verified,
                    failed_attempts, password_changed_at, created_at)
                   VALUES (?,?,?,?,1,0,0,?,?)""",
                (clean_email, username.strip(), hash_password(password),
                 role, now, now),
            )
        except IntegrityError as exc:
            raise ValueError("Could not create account") from exc
        conn.commit()
    return get_user_by_id(int(cur.lastrowid)) or {}


def record_login_attempt(email: str, ip: str, succeeded: bool) -> None:
    with db._lock:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO login_attempts (email, ip, succeeded, created_at) VALUES (?,?,?,?)",
            (email, ip, int(succeeded), _iso(_now())),
        )
        if succeeded:
            conn.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE email=?",
                         (email,))
        else:
            conn.execute(
                """UPDATE users SET failed_attempts=failed_attempts+1,
                   locked_until=CASE WHEN failed_attempts+1 >= 5
                   THEN ? ELSE locked_until END WHERE email=?""",
                (_iso(_now() + timedelta(minutes=15)), email),
            )
        conn.commit()


def check_lockout(email: str, ip: str) -> None:
    with db._lock:
        row = db.get_conn().execute(
            """SELECT COUNT(*) c FROM login_attempts
               WHERE ip=? AND created_at > ?""",
            (ip, _iso(_now() - timedelta(minutes=10))),
        ).fetchone()
    if row and int(row["c"]) >= 30:
        raise HTTPException(429, "Too many attempts. Try again later.")
    record = _get_auth_record(email)
    if record and record["locked_until"]:
        try:
            locked = datetime.fromisoformat(record["locked_until"])
            if locked > _now():
                raise HTTPException(423, "Account temporarily locked after repeated failures")
        except (ValueError, TypeError):
            pass


def audit(user_id: int | None, action: str, detail: str, ip: str = "") -> None:
    with db._lock:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO audit_log (at, user_id, action, detail, ip) VALUES (?,?,?,?,?)",
            (_iso(_now()), user_id, action, detail[:500], ip[:64]),
        )
        conn.commit()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user: dict[str, Any], ip: str = "", ua: str = "") -> dict[str, str]:
    now = _now()
    family = secrets.token_urlsafe(16)
    with db._lock:
        conn = db.get_conn()
        cur = conn.execute(
            """INSERT INTO sessions
               (user_id, token_hash, refresh_hash, family_id, created_at, expires_at,
                revoked, ip, user_agent)
               VALUES (?,?,?,?,?,?,0,?,?)""",
            (user["id"], secrets.token_hex(32), secrets.token_hex(32), family, _iso(now),
             _iso(now + timedelta(days=int(config.REFRESH_TOKEN_DAYS))), ip[:64], ua[:200]),
        )
        conn.commit()
    session_id = int(cur.lastrowid)
    access = _make_access_token(user, session_id)
    refresh = f"{family}.{secrets.token_urlsafe(48)}"
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE sessions SET token_hash=?, refresh_hash=? WHERE id=?",
                     (_hash_token(access), _hash_token(refresh), session_id))
        conn.commit()
    return {
        "access_token": access,
        "refresh_token": refresh,
        "csrf_token": secrets.token_urlsafe(32),
        "session_id": str(session_id),
    }


def _make_access_token(user: dict[str, Any], session_id: int) -> str:
    now = _now()
    payload = {
        "sub": str(user["id"]), "role": user["role"], "sid": str(session_id),
        "iat": int(now.timestamp()), "nbf": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=int(config.ACCESS_TOKEN_MINUTES))).timestamp()),
        "iss": config.JWT_ISSUER, "aud": config.JWT_AUDIENCE,
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def decode_access(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM],
                          issuer=config.JWT_ISSUER, audience=config.JWT_AUDIENCE,
                          options={"require": ["exp", "iss", "aud", "sub"]})
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "Invalid or expired session") from exc


def rotate_refresh(refresh: str, ip: str = "", ua: str = "") -> dict[str, str]:
    family = refresh.split(".", 1)[0]
    with db._lock:
        conn = db.get_conn()
        row = conn.execute("SELECT * FROM sessions WHERE refresh_hash=?",
                           (_hash_token(refresh),)).fetchone()
        if row is None:
            conn.execute("UPDATE sessions SET revoked=1 WHERE family_id=?", (family,))
            conn.commit()
            raise HTTPException(401, "Invalid refresh token")
        if row["revoked"] or datetime.fromisoformat(row["expires_at"]) <= _now():
            raise HTTPException(401, "Invalid refresh token")
        user = get_user_by_id(row["user_id"])
        if not user or not user["is_active"]:
            raise HTTPException(401, "Invalid refresh token")
        new_refresh = f"{family}.{secrets.token_urlsafe(48)}"
        conn.execute("UPDATE sessions SET refresh_hash=? WHERE id=?",
                     (_hash_token(new_refresh), row["id"]))
        conn.commit()
    access = _make_access_token(user, row["id"])
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE sessions SET token_hash=? WHERE id=?",
                     (_hash_token(access), row["id"]))
        conn.commit()
    return {"access_token": access, "refresh_token": new_refresh,
            "csrf_token": secrets.token_urlsafe(32)}


def revoke_session(access_token: str) -> None:
    try:
        payload = decode_access(access_token)
    except HTTPException:
        return
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE sessions SET revoked=1 WHERE id=?", (payload.get("sid"),))
        conn.commit()


def revoke_all_sessions(user_id: int) -> None:
    with db._lock:
        conn = db.get_conn()
        conn.execute("UPDATE sessions SET revoked=1 WHERE user_id=?", (user_id,))
        conn.commit()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded and config.PRODUCTION:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


def current_user(request: Request) -> dict[str, Any]:
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        raise HTTPException(401, "Authentication required")
    payload = decode_access(token)
    with db._lock:
        session = db.get_conn().execute(
            "SELECT revoked, expires_at FROM sessions WHERE id=?",
            (payload.get("sid"),),
        ).fetchone()
    if session is None or session["revoked"]:
        raise HTTPException(401, "Session revoked")
    user = get_user_by_id(int(payload["sub"]))
    if not user or not user["is_active"]:
        raise HTTPException(401, "Account inactive")
    return user


def optional_user(request: Request) -> dict[str, Any] | None:
    """Allow anonymous access only in development; production requires a session."""
    if not request.cookies.get(ACCESS_COOKIE):
        if config.PRODUCTION:
            raise HTTPException(401, "Authentication required")
        return None
    return current_user(request)


def admin_user(request: Request) -> dict[str, Any]:
    user = current_user(request)
    if user["role"] != "admin":
        raise HTTPException(403, "Administrator access required")
    if not user.get("totp_enabled"):
        raise HTTPException(403, "Administrators must enable two-factor authentication")
    return user


def csrf_protect(request: Request) -> None:
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get("x-csrf-token")
    if not cookie or not header or not secrets.compare_digest(cookie, header):
        raise HTTPException(403, "CSRF token missing or invalid")


def create_email_token(user_id: int, kind: str) -> str:
    token = secrets.token_urlsafe(32)
    with db._lock:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO email_tokens (user_id, token_hash, kind, expires_at, used, created_at) VALUES (?,?,?,?,0,?)",
            (user_id, _hash_token(token), kind,
             _iso(_now() + timedelta(hours=24)), _iso(_now())),
        )
        conn.commit()
    return token


def consume_email_token(token: str, kind: str) -> int:
    with db._lock:
        conn = db.get_conn()
        row = conn.execute(
            "SELECT * FROM email_tokens WHERE token_hash=? AND kind=? AND used=0",
            (_hash_token(token), kind),
        ).fetchone()
        if row is None or datetime.fromisoformat(row["expires_at"]) <= _now():
            raise HTTPException(400, "Invalid or expired token")
        conn.execute("UPDATE email_tokens SET used=1 WHERE id=?", (row["id"],))
        conn.commit()
    return int(row["user_id"])


def change_password(user: dict[str, Any], new_password: str) -> None:
    validate_password(new_password)
    with db._lock:
        conn = db.get_conn()
        conn.execute(
            "UPDATE users SET password_hash=?, password_changed_at=? WHERE id=?",
            (hash_password(new_password), _iso(_now()), user["id"]),
        )
        conn.commit()
    revoke_all_sessions(user["id"])
    audit(user["id"], "password_changed", "sessions revoked", "")
