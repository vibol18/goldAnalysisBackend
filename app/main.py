"""GoldSight FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app import config
from app.api import alerts as alerts_api
from app.api import auth as auth_api
from app.api import chart, concepts, intelligence, market, misc, signals, stream
from app.services import auth, db, ratelimit, streaming
from app.services import alerts as alert_service

config.assert_production_secrets()


class SecretRedactingFilter(logging.Filter):
    """Strip API keys and tokens from log records (httpx logs full URLs)."""

    PATTERNS = (
        re.compile(r"(?i)(apikey=)[^&\s]+"),
        re.compile(r"(?i)(api[_-]?key=)[^&\s]+"),
        re.compile(r"(?i)(token=)[^&\s]+"),
        re.compile(r"/bot\d+:[A-Za-z0-9_\-]+"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in self.PATTERNS:
            message = pattern.sub(
                lambda m: (m.group(1) if m.lastindex else "") + "***REDACTED***", message)
        for name in ("TELEGRAM_BOT_TOKEN", "JWT_SECRET", "FIELD_ENCRYPTION_KEY",
                     "LLM_API_KEY", "TWELVE_DATA_API_KEY", "SMTP_PASSWORD"):
            value = os.getenv(name, "").strip()
            if len(value) >= 6:
                message = message.replace(value, "***REDACTED***")
        record.msg = message
        record.args = ()
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(SecretRedactingFilter())
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Trading gold is risky "
    "and signals can be wrong. Never risk money you can't afford to lose."
)

PUBLIC_PATHS = {
    "/api", "/api/health", "/api/auth/csrf", "/api/auth/signup",
    "/api/auth/login", "/api/auth/refresh", "/api/auth/forgot",
    "/api/auth/reset", "/api/auth/verify-email",
}
CSRF_EXEMPT = {
    "/api/auth/csrf", "/api/auth/signup", "/api/auth/login",
    "/api/auth/refresh", "/api/auth/forgot", "/api/auth/reset",
    "/api/auth/verify-email",
}
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
EXPENSIVE_PATHS = ("/api/backtest", "/api/intelligence/model", "/api/market/refresh")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Initialize storage, the alert scanner and the shared market stream."""
    db.init_db()
    await streaming.start_streaming()
    scanner = asyncio.create_task(alert_service.scanner_loop())
    try:
        yield
    finally:
        scanner.cancel()
        try:
            await scanner
        except asyncio.CancelledError:
            pass
        await streaming.stop_streaming()


app = FastAPI(
    title="GoldSight API",
    version="1.0.0",
    description=f"XAU/USD analysis, signals, backtesting and AI chat.\n\n{DISCLAIMER}",
    lifespan=lifespan,
    docs_url=None if config.PRODUCTION else "/docs",
    redoc_url=None if config.PRODUCTION else "/redoc",
    openapi_url=None if config.PRODUCTION else "/openapi.json",
)

if config.PRODUCTION:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts={u.split("//")[-1].split(":")[0] for u in config.ALLOWED_ORIGINS} | {"localhost"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-CSRF-Token", "X-Requested-With"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    ip = auth.client_ip(request)
    try:
        if path.startswith("/api/auth/login") or path.startswith("/api/auth/signup"):
            ratelimit.check(ratelimit.ip_key("auth", ip), 10)
        elif path.startswith("/api/auth/"):
            ratelimit.check(ratelimit.ip_key("auth-other", ip), 60)
        elif path.startswith("/api/chat"):
            ratelimit.check(ratelimit.ip_key("chat", ip),
                            int(config.CHAT_RATE_LIMIT_PER_MINUTE))
        elif any(path.startswith(p) for p in EXPENSIVE_PATHS):
            ratelimit.check(ratelimit.ip_key("expensive", ip), 5)
        else:
            ratelimit.check(ratelimit.ip_key("api", ip),
                            int(config.RATE_LIMIT_PER_MINUTE))
        if method in MUTATING and path.startswith("/api") and path not in CSRF_EXEMPT:
            auth.csrf_protect(request)
        if (config.PRODUCTION and path.startswith("/api")
                and path not in PUBLIC_PATHS):
            auth.current_user(request)
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; connect-src 'self' ws: wss:; font-src 'self' data:; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
    )
    if config.PRODUCTION:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


app.include_router(auth_api.router)
app.include_router(market.router)
app.include_router(signals.router)
app.include_router(misc.router)
app.include_router(concepts.router)
app.include_router(chart.router)
app.include_router(alerts_api.router)
app.include_router(stream.router)
app.include_router(intelligence.router)


@app.get("/api")
def root() -> dict[str, str]:
    return {"app": "GoldSight", "docs": "disabled" if config.PRODUCTION else "/docs",
            "disclaimer": DISCLAIMER}
