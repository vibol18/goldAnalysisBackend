"""GoldSight FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.api import alerts as alerts_api
from app.api import chart, concepts, intelligence, market, misc, signals, stream
from app.services import db, streaming
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
            message = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "***REDACTED***", message)
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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"app": "GoldSight", "docs": "/docs", "disclaimer": DISCLAIMER}
