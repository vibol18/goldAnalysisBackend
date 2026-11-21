"""Application configuration loaded from environment (.env)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
GOLD_TICKER = os.getenv("GOLD_TICKER", "XAUUSD=X")
GOLD_FALLBACK_TICKER = os.getenv("GOLD_FALLBACK_TICKER", "GC=F")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_SYMBOL = os.getenv("TWELVE_DATA_SYMBOL", "XAU/USD")
TWELVE_DATA_BASE_URL = os.getenv("TWELVE_DATA_BASE_URL", "https://api.twelvedata.com")
KRAKEN_BASE_URL = os.getenv("KRAKEN_BASE_URL", "https://api.kraken.com")
KRAKEN_GOLD_PAIR = os.getenv("KRAKEN_GOLD_PAIR", "XAUTUSD")
GOLD_API_BASE_URL = os.getenv("GOLD_API_BASE_URL", "https://api.gold-api.com")
DXY_TICKER = os.getenv("DXY_TICKER", "DX-Y.NYB")
TNX_TICKER = os.getenv("TNX_TICKER", "^TNX")
SPX_TICKER = os.getenv("SPX_TICKER", "^GSPC")
EURUSD_TICKER = os.getenv("EURUSD_TICKER", "EURUSD=X")
USDJPY_TICKER = os.getenv("USDJPY_TICKER", "JPY=X")
SILVER_TICKER = os.getenv("SILVER_TICKER", "SI=F")
COPPER_TICKER = os.getenv("COPPER_TICKER", "HG=F")
OIL_TICKER = os.getenv("OIL_TICKER", "CL=F")
GLD_TICKER = os.getenv("GLD_TICKER", "GLD")
GDX_TICKER = os.getenv("GDX_TICKER", "GDX")
FRED_BASE_URL = os.getenv("FRED_BASE_URL", "https://fred.stlouisfed.org/graph/fredgraph.csv")
CFTC_GOLD_HISTORY_URL = os.getenv(
    "CFTC_GOLD_HISTORY_URL", "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
)
REAL_YIELD_SERIES = os.getenv("REAL_YIELD_SERIES", "DFII10")
BREAKEVEN_SERIES = os.getenv("BREAKEVEN_SERIES", "T10YIE")
MACRO_TTL_SECONDS = int(os.getenv("MACRO_TTL_SECONDS", "21600"))
COT_TTL_SECONDS = int(os.getenv("COT_TTL_SECONDS", "86400"))
NEWS_LLM_ENABLED = os.getenv("NEWS_LLM_ENABLED", "0").lower() in {"1", "true", "yes"}
NEWS_SENTIMENT_HALFLIFE_HOURS = float(os.getenv("NEWS_SENTIMENT_HALFLIFE_HOURS", "24"))
ROLL_CORR_WINDOWS = tuple(int(x) for x in os.getenv("ROLL_CORR_WINDOWS", "20,60,120").split(",") if x.strip().isdigit())
ECONOMIC_CALENDAR_URL = os.getenv("ECONOMIC_CALENDAR_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.xml")
CALENDAR_TTL_SECONDS = int(os.getenv("CALENDAR_TTL_SECONDS", "3600"))
CACHE_TTL_INTRADAY = int(os.getenv("CACHE_TTL_INTRADAY", "60"))
CACHE_TTL_DAILY = int(os.getenv("CACHE_TTL_DAILY", "1800"))

ML_HORIZON = int(os.getenv("ML_HORIZON", "6"))
SIGNAL_BUY_THRESHOLD = float(os.getenv("SIGNAL_BUY_THRESHOLD", "0.22"))
SIGNAL_SELL_THRESHOLD = float(os.getenv("SIGNAL_SELL_THRESHOLD", "-0.22"))
SIGNAL_COOLDOWN_MINUTES = float(os.getenv("SIGNAL_COOLDOWN_MINUTES", "45"))
ATR_STOP_MULT = float(os.getenv("ATR_STOP_MULT", "1.5"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "1.5"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "3.0"))

SPREAD_USD = float(os.getenv("SPREAD_USD", "0.30"))  # gold spread in $/oz
COMMISSION_USD = float(os.getenv("COMMISSION_USD", "0.00"))
BACKTEST_YEARS = int(os.getenv("BACKTEST_YEARS", "5"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.04"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "goldsight.db"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))

# --- Security -----------------------------------------------------------------
PRODUCTION = os.getenv("PRODUCTION", "0").strip().lower() in {"1", "true", "yes"}
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "goldsight").strip()
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "goldsight-app").strip()
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "7"))
FIELD_ENCRYPTION_KEY = os.getenv("FIELD_ENCRYPTION_KEY", "").strip()
ALLOWED_ORIGINS = tuple(
    origin.strip() for origin in
    os.getenv("ALLOWED_ORIGINS",
              "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
CHAT_RATE_LIMIT_PER_MINUTE = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "10"))
DEFAULT_SECRETS = {"", "CHANGE_ME_64_RANDOM_CHARS_MINIMUM",
                   "CHANGE_ME_ANOTHER_RANDOM_48_CHARS", "secret", "changeme"}


def assert_production_secrets() -> None:
    """Refuse to start in production with missing or default secrets."""
    if not PRODUCTION:
        return
    problems: list[str] = []
    if JWT_SECRET in DEFAULT_SECRETS or len(JWT_SECRET) < 32:
        problems.append("JWT_SECRET must be a random value of at least 32 characters")
    if FIELD_ENCRYPTION_KEY in DEFAULT_SECRETS or len(FIELD_ENCRYPTION_KEY) < 32:
        problems.append("FIELD_ENCRYPTION_KEY must be a random value of at least 32 characters")
    if "*" in ALLOWED_ORIGINS:
        problems.append("ALLOWED_ORIGINS must not contain '*'")
    if problems:
        raise RuntimeError("Production configuration is invalid: " + "; ".join(problems))


@lru_cache(maxsize=1)
def llm_configured() -> bool:
    """Return True when a usable provider + API key pair is present."""
    if LLM_PROVIDER not in {"anthropic", "openai"}:
        return False
    return bool(LLM_API_KEY)
