"""Application configuration loaded from environment (.env)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


def _environment() -> str:
    value = os.getenv("ENV")
    if value is None:
        value = "production" if os.getenv("PRODUCTION", "0").strip().lower() in {
            "1", "true", "yes",
        } else "development"
    value = value.strip().lower()
    if value not in {"development", "production"}:
        raise ValueError("ENV must be either 'development' or 'production'")
    return value


ENV = _environment()
PRODUCTION = ENV == "production"
PORT = int(os.getenv("PORT", "8000"))
APP_VERSION = os.getenv("APP_VERSION", "1.0.0").strip() or "1.0.0"
TRUSTED_HOSTS = tuple(
    host.strip() for host in os.getenv(
        "TRUSTED_HOSTS",
        ",".join(filter(None, (os.getenv("RENDER_EXTERNAL_HOSTNAME", ""), "*.onrender.com", "localhost"))),
    ).split(",") if host.strip()
)
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "none" if PRODUCTION else "lax").strip().lower()
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "1" if PRODUCTION else "0").strip().lower() in {
    "1", "true", "yes",
}
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
if not PUBLIC_BASE_URL and PRODUCTION:
    _render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    PUBLIC_BASE_URL = f"https://{_render_host}" if _render_host else ""
ADMIN_CHAT_IDS = tuple(
    int(value.strip()) for value in os.getenv("ADMIN_CHAT_IDS", "").split(",")
    if value.strip().lstrip("-").isdigit()
)
TELEGRAM_LINK_CODE_TTL_SECONDS = max(60, int(os.getenv("TELEGRAM_LINK_CODE_TTL_SECONDS", "600")))
TELEGRAM_DAILY_CAP = max(1, int(os.getenv("TELEGRAM_DAILY_CAP", "50")))
TELEGRAM_RATE_PER_SECOND = max(1, int(os.getenv("TELEGRAM_RATE_PER_SECOND", "25")))
TELEGRAM_PER_CHAT_SECONDS = max(0.0, float(os.getenv("TELEGRAM_PER_CHAT_SECONDS", "1")))
TELEGRAM_CATCHUP_ENABLED = os.getenv("TELEGRAM_CATCHUP_ENABLED", "1").strip().lower() in {
    "1", "true", "yes",
}
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "").strip()
SMTP_TO = os.getenv("SMTP_TO", "").strip()
# --- live price feed ---------------------------------------------------------
DATA_SOURCE_PRIORITY = tuple(
    value.strip().lower() for value in
    os.getenv("DATA_SOURCE_PRIORITY", "mt5,oanda,twelve,rest").split(",")
    if value.strip()
)
PRICE_BID_MODE = os.getenv("PRICE_BID_MODE", "bid").strip().lower()
PRICE_FAILOVER_SECONDS = float(os.getenv("PRICE_FAILOVER_SECONDS", "10"))
PRICE_DIVERGENCE_TOLERANCE_USD = float(os.getenv("PRICE_DIVERGENCE_TOLERANCE_USD", "2.0"))
REQUIRE_BRIDGE_FOR_SIGNALS = os.getenv("REQUIRE_BRIDGE_FOR_SIGNALS", "0").strip().lower() in {"1", "true", "yes"}
PRICE_LIVE_SECONDS = float(os.getenv("PRICE_LIVE_SECONDS", "5"))
PRICE_DELAYED_SECONDS = float(os.getenv("PRICE_DELAYED_SECONDS", "30"))
MARKET_TIMEZONE = os.getenv("MARKET_TIMEZONE", "America/New_York").strip() or "America/New_York"
MARKET_CLOSE_HOUR = float(os.getenv("MARKET_CLOSE_HOUR", "17"))
MARKET_OPEN_HOUR = float(os.getenv("MARKET_OPEN_HOUR", "17"))
WATCHDOG_INTERVAL_SECONDS = float(os.getenv("WATCHDOG_INTERVAL_SECONDS", "5"))
WATCHDOG_ALERT_SECONDS = float(os.getenv("WATCHDOG_ALERT_SECONDS", "120"))
STREAM_TICK_THROTTLE_SECONDS = float(os.getenv("STREAM_TICK_THROTTLE_SECONDS", "0.5"))

STREAM_INTERVAL_SECONDS = float(os.getenv("STREAM_INTERVAL_SECONDS", "2"))
STREAM_FORCE_REFRESH_SECONDS = float(os.getenv("STREAM_FORCE_REFRESH_SECONDS", "10"))
STREAM_MACRO_INTERVAL_SECONDS = float(os.getenv("STREAM_MACRO_INTERVAL_SECONDS", "15"))
STREAM_NEWS_INTERVAL_SECONDS = float(os.getenv("STREAM_NEWS_INTERVAL_SECONDS", "120"))
STREAM_ANNOTATION_INTERVAL_SECONDS = float(os.getenv("STREAM_ANNOTATION_INTERVAL_SECONDS", "10"))
STREAM_QUEUE_MAXSIZE = int(os.getenv("STREAM_QUEUE_MAXSIZE", "100"))
DATA_CACHE_MAX_ENTRIES = int(os.getenv("DATA_CACHE_MAX_ENTRIES", "16"))
DATA_CACHE_MAX_ROWS = int(os.getenv("DATA_CACHE_MAX_ROWS", "50000"))
SIGNAL_CATCHUP_BARS_PER_PASS = max(1, int(os.getenv("SIGNAL_CATCHUP_BARS_PER_PASS", "1")))
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
CANDLE_DISPLAY_TIMEZONE = os.getenv("CANDLE_DISPLAY_TIMEZONE", "Asia/Phnom_Penh").strip() or "Asia/Phnom_Penh"
BROKER_SERVER_ANCHOR_OFFSET_SECONDS = int(
    os.getenv("BROKER_SERVER_ANCHOR_OFFSET_SECONDS", "0")
)
DAILY_SESSION_ANCHOR = os.getenv("DAILY_SESSION_ANCHOR", "17:00_NEW_YORK").strip()
OUTLIER_ATR_MULTIPLIER = float(os.getenv("OUTLIER_ATR_MULTIPLIER", "8"))
DATA_SOURCE_PRIORITY = tuple(
    value.strip().lower()
    for value in os.getenv("DATA_SOURCE_PRIORITY", "mt5,twelvedata,oanda,rest,yfinance").split(",")
    if value.strip()
)
CANDLE_PRICE_SIDE = os.getenv("CANDLE_PRICE_SIDE", "bid").strip().lower()
if CANDLE_PRICE_SIDE not in {"bid", "ask", "mid"}:
    CANDLE_PRICE_SIDE = "bid"
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN", "").strip()
OANDA_API_BASE_URL = os.getenv(
    "OANDA_API_BASE_URL", "https://api-fxpractice.oanda.com",
).strip().rstrip("/")
OANDA_GOLD_INSTRUMENT = os.getenv("OANDA_GOLD_INSTRUMENT", "XAU_USD").strip() or "XAU_USD"

# --- MT5 bridge --------------------------------------------------------------
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "").strip()
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSD").strip() or "XAUUSD"
MT5_BROKER_NAME = os.getenv("MT5_BROKER_NAME", "").strip()
BRIDGE_TICK_MAX_AGE_SECONDS = int(os.getenv("BRIDGE_TICK_MAX_AGE_SECONDS", "120"))
BRIDGE_CANDLE_MAX_AGE_SECONDS = int(os.getenv("BRIDGE_CANDLE_MAX_AGE_SECONDS", "600"))
BRIDGE_RATE_LIMIT_PER_MINUTE = int(os.getenv("BRIDGE_RATE_LIMIT_PER_MINUTE", "240"))
BRIDGE_MAX_BODY_BYTES = int(os.getenv("BRIDGE_MAX_BODY_BYTES", str(2 * 1024 * 1024)))
BRIDGE_MAX_CANDLES_PER_REQUEST = int(os.getenv("BRIDGE_MAX_CANDLES_PER_REQUEST", "2000"))
BRIDGE_TIMEFRAMES = tuple(
    value.strip() for value in os.getenv("BRIDGE_TIMEFRAMES", "M1,M15,H1,H4,D1").split(",")
    if value.strip()
)
BRIDGE_SIGNAL_TIMEFRAMES = ("15m", "1h", "4h")
MIN_CANDLES_FOR_DETECTION = max(200, int(os.getenv("MIN_CANDLES_FOR_DETECTION", "1000")))

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


def normalize_database_url(value: str | None = None, db_path: Path | None = None) -> str:
    value = (value if value is not None else os.getenv("DATABASE_URL", "")).strip()
    if not value:
        local_path = db_path or DB_PATH
        return "sqlite:///" + local_path.resolve().as_posix()
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    parsed = urlsplit(value)
    if parsed.scheme in {"postgres", "postgresql"} and (
        parsed.hostname or ""
    ).lower().endswith(".render.com"):
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.setdefault("sslmode", "require")
        value = urlunsplit(parsed._replace(query=urlencode(query)))
    return value


def parse_cors_origins(value: str | None = None, *, production: bool | None = None) -> tuple[str, ...]:
    is_production = PRODUCTION if production is None else production
    raw = value if value is not None else os.getenv("CORS_ORIGINS", "")
    if value is None and not raw.strip():
        raw = os.getenv("ALLOWED_ORIGINS", "")
    if not raw.strip() and not is_production:
        raw = "http://localhost:5173,http://127.0.0.1:5173"
    return tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())


def origin_is_allowed(origin: str | None) -> bool:
    return not origin or origin.rstrip("/") in CORS_ORIGINS


DATABASE_URL = normalize_database_url()
DATABASE_TYPE = "postgres" if urlsplit(DATABASE_URL).scheme.startswith("postgres") else "sqlite"
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))

# --- Security -----------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "goldsight").strip()
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "goldsight-app").strip()
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "15"))
REFRESH_TOKEN_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "7"))
FIELD_ENCRYPTION_KEY = os.getenv("FIELD_ENCRYPTION_KEY", "").strip()
CORS_ORIGINS = parse_cors_origins()
ALLOWED_ORIGINS = CORS_ORIGINS
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
    if LLM_PROVIDER in {"anthropic", "openai"} and not LLM_API_KEY:
        problems.append("LLM_API_KEY is required when LLM_PROVIDER is enabled")
    if DATABASE_TYPE != "postgres":
        problems.append("DATABASE_URL must point to PostgreSQL in production")
    if "*" in CORS_ORIGINS:
        problems.append("CORS_ORIGINS must not contain '*' when credentials are enabled")
    if any("localhost" in origin or "127.0.0.1" in origin for origin in CORS_ORIGINS):
        problems.append("CORS_ORIGINS must not use a local development origin in production")
    if COOKIE_SAMESITE == "none" and not COOKIE_SECURE:
        problems.append("COOKIE_SECURE must be enabled when COOKIE_SAMESITE is 'none'")
    if TELEGRAM_BOT_TOKEN and not TELEGRAM_WEBHOOK_SECRET:
        problems.append("TELEGRAM_WEBHOOK_SECRET is required when TELEGRAM_BOT_TOKEN is set")
    if TELEGRAM_BOT_TOKEN and not PUBLIC_BASE_URL:
        problems.append("PUBLIC_BASE_URL is required when TELEGRAM_BOT_TOKEN is set")
    if BRIDGE_API_KEY and len(BRIDGE_API_KEY) < 16:
        problems.append("BRIDGE_API_KEY must be at least 16 characters")
    smtp_configured = bool(SMTP_HOST)
    if smtp_configured and not all(
        (SMTP_USER, SMTP_PASSWORD, SMTP_TO)
    ):
        problems.append("SMTP_USER, SMTP_PASSWORD, and SMTP_TO are required when SMTP_HOST is set")
    if problems:
        raise RuntimeError("Production configuration is invalid: " + "; ".join(problems))


@lru_cache(maxsize=1)
def llm_configured() -> bool:
    """Return True when a usable provider + API key pair is present."""
    if LLM_PROVIDER not in {"anthropic", "openai"}:
        return False
    return bool(LLM_API_KEY)


def telegram_enabled() -> bool:
    """Return True when a bot token is configured (bot otherwise disabled)."""
    return bool(TELEGRAM_BOT_TOKEN)
