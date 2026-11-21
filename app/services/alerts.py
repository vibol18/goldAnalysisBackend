"""Background alert scanner, SQLite storage, SSE pub/sub and notification delivery."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
import threading
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

import httpx

from app import config
from app.services import concept_context, data, db
from app.services.signal_engine import generate_signal

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational analysis only. Not financial advice. Signals can be wrong. "
    "Never risk money you can't afford to lose."
)

TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h", "1D")
MIN_GRADES: tuple[str, ...] = ("A+", "A", "B")
CHANNEL_NAMES: tuple[str, ...] = ("browser", "sound", "toast", "telegram", "email")
DEFAULT_CHANNELS: dict[str, bool] = {
    "browser": True, "sound": False, "toast": True,
    "telegram": False, "email": False,
}
DEFAULT_SETTINGS: dict[str, Any] = {
    "timeframes": ["15m", "1h", "4h"],
    "min_grade": "A",
    "channels": dict(DEFAULT_CHANNELS),
}
SCAN_INTERVAL = 60.0
KEEPALIVE_SECONDS = 20
NARRATIVE_LIMIT = 240

_lock = threading.Lock()
_subscribers: set[asyncio.Queue] = set()
_dedupe: set[tuple[str, str, str]] = set()
_DEDUPE_LIMIT = 1000


def _remember(key: tuple[str, str, str]) -> None:
    """Track a fired alert key without letting the in-memory set grow forever."""
    if len(_dedupe) >= _DEDUPE_LIMIT:
        _dedupe.clear()
    _dedupe.add(key)


# --- environment-backed configuration -------------------------------------
def telegram_configured() -> bool:
    return bool(
        os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        and os.getenv("TELEGRAM_CHAT_ID", "").strip()
    )


def email_configured() -> bool:
    return all(
        os.getenv(name, "").strip()
        for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_TO")
    )


# --- lazy storage -----------------------------------------------------------
def _ensure_tables() -> Any:
    conn = db.get_conn()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS alert_settings "
        "(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS alerts (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               time TEXT NOT NULL,
               type TEXT NOT NULL,
               timeframe TEXT NOT NULL,
               action TEXT NOT NULL,
               grade TEXT NOT NULL,
               price REAL,
               message TEXT NOT NULL,
               channels TEXT NOT NULL
           )"""
    )
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_settings(payload: dict[str, Any] | None) -> dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    raw_tfs = src.get("timeframes", DEFAULT_SETTINGS["timeframes"])
    if not isinstance(raw_tfs, (list, tuple)):
        raw_tfs = DEFAULT_SETTINGS["timeframes"]
    timeframes = [tf for tf in TIMEFRAMES if tf in raw_tfs]
    min_grade = src.get("min_grade", DEFAULT_SETTINGS["min_grade"])
    if min_grade not in MIN_GRADES:
        min_grade = DEFAULT_SETTINGS["min_grade"]
    raw_channels = src.get("channels", {})
    if not isinstance(raw_channels, dict):
        raw_channels = {}
    channels = {
        name: bool(raw_channels.get(name, DEFAULT_CHANNELS[name]))
        for name in CHANNEL_NAMES
    }
    return {"timeframes": timeframes, "min_grade": min_grade, "channels": channels}


def get_settings() -> dict[str, Any]:
    with _lock:
        row = _ensure_tables().execute(
            "SELECT payload FROM alert_settings WHERE id=1"
        ).fetchone()
    stored = {}
    if row is not None:
        try:
            stored = json.loads(row["payload"])
        except (TypeError, ValueError):
            stored = {}
    result = _coerce_settings(stored)
    result["telegram_configured"] = telegram_configured()
    result["email_configured"] = email_configured()
    result["disclaimer"] = DISCLAIMER
    return result


def save_settings(payload: dict[str, Any]) -> dict[str, Any]:
    clean = _coerce_settings(payload)
    persisted = {key: clean[key] for key in ("timeframes", "min_grade", "channels")}
    with _lock:
        conn = _ensure_tables()
        conn.execute(
            "INSERT INTO alert_settings (id, payload) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
            (json.dumps(persisted),),
        )
        conn.commit()
    return get_settings()


def _row_to_alert(row: Any) -> dict[str, Any]:
    try:
        channels = json.loads(row["channels"])
    except (TypeError, ValueError):
        channels = []
    return {
        "id": int(row["id"]),
        "time": row["time"],
        "type": row["type"],
        "timeframe": row["timeframe"],
        "action": row["action"],
        "grade": row["grade"],
        "price": row["price"],
        "message": row["message"],
        "channels": channels if isinstance(channels, list) else [],
    }


def store_alert(alert: dict[str, Any]) -> dict[str, Any]:
    timestamp = alert.get("time") or _now()
    channels = alert.get("channels") or []
    with _lock:
        conn = _ensure_tables()
        cur = conn.execute(
            """INSERT INTO alerts (time, type, timeframe, action, grade, price,
                                   message, channels)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                timestamp,
                str(alert.get("type") or "test"),
                str(alert.get("timeframe") or ""),
                str(alert.get("action") or ""),
                str(alert.get("grade") or ""),
                alert.get("price"),
                str(alert.get("message") or ""),
                json.dumps(channels),
            ),
        )
        conn.commit()
        alert_id = int(cur.lastrowid)
    stored = dict(alert)
    stored.update({"id": alert_id, "time": timestamp, "channels": list(channels)})
    return stored


def list_alerts(limit: int = 200) -> list[dict[str, Any]]:
    with _lock:
        rows = _ensure_tables().execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)
        ).fetchall()
    return [_row_to_alert(row) for row in rows]


def clear_alerts() -> None:
    with _lock:
        conn = _ensure_tables()
        conn.execute("DELETE FROM alerts")
        conn.commit()


# --- asyncio pub/sub --------------------------------------------------------
def subscribe() -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    _subscribers.discard(queue)


async def publish(alert: dict[str, Any]) -> dict[str, Any]:
    stored = store_alert(alert)
    for queue in list(_subscribers):
        try:
            queue.put_nowait(stored)
        except asyncio.QueueFull:  # pragma: no cover - unbounded queues
            logger.debug("dropped alert for a slow SSE subscriber")
    return stored


# --- notification delivery --------------------------------------------------
def send_telegram(text: str) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not (token and chat_id):
        return "Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)."
    try:
        response = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=5.0,
        )
        if response.is_success:
            return "Telegram message sent."
        return (f"Telegram failed (HTTP {response.status_code}): "
                f"{response.text[:200]}")
    except Exception as exc:  # noqa: BLE001 - network/SDK errors vary
        return f"Telegram failed: {exc}"


def send_email(text: str) -> str:
    host = os.getenv("SMTP_HOST", "").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    recipient = os.getenv("SMTP_TO", "").strip()
    if not (host and user and password and recipient):
        return "Email not configured (set SMTP_HOST, SMTP_USER, SMTP_PASSWORD and SMTP_TO)."
    try:
        port = int(os.getenv("SMTP_PORT", "587").strip() or 587)
    except ValueError:
        port = 587
    sender = os.getenv("SMTP_FROM", "").strip() or user
    message = EmailMessage()
    message["Subject"] = "GoldSight alert"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(text)
    try:
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(message)
        return f"Email sent to {recipient}."
    except Exception as exc:  # noqa: BLE001 - SMTP errors vary
        return f"Email failed: {exc}"


def _notification_text(alert: dict[str, Any]) -> str:
    return (
        f"GoldSight {alert.get('type', 'alert')} alert - "
        f"{alert.get('timeframe', '')} | grade {alert.get('grade', '')}\n"
        f"{alert.get('message', '')}\n\n{DISCLAIMER}"
    )


async def notify(alert: dict[str, Any]) -> dict[str, str]:
    settings = get_settings()
    channels = settings["channels"]
    results: dict[str, str] = {}
    text = _notification_text(alert)
    if not channels["telegram"]:
        results["telegram"] = "Telegram disabled in settings."
    elif not telegram_configured():
        results["telegram"] = (
            "Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)."
        )
    else:
        results["telegram"] = await asyncio.to_thread(send_telegram, text)

    if not channels["email"]:
        results["email"] = "Email disabled in settings."
    elif not email_configured():
        results["email"] = (
            "Email not configured (set SMTP_HOST, SMTP_USER, SMTP_PASSWORD and SMTP_TO)."
        )
    else:
        results["email"] = await asyncio.to_thread(send_email, text)
    return results


# --- scanner ----------------------------------------------------------------
def _grade_rank(grade: Any) -> int:
    return {"A+": 2, "A": 1, "B": 0, "C": -1}.get(str(grade).strip().upper(), -99)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _signal_message(signal: dict[str, Any], timeframe: str) -> str:
    action = str(signal.get("action") or "").upper()
    grade = signal.get("grade") or "n/a"
    zone = signal.get("entry_zone")
    if isinstance(zone, (list, tuple)) and len(zone) == 2:
        entry = f"{_fmt(zone[0])}-{_fmt(zone[1])}"
    else:
        entry = _fmt(signal.get("entry"))
    explanation = signal.get("explanation")
    narrative = ""
    if isinstance(explanation, dict):
        narrative = str(explanation.get("narrative") or "")
    elif explanation:
        narrative = str(explanation)
    parts = [
        f"{action} XAU/USD on {timeframe} - grade {grade} "
        f"(price {_fmt(signal.get('price'))}).",
        f"Entry: {entry} | SL: {_fmt(signal.get('stop_loss'))} | "
        f"TP1: {_fmt(signal.get('take_profit1'))} | "
        f"TP2: {_fmt(signal.get('take_profit2'))} | "
        f"TP3: {_fmt(signal.get('take_profit3'))}.",
    ]
    if narrative:
        parts.append(narrative[:NARRATIVE_LIMIT])
    return " ".join(parts)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_signal_event(signal: dict[str, Any], high: float,
                         low: float) -> tuple[str | None, float | None]:
    action = str(signal.get("action") or "").upper()
    sl = _number(signal.get("stop_loss"))
    tp1 = _number(signal.get("take_profit1"))
    tp2 = _number(signal.get("take_profit2"))
    entry = _number(signal.get("entry"))
    zone = signal.get("entry_zone")
    if isinstance(zone, str):
        try:
            zone = json.loads(zone)
        except (TypeError, ValueError):
            zone = None
    zone_low = zone_high = None
    if isinstance(zone, (list, tuple)) and len(zone) == 2:
        zone_low, zone_high = _number(zone[0]), _number(zone[1])
    if zone_low is not None and zone_high is not None:
        zlo, zhi = sorted((zone_low, zone_high))
    elif entry is not None:
        pad = max(abs(entry) * 0.0005, 0.01)
        zlo, zhi = entry - pad, entry + pad
    else:
        zlo = zhi = None
    if zlo is not None and zhi is not None:
        zlo, zhi = float(zlo), float(zhi)

    sl_hit = sl is not None and (
        (action == "BUY" and low <= sl) or (action == "SELL" and high >= sl)
    )
    tp2_hit = tp2 is not None and (
        (action == "BUY" and high >= tp2) or (action == "SELL" and low <= tp2)
    )
    tp1_hit = tp1 is not None and (
        (action == "BUY" and high >= tp1) or (action == "SELL" and low <= tp1)
    )
    entry_hit = (zlo is not None and zhi is not None
                 and low <= zhi and high >= zlo)
    if sl_hit:
        return "sl", sl
    if tp2_hit:
        return "tp", tp2
    if tp1_hit:
        return "tp", tp1
    if entry_hit:
        return "entry", (zlo + zhi) / 2.0
    return None, None


def _event_message(kind: str, signal: dict[str, Any], level: float | None,
                   timeframe: str, price: float | None) -> str:
    labels = {"entry": "Entry zone touched", "tp": "Take-profit touched",
              "sl": "Stop-loss hit"}
    return (
        f"{labels.get(kind, kind.upper())} on {signal.get('action', '')} "
        f"signal #{signal.get('id')} ({timeframe}, grade "
        f"{signal.get('grade') or 'n/a'}): {_fmt(level)} "
        f"(last close {_fmt(price)})."
    )


async def scan_once() -> list[dict[str, Any]]:
    settings = get_settings()
    enabled_channels = [
        name for name in CHANNEL_NAMES if settings["channels"][name]
    ]
    try:
        open_signals = await asyncio.to_thread(db.list_signals, 200)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load open signals: %s", exc)
        open_signals = []
    created: list[dict[str, Any]] = []

    for timeframe in settings["timeframes"]:
        if timeframe not in TIMEFRAMES:
            continue
        try:
            entry = await asyncio.to_thread(
                data.get_ohlcv, config.GOLD_TICKER, timeframe)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alert scan: %s data fetch failed: %s", timeframe, exc)
            continue
        frame = getattr(entry, "df", entry)
        if frame is None or len(frame) < 31:
            continue
        closed = frame.iloc[:-1]
        try:
            last = closed.iloc[-1]
            high, low = float(last["High"]), float(last["Low"])
            close_price = float(last["Close"])
            signal_time = str(closed.index[-1])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("alert scan: bad %s candle: %s", timeframe, exc)
            continue

        try:
            context = await asyncio.to_thread(
                concept_context.build_context, timeframe)
        except Exception as exc:  # noqa: BLE001
            logger.debug("alert scan: no %s context: %s", timeframe, exc)
            context = {}
        try:
            signal = await asyncio.to_thread(
                generate_signal, closed, timeframe,
                train_model=False, concept_context=context)
            payload = signal.to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning("alert scan: %s signal failed: %s", timeframe, exc)
            continue

        action = str(payload.get("action") or "").upper()
        grade = payload.get("grade")
        if action in ("BUY", "SELL") and _grade_rank(grade) >= _grade_rank(
                settings["min_grade"]):
            key = ("signal", timeframe, signal_time)
            if key not in _dedupe:
                _remember(key)
                alert = await publish({
                    "type": "signal",
                    "timeframe": timeframe,
                    "action": action,
                    "grade": str(grade),
                    "price": _number(payload.get("price")),
                    "message": _signal_message(payload, timeframe),
                    "channels": list(enabled_channels),
                })
                created.append(alert)

        for stored_signal in open_signals:
            if (stored_signal.get("timeframe") != timeframe
                    or stored_signal.get("status") != "open"):
                continue
            if str(stored_signal.get("action") or "").upper() not in ("BUY", "SELL"):
                continue
            kind, level = _detect_signal_event(stored_signal, high, low)
            if kind is None:
                continue
            key = (kind, timeframe, str(stored_signal.get("id")))
            if key in _dedupe:
                continue
            _remember(key)
            alert = await publish({
                "type": kind,
                "timeframe": timeframe,
                "action": str(stored_signal.get("action") or "").upper(),
                "grade": str(stored_signal.get("grade") or ""),
                "price": close_price,
                "message": _event_message(kind, stored_signal, level,
                                          timeframe, close_price),
                "channels": list(enabled_channels),
            })
            created.append(alert)
    return created


async def scanner_loop() -> None:
    """Run the scanner once a minute until cancelled."""
    while True:
        try:
            await scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            logger.warning("alert scanner pass failed: %s", exc)
        await asyncio.sleep(SCAN_INTERVAL)
