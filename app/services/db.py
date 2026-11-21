"""SQLite persistence for signals and chat history (stdlib sqlite3, thread-safe)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from app.config import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    final_score REAL NOT NULL,
    rule_score REAL NOT NULL,
    ml_probability REAL,
    entry REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit1 REAL NOT NULL,
    take_profit2 REAL NOT NULL,
    risk_reward REAL NOT NULL,
    atr REAL NOT NULL,
    price REAL NOT NULL,
    explanation TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    hit TEXT,
    result_pct REAL,
    closed_at TEXT,
    signal_time TEXT,
    grade TEXT,
    source TEXT NOT NULL DEFAULT 'live',
    setup TEXT,
    session TEXT
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    created_at TEXT NOT NULL,
    closed_at TEXT,
    symbol TEXT NOT NULL DEFAULT 'XAU/USD',
    timeframe TEXT,
    action TEXT,
    grade TEXT,
    entry REAL,
    stop_loss REAL,
    take_profit REAL,
    exit_price REAL,
    result_pct REAL,
    setup TEXT,
    session TEXT,
    notes TEXT NOT NULL DEFAULT '',
    screenshot TEXT,
    source TEXT NOT NULL DEFAULT 'signal',
    UNIQUE(signal_id, source)
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    username TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    is_active INTEGER NOT NULL DEFAULT 1,
    email_verified INTEGER NOT NULL DEFAULT 0,
    totp_secret TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    password_changed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    refresh_hash TEXT NOT NULL UNIQUE,
    family_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    ip TEXT,
    user_agent TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS email_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    ip TEXT NOT NULL,
    succeeded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    user_id INTEGER,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    ip TEXT
);
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    checked_at TEXT NOT NULL,
    status TEXT NOT NULL,
    hit TEXT,
    result_pct REAL,
    FOREIGN KEY(signal_id) REFERENCES signals(id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    """Return a process-wide SQLite connection with WAL-ish pragmas."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(SCHEMA)
        columns = {r["name"] for r in _conn.execute("PRAGMA table_info(signals)")}
        for column, declaration in (
            ("signal_time", "TEXT"), ("grade", "TEXT"),
            ("source", "TEXT NOT NULL DEFAULT 'live'"),
            ("setup", "TEXT"), ("session", "TEXT"),
        ):
            if column not in columns:
                _conn.execute(f"ALTER TABLE signals ADD COLUMN {column} {declaration}")
        existing = {r["name"] for r in _conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("signals", "journal_entries", "chat_messages",
                      "alerts", "alert_settings"):
            if table not in existing:
                continue
            cols = {r["name"] for r in _conn.execute(f"PRAGMA table_info({table})")}
            if "user_id" not in cols:
                _conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")
        _conn.commit()
    return _conn


def init_db() -> None:
    """Create tables if they do not exist."""
    get_conn()


def save_signal(sig: dict[str, Any]) -> int:
    """Insert a signal row and return its id."""
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            """INSERT INTO signals
               (created_at, timeframe, action, confidence, final_score, rule_score,
                ml_probability, entry, stop_loss, take_profit1, take_profit2,
                 risk_reward, atr, price, explanation, status, signal_time, grade,
                 source, setup, session)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'open', ?, ?, ?, ?, ?)""",
            (
                sig.get("created_at") or _now(),
                sig["timeframe"],
                sig["action"],
                float(sig["confidence"]),
                float(sig["final_score"]),
                float(sig["rule_score"]),
                sig.get("ml_probability"),
                float(sig["entry"]),
                float(sig["stop_loss"]),
                float(sig["take_profit1"]),
                float(sig["take_profit2"]),
                float(sig["risk_reward"]),
                float(sig["atr"]),
                float(sig["price"]),
                json.dumps(sig["explanation"]),
                sig.get("signal_time") or sig.get("created_at") or _now(),
                sig.get("grade") or "C",
                sig.get("source") or "live",
                sig.get("setup"),
                sig.get("session"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_signals(limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    """Return recent signals newest-first."""
    with _lock:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["explanation"] = json.loads(d["explanation"])
        except (TypeError, ValueError):
            d["explanation"] = {"summary": str(d["explanation"]), "reasons": []}
        out.append(d)
    return out


def update_signal_outcome(signal_id: int, status: str, hit: str | None,
                          result_pct: float | None) -> None:
    """Update open/closed status of a stored signal."""
    with _lock:
        conn = get_conn()
        conn.execute(
            """UPDATE signals SET status=?, hit=?, result_pct=?, closed_at=?
               WHERE id=?""",
            (status, hit, result_pct, _now(), signal_id),
        )
        conn.commit()


def save_chat(role: str, content: str, user_id: int | None = None) -> int:
    """Persist one chat message; returns id."""
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO chat_messages (created_at, role, content, user_id) VALUES (?,?,?,?)",
            (_now(), role, content, user_id),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_chat(limit: int = 100, user_id: int | None = None) -> list[dict[str, Any]]:
    """Return recent chat messages oldest-first."""
    with _lock:
        conn = get_conn()
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM chat_messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [dict(r) for r in reversed(rows)]


def clear_chat(user_id: int | None = None) -> None:
    """Delete all chat history."""
    with _lock:
        conn = get_conn()
        if user_id is None:
            conn.execute("DELETE FROM chat_messages")
        else:
            conn.execute("DELETE FROM chat_messages WHERE user_id=?", (user_id,))
        conn.commit()


def signal_stats() -> dict[str, Any]:
    """Aggregate hit statistics over closed signals."""
    with _lock:
        conn = get_conn()
        total = conn.execute("SELECT COUNT(*) c FROM signals").fetchone()["c"]
        closed = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE status='closed'"
        ).fetchone()["c"]
        tp = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE hit='TP'"
        ).fetchone()["c"]
        sl = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE hit='SL'"
        ).fetchone()["c"]
        open_n = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE status='open'"
        ).fetchone()["c"]
        avg = conn.execute(
            "SELECT AVG(result_pct) a FROM signals WHERE result_pct IS NOT NULL"
        ).fetchone()["a"]
    wins = tp
    losses = sl
    win_rate = (wins / closed * 100.0) if closed else 0.0
    return {
        "total": total,
        "closed": closed,
        "open": open_n,
        "hit_tp": tp,
        "hit_sl": sl,
        "win_rate": round(win_rate, 2),
        "avg_result_pct": round(avg, 4) if avg is not None else None,
    }


def ensure_journal_entry(signal: dict[str, Any]) -> bool:
    """Insert an automatic journal row for a resolved signal, once. Returns True if new."""
    with _lock:
        conn = get_conn()
        exists = conn.execute(
            "SELECT 1 FROM journal_entries WHERE signal_id=? AND source='signal'",
            (signal.get("id"),),
        ).fetchone()
        if exists:
            return False
        conn.execute(
            """INSERT OR IGNORE INTO journal_entries
               (signal_id, created_at, closed_at, symbol, timeframe, action, grade,
                entry, stop_loss, take_profit, exit_price, result_pct, setup, session,
                notes, screenshot, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'signal')""",
            (
                signal.get("id"), signal.get("created_at") or _now(),
                signal.get("closed_at"), "XAU/USD", signal.get("timeframe"),
                signal.get("action"), signal.get("grade"), signal.get("entry"),
                signal.get("stop_loss"), signal.get("take_profit2"),
                signal.get("entry"), signal.get("result_pct"),
                signal.get("setup"), signal.get("session"),
                f"Auto-imported from signal #{signal.get('id')}", None,
            ),
        )
        conn.commit()
        return True


def save_journal_entry(entry: dict[str, Any], user_id: int | None = None) -> int:
    """Store a manually logged trade and return its id."""
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            """INSERT INTO journal_entries
               (created_at, closed_at, symbol, timeframe, action, grade, entry,
                stop_loss, take_profit, exit_price, result_pct, setup, session,
                notes, screenshot, source, user_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'manual', ?)""",
            (
                entry.get("created_at") or _now(), entry.get("closed_at") or _now(),
                entry.get("symbol") or "XAU/USD", entry.get("timeframe") or "1h",
                entry.get("action") or "BUY", entry.get("grade") or "C",
                entry.get("entry"), entry.get("stop_loss"), entry.get("take_profit"),
                entry.get("exit_price"), entry.get("result_pct"),
                entry.get("setup"), entry.get("session"),
                entry.get("notes") or "", entry.get("screenshot"), user_id,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_journal(limit: int = 200, user_id: int | None = None) -> list[dict[str, Any]]:
    """Return journal trades newest-first."""
    with _lock:
        conn = get_conn()
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM journal_entries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM journal_entries WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def delete_journal_entry(entry_id: int, user_id: int | None = None) -> bool:
    with _lock:
        conn = get_conn()
        if user_id is None:
            cur = conn.execute("DELETE FROM journal_entries WHERE id=?", (entry_id,))
        else:
            cur = conn.execute("DELETE FROM journal_entries WHERE id=? AND user_id=?",
                               (entry_id, user_id))
        conn.commit()
    return cur.rowcount > 0
