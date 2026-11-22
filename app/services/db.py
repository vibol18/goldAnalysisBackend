"""SQLAlchemy-backed persistence shared by SQLite and PostgreSQL."""
from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from app import config

logger = logging.getLogger(__name__)
_lock = threading.RLock()
_shared_connection: ConnectionAdapter | None = None
_initialized = False

metadata = MetaData()

signals = Table(
    "signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(32), nullable=False, default="XAU/USD"),
    Column("created_at", String(40), nullable=False),
    Column("timeframe", String(16), nullable=False),
    Column("action", String(16), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("final_score", Float, nullable=False),
    Column("rule_score", Float, nullable=False),
    Column("ml_probability", Float),
    Column("entry", Float, nullable=False),
    Column("stop_loss", Float, nullable=False),
    Column("take_profit1", Float, nullable=False),
    Column("take_profit2", Float, nullable=False),
    Column("risk_reward", Float, nullable=False),
    Column("atr", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("explanation", Text, nullable=False),
    Column("status", String(16), nullable=False, default="open"),
    Column("hit", String(16)),
    Column("result_pct", Float),
    Column("closed_at", String(40)),
    Column("signal_time", String(40)),
    Column("grade", String(8)),
    Column("source", String(32), nullable=False, default="live"),
    Column("setup", String(120)),
    Column("session", String(32)),
    Column("user_id", Integer),
    UniqueConstraint("symbol", "timeframe", "signal_time", name="uq_signal_candle"),
)

chat_messages = Table(
    "chat_messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", String(40), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("user_id", Integer),
)

journal_entries = Table(
    "journal_entries", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("signal_id", Integer),
    Column("created_at", String(40), nullable=False),
    Column("closed_at", String(40)),
    Column("symbol", String(32), nullable=False, default="XAU/USD"),
    Column("timeframe", String(16)),
    Column("action", String(16)),
    Column("grade", String(8)),
    Column("entry", Float),
    Column("stop_loss", Float),
    Column("take_profit", Float),
    Column("exit_price", Float),
    Column("result_pct", Float),
    Column("setup", String(120)),
    Column("session", String(32)),
    Column("notes", Text, nullable=False, default=""),
    Column("screenshot", Text),
    Column("source", String(32), nullable=False, default="signal"),
    Column("user_id", Integer),
    UniqueConstraint("signal_id", "source", name="uq_journal_signal_source"),
)

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("email", String(254), nullable=False, unique=True),
    Column("username", String(64), nullable=False),
    Column("password_hash", Text, nullable=False),
    Column("role", String(16), nullable=False, default="user"),
    Column("is_active", Integer, nullable=False, default=1),
    Column("email_verified", Integer, nullable=False, default=0),
    Column("totp_secret", Text),
    Column("failed_attempts", Integer, nullable=False, default=0),
    Column("locked_until", String(40)),
    Column("password_changed_at", String(40), nullable=False),
    Column("created_at", String(40), nullable=False),
)

sessions = Table(
    "sessions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("token_hash", String(64), nullable=False, unique=True),
    Column("refresh_hash", String(64), nullable=False, unique=True),
    Column("family_id", String(64), nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("expires_at", String(40), nullable=False),
    Column("revoked", Integer, nullable=False, default=0),
    Column("ip", String(64)),
    Column("user_agent", String(200)),
)

email_tokens = Table(
    "email_tokens", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("token_hash", String(64), nullable=False, unique=True),
    Column("kind", String(32), nullable=False),
    Column("expires_at", String(40), nullable=False),
    Column("used", Integer, nullable=False, default=0),
    Column("created_at", String(40), nullable=False),
)

login_attempts = Table(
    "login_attempts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("email", String(254), nullable=False),
    Column("ip", String(64), nullable=False),
    Column("succeeded", Integer, nullable=False, default=0),
    Column("created_at", String(40), nullable=False),
)

audit_log = Table(
    "audit_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", String(40), nullable=False),
    Column("user_id", Integer),
    Column("action", String(64), nullable=False),
    Column("detail", Text, nullable=False),
    Column("ip", String(64)),
)

signal_outcomes = Table(
    "signal_outcomes", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("signal_id", Integer, ForeignKey("signals.id"), nullable=False),
    Column("checked_at", String(40), nullable=False),
    Column("status", String(16), nullable=False),
    Column("hit", String(16)),
    Column("result_pct", Float),
)

alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("time", String(40), nullable=False),
    Column("type", String(24), nullable=False),
    Column("timeframe", String(16), nullable=False),
    Column("action", String(16), nullable=False),
    Column("grade", String(8), nullable=False),
    Column("price", Float),
    Column("message", Text, nullable=False),
    Column("channels", Text, nullable=False),
    Column("user_id", Integer),
)

user_alert_settings = Table(
    "user_alert_settings", metadata,
    Column("user_id", Integer, primary_key=True),
    Column("payload", Text, nullable=False),
)

app_settings = Table(
    "app_settings", metadata,
    Column("key", String(120), primary_key=True),
    Column("value", Text, nullable=False),
    Column("updated_at", String(40), nullable=False),
)

model_metadata = Table(
    "model_metadata", metadata,
    Column("name", String(120), primary_key=True),
    Column("value", Text, nullable=False),
    Column("updated_at", String(40), nullable=False),
)

telegram_users = Table(
    "telegram_users", metadata,
    Column("chat_id", Integer, primary_key=True),
    Column("username", String(64)),
    Column("first_name", String(128)),
    Column("language", String(8), nullable=False, default="en"),
    Column("is_active", Integer, nullable=False, default=1),
    Column("min_grade", String(8), nullable=False, default="C"),
    Column("timeframes", String(120), nullable=False, default="15m,1h,4h"),
    Column("symbols", String(200), nullable=False, default="XAU/USD"),
    Column("quiet_start", String(8)),
    Column("quiet_end", String(8)),
    Column("daily_cap", Integer, nullable=False, default=50),
    Column("user_id", Integer),
    Column("created_at", String(40), nullable=False),
    Column("last_sent_at", String(40)),
)

telegram_deliveries = Table(
    "telegram_deliveries", metadata,
    Column("chat_id", Integer, nullable=False),
    Column("signal_id", Integer, nullable=False),
    Column("kind", String(32), nullable=False, default="signal"),
    Column("message_id", Integer),
    Column("created_at", String(40), nullable=False),
    UniqueConstraint("chat_id", "signal_id", "kind", name="uq_tg_delivery"),
)

telegram_link_codes = Table(
    "telegram_link_codes", metadata,
    Column("code", String(32), primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("chat_id", Integer),
    Column("expires_at", String(40), nullable=False),
    Column("used", Integer, nullable=False, default=0),
    Column("created_at", String(40), nullable=False),
)


bridge_candles = Table(
    "bridge_candles", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(32), nullable=False),
    Column("timeframe", String(8), nullable=False),
    Column("broker_time", Integer, nullable=False),
    Column("time", Integer, nullable=False),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Float, nullable=False, default=0.0),
    Column("spread", Float, nullable=False, default=0.0),
    Column("real_volume", Float),
    Column("received_at", String(40), nullable=False),
    UniqueConstraint("symbol", "timeframe", "broker_time", name="uq_bridge_candle"),
)

bridge_ticks = Table(
    "bridge_ticks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(32), nullable=False),
    Column("broker_time", Integer, nullable=False),
    Column("time", Integer, nullable=False),
    Column("bid", Float, nullable=False),
    Column("ask", Float, nullable=False),
    Column("spread", Float, nullable=False),
    Column("digits", Integer),
    Column("point", Float),
    Column("contract_size", Float),
    Column("broker", String(64)),
    Column("server", String(64)),
    Column("received_at", String(40), nullable=False),
)

bridge_meta = Table(
    "bridge_meta", metadata,
    Column("symbol", String(32), primary_key=True),
    Column("broker", String(64), nullable=False, default=""),
    Column("server", String(64), nullable=False, default=""),
    Column("digits", Integer, nullable=False, default=2),
    Column("point", Float, nullable=False, default=0.01),
    Column("contract_size", Float, nullable=False, default=100.0),
    Column("timezone_offset_seconds", Integer, nullable=False, default=0),
    Column("currency", String(16), nullable=False, default="USD"),
    Column("last_candle_at", String(40)),
    Column("last_tick_at", String(40)),
    Column("updated_at", String(40), nullable=False),
)


class ResultAdapter:
    def __init__(self, result: Any, lastrowid: int | None = None) -> None:
        self._result = result
        self.lastrowid = lastrowid
        self.rowcount = result.rowcount

    def fetchone(self) -> Any:
        return self._result.mappings().fetchone()

    def fetchall(self) -> list[Any]:
        return list(self._result.mappings().fetchall())


class ConnectionAdapter:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def execute(
        self, statement: str, parameters: Sequence[Any] | Mapping[str, Any] = (),
    ) -> ResultAdapter:
        values: Mapping[str, Any]
        if isinstance(parameters, Mapping):
            values = parameters
        else:
            parts = statement.split("?")
            if len(parts) - 1 != len(parameters):
                raise ValueError("SQL placeholder count does not match parameters")
            statement = parts[0]
            values = {}
            for index, value in enumerate(parameters):
                name = f"p{index}"
                statement += f":{name}" + parts[index + 1]
                values[name] = value

        insert = statement.lstrip().upper().startswith("INSERT")
        returns_id = insert and "RETURNING" not in statement.upper()
        table_match = re.search(r"\bINSERT\s+INTO\s+([a-z_]+)", statement, re.I)
        if self._connection.dialect.name == "postgresql" and returns_id and table_match:
            table_name = table_match.group(1).lower()
            if table_name in {
                "signals", "chat_messages", "journal_entries", "users", "sessions",
                "email_tokens", "login_attempts", "audit_log", "signal_outcomes",
                "alerts",
            } and "ON CONFLICT" not in statement.upper():
                statement += " RETURNING id"

        try:
            result = self._connection.execute(text(statement), values)
            lastrowid = None
            if self._connection.dialect.name == "postgresql" and "RETURNING id" in statement.upper():
                lastrowid = result.scalar_one_or_none()
            elif insert:
                lastrowid = result.lastrowid
            return ResultAdapter(result, int(lastrowid) if lastrowid is not None else None)
        except Exception:
            self._connection.rollback()
            raise

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()


def create_database_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    connect_args: dict[str, Any] = {}
    engine_options: dict[str, Any] = {"pool_pre_ping": True}
    if url.get_backend_name() == "sqlite":
        connect_args = {"check_same_thread": False, "timeout": 30}
        engine_options["connect_args"] = connect_args
        if url.database == ":memory:":
            engine_options["poolclass"] = StaticPool
        elif url.database:
            Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url, **engine_options)
    if url.get_backend_name() == "sqlite":
        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if url.database != ":memory:":
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


engine = create_database_engine(config.DATABASE_URL)


def get_conn() -> ConnectionAdapter:
    global _shared_connection
    with _lock:
        stale = (
            _shared_connection is None
            or _shared_connection._connection.closed
            or _shared_connection._connection.engine is not engine
        )
        if stale:
            if _shared_connection is not None and not _shared_connection._connection.closed:
                _shared_connection._connection.close()
            _shared_connection = ConnectionAdapter(engine.connect())
        return _shared_connection


def init_db() -> None:
    """Apply portable migrations before serving requests."""
    global _initialized
    if _initialized:
        return
    migration_config = AlembicConfig()
    migration_config.set_main_option("script_location", str(config.BASE_DIR / "migrations"))
    migration_connection = engine.connect()
    migration_config.attributes["connection"] = migration_connection
    try:
        command.upgrade(migration_config, "head")
    except Exception:
        migration_connection.rollback()
        logger.exception("Database migrations failed")
        raise
    finally:
        migration_connection.close()
    _initialized = True


def close_db() -> None:
    global _initialized, _shared_connection
    with _lock:
        if _shared_connection is not None:
            if not _shared_connection._connection.closed:
                _shared_connection._connection.close()
            _shared_connection = None
        engine.dispose()
        _initialized = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_signal(sig: dict[str, Any]) -> int:
    signal_time = sig.get("signal_time") or sig.get("created_at") or _now()
    symbol = sig.get("symbol") or "XAU/USD"
    with _lock:
        conn = get_conn()
        existing = _find_signal_id_optional(symbol, sig["timeframe"], signal_time)
        if existing is not None:
            return existing
        try:
            cur = conn.execute(
                """INSERT INTO signals
                   (symbol, created_at, timeframe, action, confidence, final_score,
                    rule_score, ml_probability, entry, stop_loss, take_profit1,
                    take_profit2, risk_reward, atr, price, explanation, status,
                    signal_time, grade, source, setup, session, user_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?,?,?,?)""",
                (
                    symbol, sig.get("created_at") or _now(), sig["timeframe"],
                    sig["action"], float(sig["confidence"]), float(sig["final_score"]),
                    float(sig["rule_score"]), sig.get("ml_probability"),
                    float(sig["entry"]), float(sig["stop_loss"]),
                    float(sig["take_profit1"]), float(sig["take_profit2"]),
                    float(sig["risk_reward"]), float(sig["atr"]), float(sig["price"]),
                    json.dumps(sig["explanation"]), signal_time, sig.get("grade") or "C",
                    sig.get("source") or "live", sig.get("setup"), sig.get("session"),
                    sig.get("user_id"),
                ),
            )
        except IntegrityError:
            duplicate = _find_signal_id_optional(symbol, sig["timeframe"], signal_time)
            if duplicate is None:
                raise
            return duplicate
        conn.commit()
        return int(cur.lastrowid or _find_signal_id(symbol, sig["timeframe"], signal_time))


def _find_signal_id_optional(
    symbol: str, timeframe: str, signal_time: str,
) -> int | None:
    row = get_conn().execute(
        "SELECT id FROM signals WHERE symbol=? AND timeframe=? AND signal_time=?",
        (symbol, timeframe, signal_time),
    ).fetchone()
    return int(row["id"]) if row else None


def _find_signal_id(symbol: str, timeframe: str, signal_time: str) -> int:
    signal_id = _find_signal_id_optional(symbol, timeframe, signal_time)
    if signal_id is None:
        raise RuntimeError("Signal was not persisted")
    return signal_id


def list_signals(limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    with _lock:
        rows = get_conn().execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["explanation"] = json.loads(item["explanation"])
        except (TypeError, ValueError):
            item["explanation"] = {"summary": str(item["explanation"]), "reasons": []}
        out.append(item)
    return out


def update_signal_outcome(
    signal_id: int, status: str, hit: str | None, result_pct: float | None,
) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """UPDATE signals SET status=?, hit=?, result_pct=?, closed_at=?
               WHERE id=?""",
            (status, hit, result_pct, _now(), signal_id),
        )
        conn.commit()


def save_chat(role: str, content: str, user_id: int | None = None) -> int:
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO chat_messages (created_at, role, content, user_id) VALUES (?,?,?,?)",
            (_now(), role, content, user_id),
        )
        conn.commit()
        if cur.lastrowid is None:
            raise RuntimeError("Chat message was not persisted")
        return cur.lastrowid


def list_chat(limit: int = 100, user_id: int | None = None) -> list[dict[str, Any]]:
    with _lock:
        conn = get_conn()
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM chat_messages ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [dict(row) for row in reversed(rows)]


def clear_chat(user_id: int | None = None) -> None:
    with _lock:
        conn = get_conn()
        if user_id is None:
            conn.execute("DELETE FROM chat_messages")
        else:
            conn.execute("DELETE FROM chat_messages WHERE user_id=?", (user_id,))
        conn.commit()


def signal_stats() -> dict[str, Any]:
    with _lock:
        conn = get_conn()
        total = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE COALESCE(source,'live')!='legacy_data'",
        ).fetchone()["c"]
        closed = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE status='closed' "
            "AND COALESCE(source,'live')!='legacy_data'",
        ).fetchone()["c"]
        tp = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE hit='TP' "
            "AND COALESCE(source,'live')!='legacy_data'",
        ).fetchone()["c"]
        sl = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE hit='SL' "
            "AND COALESCE(source,'live')!='legacy_data'",
        ).fetchone()["c"]
        open_count = conn.execute(
            "SELECT COUNT(*) c FROM signals WHERE status='open' "
            "AND COALESCE(source,'live')!='legacy_data'",
        ).fetchone()["c"]
        average = conn.execute(
            "SELECT AVG(result_pct) a FROM signals WHERE result_pct IS NOT NULL "
            "AND COALESCE(source,'live')!='legacy_data'",
        ).fetchone()["a"]
    return {
        "total": total,
        "closed": closed,
        "open": open_count,
        "hit_tp": tp,
        "hit_sl": sl,
        "win_rate": round(tp / closed * 100.0, 2) if closed else 0.0,
        "avg_result_pct": round(average, 4) if average is not None else None,
    }


def ensure_journal_entry(signal: dict[str, Any]) -> bool:
    with _lock:
        conn = get_conn()
        result = conn.execute(
            """INSERT INTO journal_entries
               (signal_id, created_at, closed_at, symbol, timeframe, action, grade,
                entry, stop_loss, take_profit, exit_price, result_pct, setup, session,
                notes, screenshot, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'signal')
               ON CONFLICT(signal_id, source) DO NOTHING""",
            (
                signal.get("id"), signal.get("created_at") or _now(),
                signal.get("closed_at"), signal.get("symbol") or "XAU/USD",
                signal.get("timeframe"), signal.get("action"), signal.get("grade"),
                signal.get("entry"), signal.get("stop_loss"),
                signal.get("take_profit2"), signal.get("entry"),
                signal.get("result_pct"), signal.get("setup"), signal.get("session"),
                f"Auto-imported from signal #{signal.get('id')}", None,
            ),
        )
        conn.commit()
        return result.rowcount > 0


def save_journal_entry(entry: dict[str, Any], user_id: int | None = None) -> int:
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
                entry.get("exit_price"), entry.get("result_pct"), entry.get("setup"),
                entry.get("session"), entry.get("notes") or "", entry.get("screenshot"),
                user_id,
            ),
        )
        conn.commit()
        if cur.lastrowid is None:
            raise RuntimeError("Journal entry was not persisted")
        return cur.lastrowid


def list_journal(limit: int = 200, user_id: int | None = None) -> list[dict[str, Any]]:
    with _lock:
        conn = get_conn()
        if user_id is None:
            rows = conn.execute(
                "SELECT * FROM journal_entries ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM journal_entries WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return [dict(row) for row in rows]


def delete_journal_entry(entry_id: int, user_id: int | None = None) -> bool:
    with _lock:
        conn = get_conn()
        if user_id is None:
            result = conn.execute("DELETE FROM journal_entries WHERE id=?", (entry_id,))
        else:
            result = conn.execute(
                "DELETE FROM journal_entries WHERE id=? AND user_id=?", (entry_id, user_id),
            )
        conn.commit()
    return result.rowcount > 0


def get_app_setting(key: str) -> str | None:
    with _lock:
        row = get_conn().execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_app_setting(key: str, value: str) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO app_settings (key, value, updated_at) VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, _now()),
        )
        conn.commit()


def set_model_metadata(name: str, value: dict[str, Any]) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO model_metadata (name, value, updated_at) VALUES (?,?,?)
               ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (name, json.dumps(value, default=str), _now()),
        )
        conn.commit()


def get_model_metadata(name: str) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute("SELECT value FROM model_metadata WHERE name=?", (name,)).fetchone()
    return json.loads(row["value"]) if row else None


def tg_upsert_user(chat_id: int, username: str | None, first_name: str | None) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO telegram_users
               (chat_id, username, first_name, language, is_active, min_grade,
                timeframes, symbols, daily_cap, created_at)
               VALUES (?,?,?,'en',1,'C','15m,1h,4h','XAU/USD',?,?)
               ON CONFLICT(chat_id) DO UPDATE SET
                 username=excluded.username, first_name=excluded.first_name,
                 is_active=1""",
            (chat_id, username, first_name, config.TELEGRAM_DAILY_CAP, _now()),
        )
        conn.commit()


def tg_get_user(chat_id: int) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute(
            "SELECT * FROM telegram_users WHERE chat_id=?", (chat_id,),
        ).fetchone()
    return dict(row) if row else None


def tg_set_active(chat_id: int, active: bool) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            "UPDATE telegram_users SET is_active=? WHERE chat_id=?",
            (1 if active else 0, chat_id),
        )
        conn.commit()


_TG_USER_FIELDS = {
    "username", "first_name", "language", "min_grade", "timeframes", "symbols",
    "quiet_start", "quiet_end", "daily_cap", "user_id", "is_active",
}


def tg_update_user(chat_id: int, **fields: Any) -> None:
    updates = {key: value for key, value in fields.items() if key in _TG_USER_FIELDS}
    if not updates:
        return
    assignments = ", ".join(f"{key}=?" for key in updates)
    with _lock:
        conn = get_conn()
        conn.execute(
            f"UPDATE telegram_users SET {assignments} WHERE chat_id=?",
            (*updates.values(), chat_id),
        )
        conn.commit()


def tg_get_user_by_account(user_id: int) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute(
            "SELECT * FROM telegram_users WHERE user_id=?", (user_id,),
        ).fetchone()
    return dict(row) if row else None


def tg_list_active_users() -> list[dict[str, Any]]:
    with _lock:
        rows = get_conn().execute(
            "SELECT * FROM telegram_users WHERE is_active=1 ORDER BY chat_id",
        ).fetchall()
    return [dict(row) for row in rows]


def tg_count_users(active_only: bool = False) -> int:
    with _lock:
        statement = "SELECT COUNT(*) c FROM telegram_users"
        if active_only:
            statement += " WHERE is_active=1"
        row = get_conn().execute(statement).fetchone()
    return int(row["c"])


def tg_create_link_code(code: str, user_id: int, ttl_seconds: int) -> None:
    expires = datetime.now(timezone.utc).timestamp() + ttl_seconds
    expires_at = datetime.fromtimestamp(expires, timezone.utc).isoformat()
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO telegram_link_codes (code, user_id, expires_at, used, created_at)
               VALUES (?,?,?,0,?)
               ON CONFLICT(code) DO UPDATE SET user_id=excluded.user_id,
                 expires_at=excluded.expires_at, used=0, created_at=excluded.created_at""",
            (code, user_id, expires_at, _now()),
        )
        conn.commit()


def tg_consume_link_code(code: str, chat_id: int) -> int | None:
    with _lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT user_id, expires_at, used FROM telegram_link_codes WHERE code=?", (code,),
        ).fetchone()
        if row is None or row["used"]:
            return None
        try:
            expiry = datetime.fromisoformat(str(row["expires_at"]))
        except ValueError:
            return None
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry <= datetime.now(timezone.utc):
            return None
        conn.execute(
            "UPDATE telegram_link_codes SET used=1, chat_id=? WHERE code=?", (chat_id, code),
        )
        conn.execute(
            "UPDATE telegram_users SET user_id=? WHERE chat_id=?", (row["user_id"], chat_id),
        )
        conn.commit()
    return int(row["user_id"])


def tg_record_delivery(chat_id: int, signal_id: int, kind: str = "signal") -> bool:
    with _lock:
        conn = get_conn()
        result = conn.execute(
            """INSERT INTO telegram_deliveries (chat_id, signal_id, kind, created_at)
               VALUES (?,?,?,?)
               ON CONFLICT(chat_id, signal_id, kind) DO NOTHING""",
            (chat_id, signal_id, kind, _now()),
        )
        conn.commit()
        return result.rowcount > 0


def tg_get_delivery(
    chat_id: int, signal_id: int, kind: str = "signal",
) -> dict[str, Any] | None:
    with _lock:
        row = get_conn().execute(
            """SELECT * FROM telegram_deliveries
               WHERE chat_id=? AND signal_id=? AND kind=?""",
            (chat_id, signal_id, kind),
        ).fetchone()
    return dict(row) if row else None


def tg_set_message_id(
    chat_id: int, signal_id: int, message_id: int, kind: str = "signal",
) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """UPDATE telegram_deliveries SET message_id=?
               WHERE chat_id=? AND signal_id=? AND kind=?""",
            (message_id, chat_id, signal_id, kind),
        )
        conn.commit()


def tg_delivery_count_today(chat_id: int) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    with _lock:
        row = get_conn().execute(
            """SELECT COUNT(*) c FROM telegram_deliveries
               WHERE chat_id=? AND substr(created_at,1,10)=?""",
            (chat_id, today),
        ).fetchone()
    return int(row["c"])


def tg_last_delivery_at() -> str | None:
    with _lock:
        row = get_conn().execute(
            "SELECT MAX(created_at) value FROM telegram_deliveries",
        ).fetchone()
    return row["value"] if row else None


# --- MT5 bridge storage ------------------------------------------------------
def bridge_insert_candles(symbol: str, timeframe: str, rows: list[dict[str, Any]]) -> int:
    """Idempotently store bridge candles. Returns the number of new rows."""
    if not rows:
        return 0
    inserted = 0
    with _lock:
        conn = get_conn()
        for row in rows:
            result = conn.execute(
                """INSERT INTO bridge_candles
                   (symbol, timeframe, broker_time, time, open, high, low, close,
                    volume, spread, real_volume, received_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol, timeframe, broker_time) DO NOTHING""",
                (
                    symbol, timeframe, int(row["broker_time"]), int(row["time"]),
                    float(row["open"]), float(row["high"]), float(row["low"]),
                    float(row["close"]), float(row.get("volume") or 0.0),
                    float(row.get("spread") or 0.0), row.get("real_volume"), _now(),
                ),
            )
            inserted += max(0, int(result.rowcount or 0))
        conn.commit()
    return inserted


def bridge_insert_tick(symbol: str, tick: dict[str, Any]) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO bridge_ticks
               (symbol, broker_time, time, bid, ask, spread, digits, point,
                contract_size, broker, server, received_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                symbol, int(tick["broker_time"]), int(tick["time"]),
                float(tick["bid"]), float(tick["ask"]), float(tick["spread"]),
                tick.get("digits"), tick.get("point"), tick.get("contract_size"),
                tick.get("broker"), tick.get("server"), _now(),
            ),
        )
        conn.commit()


def bridge_update_meta(symbol: str, meta: dict[str, Any]) -> None:
    with _lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO bridge_meta
               (symbol, broker, server, digits, point, contract_size,
                timezone_offset_seconds, currency, last_candle_at, last_tick_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol) DO UPDATE SET
                 broker=excluded.broker, server=excluded.server, digits=excluded.digits,
                 point=excluded.point, contract_size=excluded.contract_size,
                 timezone_offset_seconds=excluded.timezone_offset_seconds,
                 currency=excluded.currency,
                 last_candle_at=COALESCE(excluded.last_candle_at, bridge_meta.last_candle_at),
                 last_tick_at=COALESCE(excluded.last_tick_at, bridge_meta.last_tick_at),
                 updated_at=excluded.updated_at""",
            (
                symbol, meta.get("broker") or "", meta.get("server") or "",
                int(meta.get("digits") or 2), float(meta.get("point") or 0.01),
                float(meta.get("contract_size") or 100.0),
                int(meta.get("timezone_offset_seconds") or 0),
                meta.get("currency") or "USD", meta.get("last_candle_at"),
                meta.get("last_tick_at"), _now(),
            ),
        )
        conn.commit()


def bridge_get_meta(symbol: str | None = None) -> dict[str, Any] | None:
    with _lock:
        if symbol:
            row = get_conn().execute(
                "SELECT * FROM bridge_meta WHERE symbol=?", (symbol,),
            ).fetchone()
        else:
            row = get_conn().execute(
                "SELECT * FROM bridge_meta ORDER BY updated_at DESC LIMIT 1",
            ).fetchone()
    return dict(row) if row else None


def bridge_get_candles(
    symbol: str, timeframe: str, limit: int = 1500,
) -> list[dict[str, Any]]:
    with _lock:
        rows = get_conn().execute(
            """SELECT broker_time, time, open, high, low, close, volume, spread, real_volume
               FROM bridge_candles WHERE symbol=? AND timeframe=?
               ORDER BY broker_time DESC LIMIT ?""",
            (symbol, timeframe, max(1, min(int(limit), 10000))),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def bridge_get_latest_tick(symbol: str | None = None) -> dict[str, Any] | None:
    with _lock:
        if symbol:
            row = get_conn().execute(
                """SELECT * FROM bridge_ticks WHERE symbol=?
                   ORDER BY broker_time DESC, id DESC LIMIT 1""", (symbol,),
            ).fetchone()
        else:
            row = get_conn().execute(
                "SELECT * FROM bridge_ticks ORDER BY broker_time DESC, id DESC LIMIT 1",
            ).fetchone()
    return dict(row) if row else None


def bridge_latest_candle_age(symbol: str, timeframe: str) -> float | None:
    meta = bridge_get_meta(symbol)
    stamp = (meta or {}).get("last_candle_at")
    if not stamp:
        return None
    try:
        from datetime import datetime, timezone
        when = datetime.fromisoformat(str(stamp))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - when).total_seconds()
    except (ValueError, TypeError):
        return None
