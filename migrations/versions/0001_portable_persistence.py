from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect, text

from app.services.db import metadata

revision = "0001_portable_persistence"
down_revision = None
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def _ensure_column(table_name: str, column: sa.Column) -> None:
    if table_name in inspect(op.get_bind()).get_table_names() and column.name not in _columns(table_name):
        op.add_column(table_name, column)


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind=bind)

    _ensure_column(
        "signals", sa.Column("symbol", sa.String(32), nullable=False, server_default="XAU/USD"),
    )
    for column in (
        sa.Column("signal_time", sa.String(40), nullable=True),
        sa.Column("grade", sa.String(8), nullable=True),
        sa.Column("source", sa.String(32), nullable=False, server_default="live"),
        sa.Column("setup", sa.String(120), nullable=True),
        sa.Column("session", sa.String(32), nullable=True),
    ):
        _ensure_column("signals", column)
    for table_name in ("signals", "chat_messages", "journal_entries", "alerts"):
        _ensure_column(table_name, sa.Column("user_id", sa.Integer(), nullable=True))

    table_names = set(inspect(bind).get_table_names())
    if "alert_settings" in table_names:
        legacy = bind.execute(
            text("SELECT payload FROM alert_settings WHERE id = :id"), {"id": 1},
        ).first()
        if legacy:
            bind.execute(
                text(
                    """INSERT INTO user_alert_settings (user_id, payload)
                       VALUES (:user_id, :payload)
                       ON CONFLICT (user_id) DO NOTHING"""
                ),
                {"user_id": 0, "payload": legacy[0]},
            )

    inspector = inspect(bind)
    if "signals" in inspector.get_table_names():
        existing_indexes = {index["name"] for index in inspector.get_indexes("signals")}
        existing_constraints = {
            constraint.get("name")
            for constraint in inspector.get_unique_constraints("signals")
        }
        duplicate = bind.execute(
            text(
                """SELECT 1 FROM signals
                   WHERE signal_time IS NOT NULL
                   GROUP BY symbol, timeframe, signal_time
                   HAVING COUNT(*) > 1 LIMIT 1"""
            )
        ).first()
        if (
            not duplicate
            and "uq_signal_candle" not in existing_indexes
            and "uq_signal_candle" not in existing_constraints
        ):
            op.create_index(
                "uq_signal_candle", "signals",
                ["symbol", "timeframe", "signal_time"], unique=True,
            )
        elif duplicate:
            logger.warning(
                "Existing duplicate signal candles prevent creating the idempotency index"
            )


def downgrade() -> None:
    pass
