"""Create Telegram subscriber/delivery tables.

Revision ID: 0002_telegram
Revises: 0001_portable_persistence
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

from app.services.db import metadata

revision = "0002_telegram"
down_revision = "0001_portable_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind=bind)
    table_names = set(inspect(bind).get_table_names())
    if "telegram_deliveries" in table_names:
        indexes = {index["name"] for index in inspect(bind).get_indexes("telegram_deliveries")}
        uniques = {
            constraint.get("name")
            for constraint in inspect(bind).get_unique_constraints("telegram_deliveries")
        }
        if "uq_tg_delivery" not in indexes and "uq_tg_delivery" not in uniques:
            op.create_index(
                "uq_tg_delivery", "telegram_deliveries",
                ["chat_id", "signal_id", "kind"], unique=True,
            )


def downgrade() -> None:
    pass
