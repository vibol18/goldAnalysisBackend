"""Create MT5 bridge tables (added after 0001)."""
from __future__ import annotations

from alembic import op

from app.services.db import metadata

revision = "0003_bridge_tables"
down_revision = "0002_telegram"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    for name in ("bridge_candles", "bridge_ticks", "bridge_meta"):
        metadata.tables[name].create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for name in ("bridge_ticks", "bridge_candles", "bridge_meta"):
        metadata.tables[name].drop(bind=bind, checkfirst=True)
