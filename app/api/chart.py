"""Chart drawing endpoint: the frontend renders this payload verbatim."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app import config
from app.services import data
from app.services.chart_annotations import build_payload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chart", tags=["chart"])


@router.get("/annotations")
def chart_annotations(
    timeframe: str = Query("1h", pattern="^(15m|1h|4h|1D)$"),
    show_all: bool = False,
    types: str | None = None,
    category: str | None = None,
    mode: str = Query("clean", pattern="^(clean|smart_money|classic|full)$"),
) -> dict[str, Any]:
    """Ranked, decluttered drawings for one timeframe and chart mode."""
    try:
        entry = data.get_ohlcv(config.GOLD_TICKER, timeframe)
        if entry.df is None or entry.df.empty:
            raise HTTPException(503, entry.error or "no live XAU/USD data available")
        return build_payload(entry, timeframe, show_all=show_all,
                             types=types, category=category, mode=mode)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("chart annotation build failed")
        raise HTTPException(500, f"chart annotation build failed: {exc}") from exc
