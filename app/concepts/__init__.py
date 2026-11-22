"""Concept library orchestration: registry-driven, causal detection.

Detectors register themselves via :func:`app.concepts.base.register`. Importing
this package lazily loads every family so the registry is complete before
``detect_all`` runs.
"""
from __future__ import annotations

import importlib
import inspect
import logging
from contextlib import nullcontext as _nullcontext
from typing import Any

import pandas as pd

from app.concepts.base import (
    DETECTORS,
    DETECTOR_CATEGORIES,
    Concept,
    ensure_frame,
    serialise,
    session_of,
    unlimited_mode,
)
from app.concepts.registry import allows as registry_allows, infer_timeframe as infer_tf

logger = logging.getLogger(__name__)

FAMILIES = ("structure", "smc", "classic", "price_action", "mtf", "macro", "advanced")
_loaded = False


def load_families(force: bool = False) -> None:
    """Import detector modules once so their registrations are available."""
    global _loaded
    if _loaded and not force:
        return
    for name in FAMILIES:
        try:
            importlib.import_module(f"app.concepts.{name}")
        except ImportError as exc:  # pragma: no cover - only during partial builds
            logger.warning("concept family %s unavailable: %s", name, exc)
    _loaded = True


def detect_all(df: pd.DataFrame, context: dict[str, Any] | None = None,
               families: tuple[str, ...] | None = None,
               types: tuple[str, ...] | None = None,
               unlimited: bool = False,
               timeframe: str | None = None) -> list[Concept]:
    """Run every registered detector and return a flat concept list.

    ``context`` carries optional extra inputs (other timeframes, macro frames,
    news sentiment, event calendar). Each detector receives only the kwargs it
    declares, so adding context is always safe.
    """
    load_families()
    frame = ensure_frame(df)
    ctx = context or {}
    tf = timeframe or ctx.get("timeframe") or infer_tf(frame)
    regime = ctx.get("regime")
    session = ctx.get("session")
    if session is None and len(frame.index):
        session = session_of(frame.index[-1])
    found: list[Concept] = []
    with unlimited_mode() if unlimited else _nullcontext():
        for name, fn in DETECTORS.items():
            category = DETECTOR_CATEGORIES.get(name, "classic")
            if families and category not in families:
                continue
            if types and name not in types:
                continue
            if not registry_allows(name, tf, regime, session):
                continue
            try:
                params = inspect.signature(fn).parameters
                kwargs = {k: ctx[k] for k in ctx if k in params}
                out = fn(frame, **kwargs) or []
                for concept in out:
                    if isinstance(concept, dict):
                        concept = Concept(**concept)
                    if concept.category == "classic":
                        concept.category = category
                    found.append(concept)
            except Exception as exc:  # noqa: BLE001 - one bad detector cannot break the app
                logger.warning("concept detector %s failed: %s", name, exc, exc_info=True)
    found.sort(key=lambda c: (str(c.start_time), c.type))
    return found


def detect_all_dicts(df: pd.DataFrame, context: dict[str, Any] | None = None,
                     **kwargs: Any) -> list[dict[str, Any]]:
    """Serialisable form for the API / chat / ML features."""
    return serialise(detect_all(df, context, **kwargs))


__all__ = [
    "Concept", "DETECTORS", "FAMILIES", "detect_all", "detect_all_dicts",
    "load_families",
]
