"""Concept plugin registry.

One plugin interface for every concept: identity, metadata, causal detection,
ML features and human-readable explanation. The registry is declarative
(``concepts.yaml``) so new concepts can be added without touching the engine.

Existing detectors register themselves through :func:`app.concepts.base.register`;
this module wraps them in :class:`ConceptSpec` and applies the declarative
enabled/weight/min-timeframe/regime/session rules in ``detect_all``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from app.concepts.base import DETECTORS, DETECTOR_CATEGORIES, session_of

REGISTRY_PATH = Path(__file__).resolve().parent / "concepts.yaml"

TF_RANK = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1D": 6}


def timeframe_rank(value: str | None) -> int:
    return TF_RANK.get(str(value or "").strip(), 2)


def infer_timeframe(frame: pd.DataFrame) -> str | None:
    """Best-effort timeframe from a candle index (used when no context is given)."""
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame.index) < 3:
        return None
    seconds = pd.Series(frame.index).diff().dropna().dt.total_seconds()
    if seconds.empty:
        return None
    step = float(seconds.median())
    if step <= 90:
        return "1m"
    if step <= 330:
        return "5m"
    if step <= 1000:
        return "15m"
    if step <= 2100:
        return "30m"
    if step <= 5400:
        return "1h"
    if step <= 18000:
        return "4h"
    return "1D"


@dataclass
class ConceptSpec:
    """Plugin contract for one concept."""

    id: str
    family: str = "classic"
    description: str = ""
    required_data: tuple[str, ...] = ()
    enabled: bool = True
    weight: float = 0.25
    min_timeframe: str = "15m"
    regimes: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    detect: Callable[..., list[Any]] | None = None
    features: Callable[..., pd.DataFrame] | None = None
    explain: Callable[..., str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def allows(self, timeframe: str | None = None, regime: str | None = None,
               session: str | None = None) -> bool:
        if not self.enabled:
            return False
        if timeframe and timeframe_rank(timeframe) < timeframe_rank(self.min_timeframe):
            return False
        if self.regimes and regime and regime not in self.regimes:
            return False
        if self.sessions and session and session not in self.sessions:
            return False
        return True

    def explain_text(self) -> str:
        if self.explain is not None:
            try:
                return str(self.explain())
            except Exception:  # noqa: BLE001 - explanation must never break a signal
                pass
        return self.description or self.id.replace("_", " ").capitalize()

    def detections(self, frame: pd.DataFrame, context: dict[str, Any]) -> list[Any]:
        fn = self.detect
        if fn is None:
            return []
        try:
            import inspect
            params = inspect.signature(fn).parameters
            kwargs = {k: context[k] for k in context if k in params}
            return list(fn(frame, **kwargs) or [])
        except Exception:  # noqa: BLE001 - one broken concept cannot break the app
            return []

    def feature_frame(self, frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
        if self.features is not None:
            try:
                return self.features(frame, context)
            except Exception:  # noqa: BLE001
                return pd.DataFrame(index=frame.index)
        out = pd.DataFrame(index=frame.index)
        if not self.enabled:
            return out
        for i, concept in enumerate(self.detections(frame, context)):
            direction = getattr(concept, "direction", "neutral")
            strength = float(getattr(concept, "strength", 0.0) or 0.0)
            prefix = f"{self.id}_{i}"
            out[f"{prefix}_dir"] = direction
            out[f"{prefix}_strength"] = strength
            end = getattr(concept, "end_time", None)
            if end is not None and isinstance(frame.index, pd.DatetimeIndex):
                try:
                    out[f"{prefix}_age"] = (frame.index - pd.Timestamp(end)).days
                except (TypeError, ValueError):
                    pass
        return out


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


@lru_cache(maxsize=1)
def load_registry() -> dict[str, ConceptSpec]:
    """Load declarative metadata and bind every registered detector."""
    payload: dict[str, Any] = {}
    try:
        payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - defaults always work
        import logging
        logging.getLogger(__name__).warning("concepts.yaml unavailable: %s", exc)
    entries = payload.get("concepts") if isinstance(payload, dict) else None
    entries = entries if isinstance(entries, dict) else {}

    specs: dict[str, ConceptSpec] = {}
    for name, fn in DETECTORS.items():
        raw = entries.get(name) or {}
        raw = raw if isinstance(raw, dict) else {}
        specs[name] = ConceptSpec(
            id=name,
            family=str(raw.get("family") or DETECTOR_CATEGORIES.get(name, "classic")),
            description=str(raw.get("description") or (fn.__doc__ or "").strip().split("\n")[0]),
            required_data=_as_tuple(raw.get("required_data")),
            enabled=bool(raw.get("enabled", True)),
            weight=float(raw.get("weight", 0.25)),
            min_timeframe=str(raw.get("min_timeframe", "15m")),
            regimes=_as_tuple(raw.get("regimes")),
            sessions=_as_tuple(raw.get("sessions")),
            detect=fn,
            meta=raw,
        )
    for name, raw in entries.items():
        if name in specs or not isinstance(raw, dict):
            continue
        specs[name] = ConceptSpec(
            id=name,
            family=str(raw.get("family") or "classic"),
            description=str(raw.get("description") or ""),
            required_data=_as_tuple(raw.get("required_data")),
            enabled=bool(raw.get("enabled", False)),
            weight=float(raw.get("weight", 0.0)),
            min_timeframe=str(raw.get("min_timeframe", "15m")),
            regimes=_as_tuple(raw.get("regimes")),
            sessions=_as_tuple(raw.get("sessions")),
            meta=raw,
        )
    return specs


def reload_registry() -> dict[str, ConceptSpec]:
    load_registry.cache_clear()
    return load_registry()


def get_spec(name: str) -> ConceptSpec | None:
    return load_registry().get(name)


def allows(name: str, timeframe: str | None = None, regime: str | None = None,
           session: str | None = None) -> bool:
    spec = get_spec(name)
    return spec.allows(timeframe, regime, session) if spec else True


def enabled_specs() -> list[ConceptSpec]:
    return [spec for spec in load_registry().values() if spec.enabled]


def spec_dicts() -> list[dict[str, Any]]:
    return [{
        "id": spec.id, "family": spec.family, "description": spec.description,
        "required_data": list(spec.required_data), "enabled": spec.enabled,
        "weight": spec.weight, "min_timeframe": spec.min_timeframe,
        "regimes": list(spec.regimes), "sessions": list(spec.sessions),
    } for spec in load_registry().values()]
