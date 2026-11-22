"""Weighted, category-capped confluence engine.

Turns detected concepts (plus the XGBoost probability) into a graded signal
with veto rules, entry/SL/TP levels and a list of agreeing/disagreeing factors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.services.signal_quality import check_extended_move

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "concept_weights.yaml"

DEFAULT_SETTINGS: dict[str, Any] = {
    "recency_bars": 24, "min_recency": 0.2,
    "score_buy_threshold": 0.18, "score_sell_threshold": -0.18,
    "min_grade_a_categories": 3, "min_aplus_categories": 4,
    "min_rr": 1.5, "choppy_regimes": ["ranging", "volatile", "neutral"],
    "max_event_risk": 70, "event_block_hours": 12.0, "max_atr_pct": 3.5,
    "max_spread_atr_ratio": 0.25, "ml_blend": 0.35,
    "extended_move_atr": 2.0, "rsi_oversold": 32.0,
    "rsi_overbought": 75.0, "near_zone_atr": 0.75,
}
DEFAULT_CAPS: dict[str, float] = {
    "structure": 3.2, "smc": 3.6, "classic": 2.4,
    "price_action": 1.8, "mtf": 3.0, "macro": 2.2, "advanced": 1.4,
}


@lru_cache(maxsize=1)
def load_weights() -> dict[str, Any]:
    """Load concept_weights.yaml (PyYAML optional; defaults are always usable)."""
    result: dict[str, Any] = {
        "settings": dict(DEFAULT_SETTINGS),
        "category_caps": dict(DEFAULT_CAPS),
        "weights": {},
    }
    try:
        import yaml
        payload = yaml.safe_load(WEIGHTS_PATH.read_text(encoding="utf-8")) or {}
        for key in ("settings", "category_caps", "weights"):
            if isinstance(payload.get(key), dict):
                result[key].update(payload[key])
    except Exception as exc:  # noqa: BLE001
        logger.warning("using default concept weights: %s", exc)
    return result


def reload_weights() -> dict[str, Any]:
    load_weights.cache_clear()
    return load_weights()


LIQUIDITY_TYPES = {
    "equal_highs", "equal_lows", "pdh", "pdl", "pwh", "pwl",
    "session_high", "session_low", "htf_level", "fvg", "order_block",
    "ifvg", "breaker_block", "ote", "fibonacci", "pivot_points",
    "supply_demand", "sr_zone", "dealing_range",
}
ENTRY_TYPES = {
    "fvg", "ifvg", "order_block", "breaker_block", "mitigation_block",
    "ote", "supply_demand",
}
SWING_TYPES = {"swing_high", "swing_low"}


def _ts(value: Any) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(t):
        return None
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _bar_age(concept: Any, index: pd.DatetimeIndex, last: int) -> int:
    t = _ts(concept.end_time)
    if t is None:
        return 0
    if index.tz is not None:
        t = t.tz_localize("UTC")
    pos = int(index.searchsorted(t, side="right"))
    return max(0, last - pos + 1)


def _latest(concepts: list[Any], types: set[str], limit: int | None = None) -> list[Any]:
    found = sorted((c for c in concepts if c.type in types),
                   key=lambda c: str(c.end_time or ""))
    return found[-limit:] if limit else found


@dataclass
class ConfluenceResult:
    action: str
    grade: str
    score: float
    confidence: float
    categories: dict[str, float]
    factors: list[dict[str, Any]]
    agreeing: list[str]
    disagreeing: list[str]
    veto_reasons: list[str]
    entry_zone: tuple[float, float] | None
    stop_loss: float | None
    take_profit1: float | None
    take_profit2: float | None
    take_profit3: float | None
    risk_reward: float
    invalidation: str
    regime: str
    risk_reward_tp1: float = 0.0
    risk_reward_tp2: float = 0.0
    ml_probability: float | None = None
    killzone_active: bool = False
    notes: list[str] = field(default_factory=list)
    quality_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action, "grade": self.grade,
            "score": round(self.score, 4), "confidence": round(self.confidence, 1),
            "categories": {k: round(v, 3) for k, v in self.categories.items()},
            "factors": self.factors, "agreeing": self.agreeing,
            "disagreeing": self.disagreeing, "veto_reasons": self.veto_reasons,
            "entry_zone": list(self.entry_zone) if self.entry_zone else None,
            "stop_loss": self.stop_loss, "take_profit1": self.take_profit1,
            "take_profit2": self.take_profit2, "take_profit3": self.take_profit3,
            "risk_reward": round(self.risk_reward, 2),
            "risk_reward_tp1": round(self.risk_reward_tp1, 2),
            "risk_reward_tp2": round(self.risk_reward_tp2, 2),
            "invalidation": self.invalidation, "regime": self.regime,
            "ml_probability": self.ml_probability,
            "killzone_active": self.killzone_active, "notes": self.notes,
            "quality_check": self.quality_check,
        }


def _zone_for(concepts: list[Any], direction: str, price: float,
              atr_value: float) -> tuple[float, float] | None:
    candidates = [
        c for c in _latest(concepts, ENTRY_TYPES)
        if c.direction == direction and c.top is not None and c.bottom is not None
        and c.status != "invalidated"
        and abs(((c.top + c.bottom) / 2.0) - price) <= 5.0 * max(atr_value, 1e-6)
    ]
    if not candidates:
        return None
    best = candidates[-1]
    return max(float(best.top), float(best.bottom)), min(float(best.top), float(best.bottom))


def _targets(concepts: list[Any], action: str, entry: float, risk: float,
             atr_value: float) -> list[float]:
    side_ok = (lambda v: v > entry) if action == "BUY" else (lambda v: v < entry)
    pool: list[float] = []
    for c in _latest(concepts, LIQUIDITY_TYPES):
        for key in ("top", "bottom"):
            value = getattr(c, key, None)
            if value is not None and side_ok(float(value)):
                pool.append(float(value))
        target = (c.meta or {}).get("target")
        if isinstance(target, (int, float)) and side_ok(float(target)):
            pool.append(float(target))
    pool.extend(
        [float(x) for x in np.linspace(entry + 1.0 * atr_value,
                                       entry + 6.0 * atr_value, 8)]
        if action == "BUY" else
        [float(x) for x in np.linspace(entry - 1.0 * atr_value,
                                       entry - 6.0 * atr_value, 8)]
    )
    clean = sorted({round(v, 2) for v in pool if np.isfinite(v)},
                   reverse=(action == "SELL"))
    return [v for v in clean if abs(v - entry) >= 1.5 * max(risk, 1e-6)]


def confluence(concepts: list[Any], df: pd.DataFrame,
               ml_probability: float | None = None,
               spread_usd: float | None = None) -> ConfluenceResult:
    """Score concepts, enforce veto rules and derive a full trade plan."""
    cfg = load_weights()
    settings = cfg["settings"]
    caps = cfg["category_caps"]
    weights = cfg["weights"]
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    last = len(index) - 1
    price = float(df["Close"].iloc[-1])
    atr_value = float(np.nan_to_num(
        pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    )) or price * 0.01

    factors: list[dict[str, Any]] = []
    categories: dict[str, float] = {}
    killzone_active = False
    event_risk: Any | None = None
    regime = "unknown"
    adx_value: float | None = None

    by_type: dict[str, list[Any]] = {}
    for c in concepts:
        by_type.setdefault(c.type, []).append(c)

    for type_, group in by_type.items():
        if type_ == "killzone":
            killzone_active = any(
                _bar_age(c, index, last) <= 2 for c in group
            )
            continue
        if type_ == "event_risk":
            recent = sorted(group, key=lambda c: str(c.end_time or ""))
            event_risk = recent[-1] if recent else None
            continue
        if type_ == "market_regime":
            recent = sorted(group, key=lambda c: str(c.end_time or ""))
            if recent:
                regime = str((recent[-1].meta or {}).get("regime", "unknown"))
                raw_adx = (recent[-1].meta or {}).get("adx")
                if isinstance(raw_adx, (int, float)):
                    adx_value = float(raw_adx)
        weight = float(weights.get(type_, 0.25))
        if weight <= 0:
            continue
        for c in sorted(group, key=lambda c: str(c.end_time or ""))[-2:]:
            if c.status == "invalidated" or c.direction == "neutral":
                continue
            age = _bar_age(c, index, last)
            decay = max(float(settings["min_recency"]),
                        0.5 ** (age / max(float(settings["recency_bars"]), 1.0)))
            sign = 1.0 if c.direction == "bullish" else -1.0
            contribution = sign * weight * (float(c.strength) / 100.0) * decay
            categories[c.category] = categories.get(c.category, 0.0) + contribution
            factors.append({
                "type": c.type, "category": c.category,
                "direction": c.direction, "weight": round(weight, 3),
                "strength": round(float(c.strength), 1), "age": age,
                "contribution": round(contribution, 3),
                "notes": c.notes, "status": c.status,
            })

    for key in categories:
        categories[key] = float(np.clip(categories[key], -caps.get(key, 1.0),
                                        caps.get(key, 1.0)))
    total_cap = sum(caps.get(k, 1.0) for k in categories) or 1.0
    score = float(np.clip(sum(categories.values()) / total_cap, -1.0, 1.0))
    if killzone_active:
        score = float(np.clip(score * 1.08, -1.0, 1.0))

    ml_score = None
    if ml_probability is not None:
        ml_score = (float(ml_probability) - 0.5) * 2.0
        blend = float(settings["ml_blend"])
        score = float(np.clip((1.0 - blend) * score + blend * ml_score, -1.0, 1.0))

    location_penalties: list[str] = []
    location_concepts = by_type.get("premium_discount") or []
    if location_concepts:
        location = location_concepts[-1]
        at_premium = location.direction == "bearish"
        at_discount = location.direction == "bullish"
        if score > 0 and at_premium:
            score = float(np.clip(score - 0.15, -1.0, 1.0))
            location_penalties.append("BUY at premium penalised: wait for discount")
        elif score < 0 and at_discount:
            score = float(np.clip(score + 0.15, -1.0, 1.0))
            location_penalties.append("SELL at discount penalised: wait for premium")

    action = ("BUY" if score >= float(settings["score_buy_threshold"]) else
              "SELL" if score <= float(settings["score_sell_threshold"]) else "HOLD")

    direction = 1 if action == "BUY" else -1 if action == "SELL" else 0
    agreeing = sorted({f["type"] for f in factors
                       if direction != 0 and np.sign(f["contribution"]) == direction
                       and abs(f["contribution"]) > 0.05})
    disagreeing = sorted({f["type"] for f in factors
                          if direction != 0 and np.sign(f["contribution"]) == -direction
                          and abs(f["contribution"]) > 0.05})
    agreed_categories = sum(
        1 for value in categories.values()
        if direction != 0 and np.sign(value) == direction and abs(value) > 0.15
    )

    grade = "C"
    if abs(score) >= 0.60 and agreed_categories >= int(settings["min_aplus_categories"]):
        grade = "A+"
    elif abs(score) >= 0.45 and agreed_categories >= int(settings["min_grade_a_categories"]):
        grade = "A"
    elif abs(score) >= 0.30 and agreed_categories >= 2:
        grade = "B"

    expected_direction = "bullish" if action == "BUY" else "bearish" if action == "SELL" else None
    if expected_direction:
        conflicts = [
            f for f in factors
            if f.get("direction") not in (None, "neutral", expected_direction)
            and abs(float(f.get("contribution", 0))) > 0.05
        ]
        if conflicts:
            grade = {"A+": "A", "A": "B", "B": "C"}.get(grade, grade)
            location_penalties.append(
                "direction conflict: " + ", ".join(str(f.get("type")) for f in conflicts[:4])
            )

    zone = _zone_for(concepts, "bullish" if action == "BUY" else "bearish",
                     price, atr_value) if action != "HOLD" else None
    if zone:
        entry = (zone[0] + zone[1]) / 2.0
    else:
        entry = price
        if action != "HOLD":
            zone = (price + 0.3 * atr_value, price - 0.3 * atr_value)

    sl = tp1 = tp2 = tp3 = None
    risk = 0.0
    rr = 0.0
    rr1 = 0.0
    rr2 = 0.0
    if action != "HOLD":
        buffer = 0.5 * atr_value
        recent_swings = _latest(concepts, SWING_TYPES, limit=8)
        nearby = [
            float(c.top if c.top is not None else c.bottom)
            for c in recent_swings
            if (c.top is not None or c.bottom is not None)
            and abs(float(c.top if c.top is not None else c.bottom) - price)
            <= 5.0 * max(atr_value, 1e-6)
        ]
        if action == "BUY":
            swing_low = min(nearby, default=entry)
            sl = min(zone[1] if zone else entry - 0.8 * atr_value,
                     swing_low) - buffer
        else:
            swing_high = max(nearby, default=entry)
            sl = max(zone[0] if zone else entry + 0.8 * atr_value,
                     swing_high) + buffer
        risk = abs(entry - sl)
        if risk <= 0:
            risk = max(atr_value, 1e-6)
            sl = entry - risk if action == "BUY" else entry + risk
        targets = _targets(concepts, action, entry, risk, atr_value)
        if len(targets) >= 1:
            tp1 = targets[0]
        if len(targets) >= 2:
            tp2 = targets[1]
        if len(targets) >= 3:
            tp3 = targets[2]
        from app.services.risk_reward import risk_reward_ratios
        ratios = risk_reward_ratios(entry, sl, tp1, tp2)
        rr1, rr2 = ratios["risk_reward_tp1"], ratios["risk_reward_tp2"]
        rr = rr2 if rr2 > 0 else rr1

    veto: list[str] = []
    if regime in set(settings["choppy_regimes"]):
        veto.append(f"market regime is {regime}")
    if adx_value is not None and adx_value < 20:
        veto.append(f"ADX {adx_value:.1f} is too low (ranging market)")
    if event_risk is not None:
        meta = event_risk.meta or {}
        hours_away = meta.get("hours_away")
        too_close = isinstance(hours_away, (int, float)) and float(hours_away) <= float(settings["event_block_hours"])
        if float(event_risk.strength) >= float(settings["max_event_risk"]) or too_close:
            veto.append(f"high-impact event near: {event_risk.notes}")
    atr_pct = (atr_value / max(price, 1e-9)) * 100.0
    if atr_pct > float(settings["max_atr_pct"]):
        veto.append(f"ATR {atr_pct:.2f}% exceeds {settings['max_atr_pct']}%")
    spread = float(spread_usd if spread_usd is not None else config.SPREAD_USD)
    if atr_value > 0 and spread / atr_value > float(settings["max_spread_atr_ratio"]):
        veto.append("spread is extreme versus ATR")
    lean = np.sign(score)
    for c in by_type.get("htf_veto", []):
        if (c.direction == "bearish" and lean > 0) or (c.direction == "bullish" and lean < 0):
            veto.append(f"higher-timeframe veto: {c.notes}")
            break
    if action != "HOLD" and rr < float(settings["min_rr"]):
        veto.append(f"risk:reward {rr:.2f} below {settings['min_rr']}:1")
    if action != "HOLD" and grade in ("A", "A+") and len(agreeing) < int(settings["min_grade_a_categories"]):
        grade = "B"

    quality = None
    if action in ("BUY", "SELL"):
        quality = check_extended_move(
            concepts, df, action,
            extended_atr=float(settings["extended_move_atr"]),
            rsi_oversold=float(settings["rsi_oversold"]),
            rsi_overbought=float(settings["rsi_overbought"]),
            near_zone_atr=float(settings["near_zone_atr"]),
        )
        if quality.blocked:
            veto.extend(quality.reasons)

    final_action = "HOLD" if veto else action
    if final_action == "HOLD":
        grade = "C" if not veto else "—"

    base_conf = min(abs(score) / 0.75, 1.0)
    agreement = len(agreeing) / max(len(agreeing) + len(disagreeing), 1)
    confidence = float(np.clip((0.6 * base_conf + 0.4 * agreement) * 100.0, 5.0, 97.0))
    if final_action == "HOLD":
        confidence = min(confidence, 45.0)

    invalidation = (
        f"A close below {sl:.2f} (stop beyond the entry zone plus ATR buffer) "
        "or a bearish CHoCH on the lower timeframe invalidates this long."
        if final_action == "BUY" and sl is not None else
        f"A close above {sl:.2f} (stop beyond the entry zone plus ATR buffer) "
        "or a bullish CHoCH on the lower timeframe invalidates this short."
        if final_action == "SELL" and sl is not None else
        "No active setup: invalidation is not applicable."
    )

    notes: list[str] = []
    if killzone_active:
        notes.append("Inside a killzone — score boosted 8%.")
    if ml_probability is not None:
        notes.append(f"XGBoost P(up) {ml_probability:.0%} blended at "
                     f"{float(settings['ml_blend']):.0%}.")
    notes.extend(location_penalties)
    notes.extend(veto)

    return ConfluenceResult(
        action=final_action, grade=grade, score=score, confidence=confidence,
        categories=categories, factors=factors, agreeing=agreeing,
        disagreeing=disagreeing, veto_reasons=veto, entry_zone=zone,
        stop_loss=sl, take_profit1=tp1, take_profit2=tp2, take_profit3=tp3,
        risk_reward=rr, invalidation=invalidation, regime=regime,
        risk_reward_tp1=rr1, risk_reward_tp2=rr2,
        ml_probability=ml_probability, killzone_active=killzone_active,
        notes=notes, quality_check=quality.to_dict() if quality else None,
    )
