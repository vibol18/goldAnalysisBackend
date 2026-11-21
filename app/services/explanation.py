"""Plain-language, analyst-style explanation of a concept-driven signal."""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.confluence import ConfluenceResult


def money(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.2f}"


def _latest(concepts: list[Any], types: set[str], direction: str | None = None):
    found = [c for c in concepts if c.type in types and c.status != "invalidated"
             and (direction is None or c.direction == direction)]
    return found[-1] if found else None


def build_explanation(concepts: list[Any], result: ConfluenceResult,
                      df: pd.DataFrame, timeframe: str,
                      rule_score: float = 0.0,
                      ml_probability: float | None = None,
                      rule_components: list[dict[str, Any]] | None = None,
                      ) -> dict[str, Any]:
    """Compose the narrative, bullet reasons and invalidation for a signal."""
    price = float(df["Close"].iloc[-1])
    action = result.action
    bias_word = {"BUY": "bullish", "SELL": "bearish"}.get(action, "neutral")

    trend = _latest(concepts, {"trend", "mtf_bias", "macro_regime"})
    bias_bits: list[str] = []
    if trend is not None:
        bias_bits.append(f"Bias: {bias_word} ({trend.notes.strip().rstrip('.')})")
    else:
        bias_bits.append(f"Bias: {bias_word} (no dominant higher-timeframe trend)")
    if result.regime != "unknown":
        bias_bits[-1] += f", regime {result.regime}"

    location_bits: list[str] = []
    pd_concept = _latest(concepts, {"premium_discount"})
    if pd_concept is not None:
        pos = (pd_concept.meta or {}).get("position_pct")
        zone_word = "discount" if pd_concept.direction == "bullish" else "premium"
        location_bits.append(
            f"Price is in {zone_word} ({float(pos):.0f}% of the range)"
            if pos is not None else f"Price is in {zone_word}"
        )
    ote = _latest(concepts, {"ote"})
    if ote is not None and ote.top is not None and ote.bottom is not None:
        location_bits.append(
            f"OTE zone {money(ote.bottom)}–{money(ote.top)}"
        )

    event_bits: list[str] = []
    sweep = _latest(concepts, {"liquidity_sweep"})
    if sweep is not None:
        swept = (sweep.meta or {}).get("swept_price")
        if swept is not None:
            side = "sell-side" if sweep.direction == "bullish" else "buy-side"
            event_bits.append(
                f"{side.title()} liquidity at {money(float(swept))} was swept, then price closed back inside"
            )
        else:
            event_bits.append(sweep.notes.strip().rstrip("."))

    structure = _latest(concepts, {"choch", "mss", "bos"}) if action != "HOLD" else None
    if structure is not None:
        extra = " with displacement" if structure.type in ("mss", "choch") and (
            _latest(concepts, {"displacement"}) is not None) else ""
        event_bits.append(
            f"a {structure.direction} {structure.type.upper()}{extra} confirmed the shift"
        )

    zone = _latest(concepts, {"fvg", "order_block", "ifvg", "breaker_block"},
                   "bullish" if action == "BUY" else "bearish" if action == "SELL" else None)
    if zone is not None and zone.top is not None and zone.bottom is not None:
        event_bits.append(
            f"a {zone.direction} {zone.type.replace('_', ' ')} at "
            f"{money(zone.bottom)}–{money(zone.top)} aligns with the setup"
        )

    killzone = _latest(concepts, {"killzone"})
    if killzone is not None and result.killzone_active:
        event_bits.append(f"inside the {killzone.meta.get('zone', 'active')} killzone")

    sentences = bias_bits + location_bits
    if event_bits:
        joined = event_bits[0]
        for part in event_bits[1:]:
            joined += f", then {part}" if not part.startswith("a ") else f", {part}"
        sentences.append(joined.capitalize())

    if result.action != "HOLD" and result.stop_loss is not None:
        levels = (f"Entry {money(result.entry_zone[0] if result.entry_zone else price)}, "
                  f"SL {money(result.stop_loss)}, TP1 {money(result.take_profit1)}, "
                  f"TP2 {money(result.take_profit2)}")
        if result.take_profit3 is not None:
            levels += f", TP3 {money(result.take_profit3)}"
        sentences.append(levels + ".")
    else:
        sentences.append("No tradeable setup: staying in HOLD.")

    total_factors = len(result.agreeing) + len(result.disagreeing)
    sentences.append(
        f"Confluence: {len(result.agreeing)} of {total_factors} factors agree. Grade {result.grade}."
    )
    if result.disagreeing:
        sentences.append(f"Against: {', '.join(result.disagreeing[:4])}.")
    if result.veto_reasons:
        sentences.append("HOLD because " + "; ".join(result.veto_reasons) + ".")
    sentences.append(result.invalidation)

    narrative = ". ".join(s.strip().rstrip(".") for s in sentences if s.strip()) + "."

    reasons: list[str] = []
    for key, label in (
        ("trend", "Higher-timeframe bias"), ("premium_discount", "Location"),
        ("liquidity_sweep", "Liquidity"), ("choch", "Structure"),
        ("fvg", "Entry zone"), ("order_block", "Entry zone"),
        ("killzone", "Session"),
    ):
        item = _latest(concepts, {key})
        if item is not None:
            reasons.append(f"{label}: {item.notes.strip().rstrip('.')}.")
    for veto in result.veto_reasons:
        reasons.append(f"Veto: {veto}.")
    if not reasons:
        reasons = [sentences[0]]

    checklist = [
        {
            "type": f["type"],
            "label": f["type"].replace("_", " ").title(),
            "agree": f["contribution"] >= 0,
            "weight": f["weight"],
            "detail": f["notes"],
            "category": f["category"],
        }
        for f in result.factors
    ]
    why_not_trading = None
    if result.action == "HOLD":
        why_not_trading = (
            "No trade taken: " + ("; ".join(result.veto_reasons) if result.veto_reasons
                                  else "confluence is below the minimum score and no edge is confirmed")
            + f". Grade {result.grade}, confluence score {result.score:+.2f}. "
              "Wait for a closed candle that satisfies every rule."
        )

    return {
        "summary": narrative,
        "narrative": narrative,
        "reasons": reasons[:8],
        "components": rule_components or [],
        "factors": result.factors,
        "rule_score": round(rule_score, 3),
        "ml_probability": ml_probability,
        "timeframe": timeframe,
        "grade": result.grade,
        "confluence": result.to_dict(),
        "invalidation": result.invalidation,
        "agreeing": result.agreeing,
        "disagreeing": result.disagreeing,
        "checklist": checklist,
        "why_not_trading": why_not_trading,
        "trade_plan": {
            "entry_zone": list(result.entry_zone) if result.entry_zone else None,
            "stop_loss": result.stop_loss,
            "take_profit1": result.take_profit1,
            "take_profit2": result.take_profit2,
            "take_profit3": result.take_profit3,
            "risk_reward": result.risk_reward,
        },
    }
