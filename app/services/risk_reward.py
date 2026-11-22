"""Reward:risk ratios derived from the actual entry, stop and targets."""
from __future__ import annotations

from typing import Any


def risk_reward_ratios(entry: float, stop_loss: float,
                       take_profit1: float | None,
                       take_profit2: float | None) -> dict[str, float]:
    """Return risk plus per-target R:R; zero risk never yields a fake ratio."""
    risk = abs(float(entry) - float(stop_loss))
    out: dict[str, float] = {"risk": round(risk, 6),
                             "risk_reward_tp1": 0.0, "risk_reward_tp2": 0.0}
    if risk <= 0:
        return out
    for key, target in (("risk_reward_tp1", take_profit1),
                        ("risk_reward_tp2", take_profit2)):
        if target is not None:
            out[key] = round(abs(float(target) - float(entry)) / risk, 6)
    return out


def rr_text(ratios: dict[str, Any]) -> str:
    first = float(ratios.get("risk_reward_tp1") or 0.0)
    second = float(ratios.get("risk_reward_tp2") or 0.0)
    if second > 0:
        return f"1:{first:.1f} / 1:{second:.1f}"
    if first > 0:
        return f"1:{first:.1f}"
    return "N/A"
