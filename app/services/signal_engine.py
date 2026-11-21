"""Signal engine: rule-based score + ML probability -> BUY/SELL/HOLD signal."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.concepts.base import ts_to_str
from app.ml.model import MODEL
from app.services.indicators import add_all

logger = logging.getLogger(__name__)


def closed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only CLOSED candles — the last bar is still forming."""
    return df.iloc[:-1] if len(df) > 2 else df


@dataclass
class Signal:
    """A trading signal with pricing and a plain-language explanation."""

    action: str
    confidence: float
    final_score: float
    rule_score: float
    ml_probability: float | None
    entry: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    risk_reward: float
    atr: float
    price: float
    timeframe: str
    explanation: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    grade: str = "C"
    signal_time: str = ""
    entry_zone: tuple[float, float] | None = None
    take_profit3: float | None = None
    invalidation: str = ""
    confluence: dict[str, Any] | None = None
    concepts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 1),
            "final_score": round(self.final_score, 4),
            "rule_score": round(self.rule_score, 2),
            "ml_probability": self.ml_probability,
            "entry": round(self.entry, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit1": round(self.take_profit1, 2),
            "take_profit2": round(self.take_profit2, 2),
            "risk_reward": round(self.risk_reward, 2),
            "atr": round(self.atr, 3),
            "price": round(self.price, 2),
            "timeframe": self.timeframe,
            "explanation": self.explanation,
            "created_at": self.created_at,
            "grade": self.grade,
            "signal_time": self.signal_time,
            "entry_zone": list(self.entry_zone) if self.entry_zone else None,
            "take_profit3": self.take_profit3,
            "invalidation": self.invalidation,
            "confluence": self.confluence,
            "concepts": self.concepts,
        }


def _rule_components(row: pd.Series, prev: pd.Series | None,
                     dxy_row: pd.Series | None) -> list[dict[str, Any]]:
    """Score each rule in [-1, 1]; positive = bullish for gold.

    Each component also carries a weight; the combined rule score is the
    weight-normalised mean (trend/momentum rules outweigh mean-reversion).
    """
    comps: list[dict[str, Any]] = []

    def add(name: str, score: float, why: str, weight: float) -> None:
        comps.append({"name": name, "score": float(np.clip(score, -1, 1)),
                      "weight": weight, "why": why})

    # --- EMA trend -------------------------------------------------------
    close = float(row["Close"])
    e20, e50, e200 = row.get("ema20"), row.get("ema50"), row.get("ema200")
    if pd.notna(e20) and pd.notna(e50) and pd.notna(e200):
        s = 0.0
        s += 0.35 if close > e20 else -0.35
        s += 0.35 if e20 > e50 else -0.35
        s += 0.30 if e50 > e200 else -0.30
        crossed = ""
        if prev is not None and pd.notna(prev.get("ema50")) and pd.notna(prev.get("ema200")):
            if prev["ema50"] <= prev["ema200"] and e50 > e200:
                s = min(1.0, s + 0.4)
                crossed = " EMA50 crossed above EMA200 (golden cross)."
            elif prev["ema50"] >= prev["ema200"] and e50 < e200:
                s = max(-1.0, s - 0.4)
                crossed = " EMA50 crossed below EMA200 (death cross)."
        add("EMA trend", s,
            f"Price {'above' if close > e20 else 'below'} EMA20, "
            f"EMA20 {'>' if e20 > e50 else '<'} EMA50, "
            f"EMA50 {'>' if e50 > e200 else '<'} EMA200." + crossed,
            weight=0.30)

    # --- RSI -------------------------------------------------------------
    rsi = row.get("rsi")
    if pd.notna(rsi):
        slope = row.get("rsi_slope", 0.0) or 0.0
        if rsi < 30:
            s = 0.8
            why = f"RSI {rsi:.0f} is oversold (<30) — bounce potential."
        elif rsi > 70:
            s = -0.8
            why = f"RSI {rsi:.0f} is overbought (>70) — pullback risk."
        else:
            # mid-range: momentum direction matters
            s = ((rsi - 50) / 50.0) * 0.6
            s += 0.3 if slope > 0 else -0.3
            why = (f"RSI {rsi:.0f} {'rising' if slope > 0 else 'falling'} "
                   f"in neutral territory.")
        add("RSI", s, why, weight=0.15)

    # --- MACD ------------------------------------------------------------
    macd, msig, mhist = row.get("macd"), row.get("macd_signal"), row.get("macd_hist")
    if pd.notna(macd) and pd.notna(msig):
        s = 0.0
        why_parts = []
        if macd > msig:
            s += 0.5
            why_parts.append("MACD above signal line")
        else:
            s -= 0.5
            why_parts.append("MACD below signal line")
        if row.get("macd_cross_up"):
            s = 1.0
            why_parts.append("fresh bullish cross this bar")
        elif row.get("macd_cross_down"):
            s = -1.0
            why_parts.append("fresh bearish cross this bar")
        if pd.notna(mhist):
            s += 0.2 if mhist > 0 else -0.2
        add("MACD", s, ", ".join(why_parts) + ".", weight=0.25)

    # --- Bollinger position ---------------------------------------------
    bbp = row.get("bb_pct")
    if pd.notna(bbp):
        if bbp >= 0.95:
            s, why = -0.6, f"Price at/above upper Bollinger band (pos {bbp:.2f}) — stretched."
        elif bbp <= 0.05:
            s, why = 0.6, f"Price at/below lower Bollinger band (pos {bbp:.2f}) — stretched low."
        else:
            s = (bbp - 0.5) * 1.2
            why = f"Price at {bbp:.0%} of the Bollinger range."
        add("Bollinger", s, why, weight=0.15)

    # --- ATR volatility (context, mild conviction scaling) --------------
    atr_pct = row.get("atr_pct")
    if pd.notna(atr_pct):
        # high volatility amplifies the *existing* lean is done later; here neutral
        if atr_pct > 1.6:
            add("Volatility", 0.0,
                f"ATR {atr_pct:.2f}% of price — elevated volatility, size positions smaller.",
                weight=0.05)
        else:
            add("Volatility", 0.0,
                f"ATR {atr_pct:.2f}% of price — normal volatility regime.",
                weight=0.05)

    # --- Support / resistance breaks ------------------------------------
    res, sup = row.get("resistance"), row.get("support")
    if pd.notna(res) and close > res:
        add("S/R break", 0.9, f"Price broke above resistance at {res:.2f}.", weight=0.10)
    elif pd.notna(sup) and close < sup:
        add("S/R break", -0.9, f"Price broke below support at {sup:.2f}.", weight=0.10)
    elif pd.notna(res) and pd.notna(sup):
        pos = (close - sup) / max(res - sup, 1e-9)
        add("S/R position", (pos - 0.5) * 0.8,
            f"Price sits {pos:.0%} of the way from support {sup:.2f} "
            f"to resistance {res:.2f}.", weight=0.10)

    # --- DXY context (gold usually inversely correlated) ----------------
    if dxy_row is not None:
        dxy_chg = dxy_row.get("change_pct")
        if dxy_chg is not None and pd.notna(dxy_chg):
            s = float(np.clip(-dxy_chg / 0.5, -0.6, 0.6))  # -0.5% DXY => +0.6 gold
            direction = "falling" if dxy_chg < 0 else "rising"
            add("DXY", s, f"DXY {direction} ({dxy_chg:+.2f}%) — "
                f"{'tailwind' if dxy_chg < 0 else 'headwind'} for gold.",
                weight=0.15)
    return comps


def combine_rule_score(comps: list[dict[str, Any]]) -> float:
    """Weight-normalised mean of component scores, scaled into [-1, 1]."""
    if not comps:
        return 0.0
    total_w = sum(c.get("weight", 1.0) for c in comps) or 1.0
    raw = sum(c["score"] * c.get("weight", 1.0) for c in comps) / total_w
    return float(np.clip(raw * 2.2, -1.0, 1.0))


def generate_signal(df: pd.DataFrame, timeframe: str,
                    dxy_change_pct: float | None = None,
                    train_model: bool = True,
                    use_concepts: bool = True,
                    concept_context: dict[str, Any] | None = None) -> Signal:
    """Produce the concept-driven BUY/SELL/HOLD signal from an OHLCV frame."""
    ind = add_all(df)
    row = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) > 1 else None
    dxy_row = pd.Series({"change_pct": dxy_change_pct}) if dxy_change_pct is not None else None

    comps = _rule_components(row, prev, dxy_row)
    rule_score = combine_rule_score(comps)

    ml_prob: float | None = None
    if train_model:
        try:
            MODEL.train(df)
            ml_prob = MODEL.predict_proba(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML unavailable: %s", exc)

    price = float(row["Close"])
    atr = float(row["atr"]) if pd.notna(row["atr"]) else price * 0.01

    concepts: list[Any] = []
    cf = None
    if use_concepts:
        try:
            from app.concepts import detect_all
            from app.services.confluence import confluence
            concepts = detect_all(df, dict(concept_context or {}))
            cf = confluence(concepts, df, ml_prob)
        except Exception as exc:  # noqa: BLE001
            logger.warning("confluence unavailable: %s", exc, exc_info=True)
            concepts, cf = [], None

    if cf is not None:
        final = cf.score
        action = cf.action
    elif ml_prob is not None:
        ml_score = (ml_prob - 0.5) * 2.0  # -1..1
        final = 0.55 * rule_score + 0.45 * ml_score
        action = ("BUY" if final >= config.SIGNAL_BUY_THRESHOLD else
                  "SELL" if final <= config.SIGNAL_SELL_THRESHOLD else "HOLD")
    else:
        final = rule_score
        action = ("BUY" if final >= config.SIGNAL_BUY_THRESHOLD else
                  "SELL" if final <= config.SIGNAL_SELL_THRESHOLD else "HOLD")

    if action == "BUY":
        entry = price
        sl = entry - config.ATR_STOP_MULT * atr
        tp1 = entry + config.TP1_ATR_MULT * atr
        tp2 = entry + config.TP2_ATR_MULT * atr
    elif action == "SELL":
        entry = price
        sl = entry + config.ATR_STOP_MULT * atr
        tp1 = entry - config.TP1_ATR_MULT * atr
        tp2 = entry - config.TP2_ATR_MULT * atr
    else:
        # HOLD: show hypothetical long levels for planning only
        entry = price
        sl = entry - config.ATR_STOP_MULT * atr
        tp1 = entry + config.TP1_ATR_MULT * atr
        tp2 = entry + config.TP2_ATR_MULT * atr

    grade = "C"
    entry_zone: tuple[float, float] | None = None
    tp3: float | None = None
    invalidation = ""
    cf_dict: dict[str, Any] | None = None
    cf_factors: list[dict[str, Any]] = []

    if cf is not None:
        grade = cf.grade
        entry_zone = cf.entry_zone
        tp3 = cf.take_profit3
        invalidation = cf.invalidation
        cf_dict = cf.to_dict()
        cf_factors = cf.factors
        if cf.entry_zone:
            entry = (cf.entry_zone[0] + cf.entry_zone[1]) / 2.0
        if cf.stop_loss is not None:
            sl = cf.stop_loss
        if cf.take_profit1 is not None:
            tp1 = cf.take_profit1
        if cf.take_profit2 is not None:
            tp2 = cf.take_profit2
        confidence = cf.confidence
    else:
        agreement = abs(sum(c["score"] for c in comps)) / max(len(comps), 1)
        base_conf = min(abs(final) / 0.8, 1.0)
        confidence = float(np.clip((0.7 * base_conf + 0.3 * agreement) * 100.0, 5.0, 97.0))

    risk = abs(entry - sl)
    reward = abs(tp2 - entry)
    rr = reward / risk if risk > 0 else 0.0
    if cf is not None:
        rr = cf.risk_reward

    bullish = [c for c in comps if c["score"] > 0.1]
    bearish = [c for c in comps if c["score"] < -0.1]
    if action == "SELL":
        reasons = [c["why"] for c in bearish][:5]
    elif action == "BUY":
        reasons = [c["why"] for c in bullish][:5]
    else:
        # HOLD: show the strongest drivers on BOTH sides so the user sees
        # exactly why the engine is undecided.
        strongest = sorted(comps, key=lambda c: abs(c["score"]), reverse=True)
        reasons = [f"[{'bull' if c['score'] > 0 else 'bear'}] {c['why']}"
                   for c in strongest if abs(c["score"]) > 0.1][:5]
    summary_bits = []
    if action == "BUY":
        summary_bits.append(f"Bullish setup: rule score {rule_score:+.2f}")
    elif action == "SELL":
        summary_bits.append(f"Bearish setup: rule score {rule_score:+.2f}")
    else:
        summary_bits.append(f"Mixed/neutral setup: rule score {rule_score:+.2f}")
    if ml_prob is not None:
        summary_bits.append(
            f"ML model gives {ml_prob:.0%} probability price is higher in "
            f"{config.ML_HORIZON} candles")
    summary = ". ".join(summary_bits) + "."

    if cf is not None:
        from app.services.explanation import build_explanation
        explanation = build_explanation(concepts, cf, df, timeframe,
                                        rule_score=rule_score,
                                        ml_probability=ml_prob,
                                        rule_components=comps)
    else:
        explanation = {
            "summary": summary,
            "reasons": reasons,
            "components": comps,
            "rule_score": round(rule_score, 3),
            "ml_probability": ml_prob,
            "timeframe": timeframe,
        }

    return Signal(
        action=action,
        confidence=confidence,
        final_score=final,
        rule_score=rule_score,
        ml_probability=ml_prob,
        entry=entry,
        stop_loss=sl,
        take_profit1=tp1,
        take_profit2=tp2,
        risk_reward=rr,
        atr=atr,
        price=price,
        timeframe=timeframe,
        explanation=explanation,
        created_at=datetime.now(timezone.utc).isoformat(),
        grade=grade,
        signal_time=ts_to_str(ind.index[-1]) or "",
        entry_zone=entry_zone,
        take_profit3=tp3,
        invalidation=invalidation,
        confluence=cf_dict,
        concepts=[c.to_dict() for c in concepts[:200]],
    )


def evaluate_outcome(sig: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Check whether an open signal hit TP or SL on bars after its creation.

    Only bars strictly after the signal time are inspected (no leakage).
    """
    created = sig.get("created_at")
    try:
        ts = pd.Timestamp(created)
    except (ValueError, TypeError):
        return {"status": sig.get("status", "open"), "hit": sig.get("hit"),
                "result_pct": sig.get("result_pct")}
    # align timezone with the candle index (yfinance intraday indexes can be tz-aware)
    idx_tz = getattr(df.index, "tz", None)
    if idx_tz is None:
        ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
    else:
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    future = df[df.index > ts]
    if len(free := future) == 0:
        return {"status": "open", "hit": None, "result_pct": None}

    action = sig["action"]
    sl, tp1, tp2 = sig["stop_loss"], sig["take_profit1"], sig["take_profit2"]
    entry = sig["entry"]
    hit = None
    for _, bar in free.iterrows():
        high, low = float(bar["High"]), float(bar["Low"])
        if action == "BUY":
            if low <= sl:
                hit = "SL"
                break
            if high >= tp2:
                hit = "TP"
                break
            if high >= tp1:
                hit = "TP1"
                break
        elif action == "SELL":
            if high >= sl:
                hit = "SL"
                break
            if low <= tp2:
                hit = "TP"
                break
            if low <= tp1:
                hit = "TP1"
                break
        else:
            break
    if hit is None:
        last = float(free["Close"].iloc[-1])
        result = (last - entry) / entry * 100.0 if action == "BUY" else (entry - last) / entry * 100.0
        if action == "HOLD":
            result = 0.0
        return {"status": "open", "hit": None, "result_pct": round(result, 4)}

    if hit in ("TP", "TP1"):
        exit_px = tp2 if hit == "TP" else tp1
        status = "closed"
    else:
        exit_px = sl
        status = "closed"
    result = (exit_px - entry) / entry * 100.0 if action == "BUY" else (entry - exit_px) / entry * 100.0
    if action == "HOLD":
        result = 0.0
    return {"status": status, "hit": "TP" if hit in ("TP", "TP1") else "SL",
            "result_pct": round(result, 4)}
