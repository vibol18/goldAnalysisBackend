"""Validation: per-concept performance and full confluence strategy backtests.

All detections used here are produced by the causal detectors; trades are
simulated only on bars *after* a concept's confirmation bar. Costs (spread +
commission) are charged per round trip. Slippage is not modelled.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.concepts import DETECTOR_CATEGORIES, detect_all
from app.concepts.base import CATEGORIES
from app.services.confluence import confluence, load_weights

logger = logging.getLogger(__name__)


def _stats(returns: list[float]) -> dict[str, Any]:
    if not returns:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0}
    arr = np.asarray(returns, dtype=float)
    wins, losses = arr[arr > 0], arr[arr <= 0]
    gp, gl = float(wins.sum()), abs(float(losses.sum()))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(np.abs(peak), 1e-9) * 100.0
    sharpe = (arr.mean() / (arr.std() + 1e-12)) * np.sqrt(252.0)
    return {
        "trades": int(len(arr)),
        "win_rate": round(float((arr > 0).mean() * 100.0), 2),
        "profit_factor": round(float(pf), 3),
        "expectancy": round(float(arr.mean() * 100.0), 4),
        "max_drawdown_pct": round(float(dd.max()), 2),
        "sharpe": round(float(sharpe), 2),
    }


def _atr(df: pd.DataFrame) -> float:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    value = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1]
    return float(value) if np.isfinite(value) else float(df["Close"].iloc[-1]) * 0.01


def per_concept_performance(df: pd.DataFrame, concepts: list[Any] | None = None,
                            max_hold: int = 20,
                            spread_usd: float | None = None) -> list[dict[str, Any]]:
    """Fixed 2R-target / 1R-stop test of every directional concept type.

    Entry is the concept's zone midpoint (or the close at confirmation),
    stop = 1 ATR and target = 2 ATR. This is a comparability test across
    concepts, not a tradeable strategy by itself.
    """
    spread = float(spread_usd if spread_usd is not None else config.SPREAD_USD)
    concepts = concepts if concepts is not None else detect_all(df, unlimited=True)
    index = df.index
    highs, lows, closes = (df["High"].to_numpy(float), df["Low"].to_numpy(float),
                           df["Close"].to_numpy(float))
    a = _atr(df)
    by_type: dict[str, list[float]] = {}
    for c in concepts:
        if c.direction not in ("bullish", "bearish") or c.status == "invalidated":
            continue
        ts = pd.Timestamp(c.end_time or c.start_time)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if index.tz is not None:
            ts = ts.tz_localize("UTC")
        pos = int(index.searchsorted(ts, side="right"))
        if pos < 1 or pos >= len(df) - 1:
            continue
        if c.top is not None and c.bottom is not None:
            entry = (float(c.top) + float(c.bottom)) / 2.0
        else:
            entry = float(closes[pos - 1])
        sign = 1.0 if c.direction == "bullish" else -1.0
        stop = entry - sign * a
        target = entry + sign * 2.0 * a
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        result = 0.0
        for j in range(pos, min(pos + max_hold, len(df))):
            high, low = float(highs[j]), float(lows[j])
            if sign > 0:
                if low <= stop:
                    result = -risk - spread
                    break
                if high >= target:
                    result = 2.0 * risk - spread
                    break
            else:
                if high >= stop:
                    result = -risk - spread
                    break
                if low <= target:
                    result = 2.0 * risk - spread
                    break
        else:
            result = (float(closes[-1]) - entry) * sign - spread
        by_type.setdefault(c.type, []).append(result / risk)

    weights = load_weights()["weights"]
    rows: list[dict[str, Any]] = []
    for type_, returns in by_type.items():
        stats = _stats(returns)
        stats.update({
            "type": type_,
            "category": DETECTOR_CATEGORIES.get(type_, "classic"),
            "weight": float(weights.get(type_, 0.25)),
        })
        rows.append(stats)
    rows.sort(key=lambda r: (r["category"], r["type"]))
    return rows


def full_confluence_backtest(df: pd.DataFrame, years: int | None = None,
                             spread_usd: float | None = None,
                             commission_usd: float | None = None,
                             context: dict[str, Any] | None = None,
                             concepts: list[Any] | None = None) -> dict[str, Any]:
    """Replay the graded confluence strategy with no future concepts."""
    years = years or config.BACKTEST_YEARS
    spread = float(spread_usd if spread_usd is not None else config.SPREAD_USD)
    commission = float(commission_usd if commission_usd is not None else config.COMMISSION_USD)
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    work = df.copy()
    work.index = index
    if years:
        cutoff = index.max() - pd.DateOffset(years=years)
        start = max(index.searchsorted(cutoff), 250)
    else:
        start = 250
    if start >= len(work) - 2:
        raise ValueError("not enough history for concept backtest")

    concepts = concepts if concepts is not None else detect_all(work, context or {}, unlimited=True)
    confirmed: dict[int, list[Any]] = {}
    for c in concepts:
        ts = pd.Timestamp(c.end_time or c.start_time)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        pos = int(index.searchsorted(ts, side="right"))
        confirmed.setdefault(pos, []).append(c)

    equity, peak, max_dd = 100_000.0, 100_000.0, 0.0
    open_trade: dict[str, Any] | None = None
    returns: list[float] = []
    trades: list[dict[str, Any]] = []
    active: list[Any] = []
    high_arr, low_arr, open_arr, close_arr = (
        work["High"].to_numpy(float), work["Low"].to_numpy(float),
        work["Open"].to_numpy(float), work["Close"].to_numpy(float))

    for i in range(start, len(work) - 1):
        active.extend(confirmed.get(i, []))
        if open_trade is not None:
            high, low = float(high_arr[i]), float(low_arr[i])
            exit_px = None
            if open_trade["side"] == "long":
                if low <= open_trade["sl"]:
                    exit_px = open_trade["sl"]
                elif high >= open_trade["tp"]:
                    exit_px = open_trade["tp"]
            else:
                if high >= open_trade["sl"]:
                    exit_px = open_trade["sl"]
                elif low <= open_trade["tp"]:
                    exit_px = open_trade["tp"]
            if exit_px is not None:
                gross = ((exit_px - open_trade["entry"]) if open_trade["side"] == "long"
                         else (open_trade["entry"] - exit_px)) * open_trade["qty"]
                pnl = gross - spread * open_trade["qty"] - commission
                equity += pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / max(peak, 1.0) * 100.0)
                returns.append(pnl / open_trade["equity_start"])
                trades.append({
                    "entry_time": str(index[i]), "side": open_trade["side"],
                    "entry": round(open_trade["entry"], 2),
                    "exit": round(exit_px, 2), "pnl": round(pnl, 2),
                    "grade": open_trade["grade"],
                })
                open_trade = None
                continue

        try:
            result = confluence(active, work.iloc[:i + 1], spread_usd=spread)
        except Exception as exc:  # noqa: BLE001
            logger.debug("confluence at bar %s failed: %s", i, exc)
            continue
        if result.action == "HOLD" or result.stop_loss is None or result.take_profit2 is None:
            continue
        entry = float(open_arr[i + 1])
        stop = float(result.stop_loss)
        target = float(result.take_profit2)
        risk = abs(entry - stop)
        if risk <= 0 or abs(target - entry) < 1.5 * risk:
            continue
        qty = (equity * 0.01) / risk
        open_trade = {
            "side": "long" if result.action == "BUY" else "short",
            "entry": entry, "sl": stop, "tp": target,
            "qty": qty, "equity_start": equity, "grade": result.grade,
        }

    if open_trade is not None:
        last = float(close_arr[-1])
        gross = ((last - open_trade["entry"]) if open_trade["side"] == "long"
                 else (open_trade["entry"] - last)) * open_trade["qty"]
        pnl = gross - spread * open_trade["qty"] - commission
        equity += pnl
        returns.append(pnl / open_trade["equity_start"])
        trades.append({"entry_time": str(index[-1]), "side": open_trade["side"],
                       "entry": round(open_trade["entry"], 2), "exit": round(last, 2),
                       "pnl": round(pnl, 2), "grade": open_trade["grade"]})

    result = _stats(returns)
    result.update({
        "years": years, "bars": int(len(work) - start),
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity / 100_000.0 - 1.0) * 100.0, 2),
        "concepts_used": len(concepts),
        "categories": list(CATEGORIES),
        "trade_list": trades[-100:],
        "assumptions": {
            "spread_usd_per_oz": spread, "commission_usd_per_trade": commission,
            "risk_per_trade_pct": 1.0, "initial_equity": 100_000,
            "slippage_modelled": False,
        },
        "note": ("Concept confluence replay on real XAU/USD candles. Detections are "
                 "causal; trades enter at the next open after confirmation. Spread "
                 "and commission included; slippage and financing are not modelled. "
                 "Past performance does not guarantee future results."),
        "disclaimer": ("Educational analysis only. Not financial advice. Signals can "
                       "be wrong. Never risk money you can't afford to lose."),
    })
    return result
