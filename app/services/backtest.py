"""Backtesting engine: rule+ML strategy over historical candles.

Assumptions: spread and commission are charged per round trip, entries happen
at the next bar's open after a signal bar closes, exits use intrabar high/low
for stop/take-profit (conservative: stop checked first).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.concepts.base import session_of
from app.services.indicators import add_all
from app.services.signal_engine import _rule_components, combine_rule_score

logger = logging.getLogger(__name__)

_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_cache_lock = threading.Lock()
_MAX_CACHED_BACKTESTS = 2


def _side_statistics(trades: list[dict[str, Any]], side: str) -> dict[str, Any]:
    """Win-rate / PnL statistics for one side ('short' = drop/SELL trades)."""
    subset = [t for t in trades if t.get("side") == side]
    if not subset:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
            "avg_pnl_pct": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        }
    wins = [t for t in subset if t["pnl"] > 0]
    losses = [t for t in subset if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0)
    total_pnl = sum(t["pnl"] for t in subset)
    return {
        "trades": len(subset),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(subset) * 100.0, 2),
        "profit_factor": round(profit_factor, 3),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(subset), 2),
        "avg_pnl_pct": round(sum(t["pnl_pct"] for t in subset) / len(subset), 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


def _group_statistics(trades: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """Win rate / PF / expectancy / drawdown by an arbitrary trade attribute."""
    grouped: dict[str, list[float]] = {}
    for trade in trades:
        key = str(trade.get(field) or "unknown")
        result = trade.get("pnl_pct")
        if isinstance(result, (int, float)) and np.isfinite(result):
            grouped.setdefault(key, []).append(float(result))
    out: dict[str, Any] = {}
    for key, results in sorted(grouped.items()):
        wins = [r for r in results if r > 0]
        losses = [r for r in results if r <= 0]
        gross = sum(wins)
        gross_loss = abs(sum(losses))
        curve = np.cumsum(results)
        peak = np.maximum.accumulate(np.concatenate(([0.0], curve)))[1:]
        equity = 100.0 + curve
        drawdown = float(np.max((np.maximum.accumulate(equity) - equity) /
                                np.maximum.accumulate(equity) * 100.0)) if len(curve) else 0.0
        out[key] = {
            "trades": len(results),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(results) * 100.0, 2),
            "profit_factor": round(gross / gross_loss, 3) if gross_loss > 0 else (
                999.0 if gross > 0 else 0.0),
            "expectancy": round(float(np.mean(results)), 4),
            "max_drawdown_pct": round(drawdown, 2),
            "total_pct": round(float(np.sum(results)), 2),
        }
    return out


def _bar_signal(row: pd.Series, prev: pd.Series | None,
                dxy_chg: float | None) -> tuple[float, str]:
    """Fast per-bar rule score (ML disabled inside the backtest loop for speed;
    the ML contribution is measured separately via walk-forward)."""
    comps = _rule_components(row, prev,
                             pd.Series({"change_pct": dxy_chg}) if dxy_chg is not None else None)
    if not comps:
        return 0.0, "HOLD"
    score = combine_rule_score(comps)
    action = ("BUY" if score >= config.SIGNAL_BUY_THRESHOLD else
              "SELL" if score <= config.SIGNAL_SELL_THRESHOLD else "HOLD")
    return score, action


def run_backtest(df: pd.DataFrame, dxy: pd.DataFrame | None = None,
                 years: int | None = None, use_cache: bool = True
                 ) -> dict[str, Any]:
    """Backtest the strategy; returns metrics, trades and equity curve."""
    years = years or config.BACKTEST_YEARS
    cache_key = f"{len(df)}|{years}|{float(df['Close'].iloc[-1]):.2f}"
    if use_cache:
        with _cache_lock:
            cached = _cache.get(cache_key)
            if cached is not None:
                _cache.move_to_end(cache_key)
                return cached

    ind = add_all(df)
    if dxy is not None and len(dxy):
        dxy_chg = dxy["Close"].pct_change() * 100.0
        dxy_series = dxy_chg.reindex(ind.index).ffill()
    else:
        dxy_series = pd.Series(np.nan, index=ind.index)

    # restrict to last `years`
    cutoff = ind.index.max() - pd.DateOffset(years=years)
    start_idx = ind.index.searchsorted(cutoff)
    start_idx = max(start_idx, 250)  # keep warmup for indicators

    spread = config.SPREAD_USD
    commission = config.COMMISSION_USD
    equity = 100_000.0
    peak = equity
    max_dd = 0.0
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    open_trade: dict[str, Any] | None = None
    daily_returns: list[float] = []

    idx_list = list(ind.index)
    for i in range(start_idx, len(ind) - 1):
        row = ind.iloc[i]
        prev = ind.iloc[i - 1] if i > 0 else None
        ts = idx_list[i]
        price = float(row["Close"])

        # mark-to-market equity for the curve
        if open_trade is not None:
            if open_trade["side"] == "long":
                unreal = (price - open_trade["entry"]) * open_trade["qty"]
            else:
                unreal = (open_trade["entry"] - price) * open_trade["qty"]
            mark = equity + unreal
        else:
            mark = equity
        equity_curve.append({"t": str(ts), "equity": round(mark, 2)})

        # manage open trade using this bar's high/low
        if open_trade is not None:
            high, low = float(row["High"]), float(row["Low"])
            exit_px = None
            if open_trade["side"] == "long":
                if low <= open_trade["sl"]:
                    exit_px = open_trade["sl"]
                elif high >= open_trade["tp2"]:
                    exit_px = open_trade["tp2"]
            else:
                if high >= open_trade["sl"]:
                    exit_px = open_trade["sl"]
                elif low <= open_trade["tp2"]:
                    exit_px = open_trade["tp2"]
            if exit_px is not None:
                gross = ((exit_px - open_trade["entry"]) if open_trade["side"] == "long"
                         else (open_trade["entry"] - exit_px)) * open_trade["qty"]
                costs = spread * open_trade["qty"] + commission
                pnl = gross - costs
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100.0
                max_dd = max(max_dd, dd)
                trades.append({
                    "entry_time": open_trade["entry_time"],
                    "exit_time": str(ts),
                    "side": open_trade["side"],
                    "entry": round(open_trade["entry"], 2),
                    "exit": round(exit_px, 2),
                    "qty": round(open_trade["qty"], 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / open_trade["equity_start"] * 100.0, 4),
                    "reason": "stop" if exit_px == open_trade["sl"] else "target",
                    "setup": open_trade["setup"], "session": open_trade["session"],
                })
                daily_returns.append(pnl / open_trade["equity_start"])
                open_trade = None
                continue  # no same-bar re-entry

        # generate signal on this closed bar -> act at next open
        score, action = _bar_signal(row, prev, float(dxy_series.iloc[i])
                                    if pd.notna(dxy_series.iloc[i]) else None)
        if action == "HOLD" or open_trade is not None:
            continue

        next_row = ind.iloc[i + 1]
        entry = float(next_row["Open"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else entry * 0.01
        if action == "BUY":
            sl = entry - config.ATR_STOP_MULT * atr
            tp2 = entry + config.TP2_ATR_MULT * atr
            side = "long"
        else:
            sl = entry + config.ATR_STOP_MULT * atr
            tp2 = entry - config.TP2_ATR_MULT * atr
            side = "short"
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            continue
        # risk 1% of equity per trade
        risk_amount = equity * 0.01
        qty = risk_amount / risk_per_unit
        open_trade = {
            "side": side, "entry": entry, "sl": sl, "tp2": tp2,
            "qty": qty, "entry_time": str(idx_list[i + 1]),
            "equity_start": equity,
            "setup": f"{action} ({'strong' if abs(score) >= 0.5 else 'standard'})",
            "session": session_of(idx_list[i + 1]),
        }

    # close any open trade at the last close
    if open_trade is not None:
        last = float(ind["Close"].iloc[-1])
        gross = ((last - open_trade["entry"]) if open_trade["side"] == "long"
                 else (open_trade["entry"] - last)) * open_trade["qty"]
        costs = spread * open_trade["qty"] + commission
        pnl = gross - costs
        equity += pnl
        trades.append({
            "entry_time": open_trade["entry_time"], "exit_time": str(ind.index[-1]),
            "side": open_trade["side"], "entry": round(open_trade["entry"], 2),
            "exit": round(last, 2), "qty": round(open_trade["qty"], 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / open_trade["equity_start"] * 100.0, 4),
            "reason": "end",
            "setup": open_trade["setup"], "session": open_trade["session"],
        })
        daily_returns.append(pnl / open_trade["equity_start"])

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)
    rets = np.array(daily_returns) if daily_returns else np.array([0.0])
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252.0)
    total_return = (equity / 100_000.0 - 1.0) * 100.0

    # downsample equity curve for the UI (~500 points)
    curve = equity_curve
    if len(curve) > 500:
        step = len(curve) // 500
        curve = curve[::step] + [curve[-1]]

    result = {
        "years": years,
        "bars": int(len(ind) - start_idx),
        "trades": len(trades),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(float(sharpe), 2),
        "total_return_pct": round(total_return, 2),
        "final_equity": round(equity, 2),
        "assumptions": {
            "spread_usd_per_oz": spread,
            "commission_usd_per_trade": commission,
            "risk_per_trade_pct": 1.0,
            "sl_atr_mult": config.ATR_STOP_MULT,
            "tp_atr_mult": config.TP2_ATR_MULT,
            "initial_equity": 100_000,
        },
        "trade_list": trades[-100:],
        "by_side": {
            "short": _side_statistics(trades, "short"),
            "long": _side_statistics(trades, "long"),
        },
        "by_setup": _group_statistics(trades, "setup"),
        "by_session": _group_statistics(trades, "session"),
        "equity_curve": curve,
        "note": ("Backtest covers the last %s years of available data. "
                 "Past performance does not guarantee future results. "
                 "Results include spread costs; slippage is not modelled." % years),
    }
    if use_cache:
        with _cache_lock:
            _cache[cache_key] = result
            _cache.move_to_end(cache_key)
            while len(_cache) > _MAX_CACHED_BACKTESTS:
                _cache.popitem(last=False)
    return result


def clear_backtest_cache() -> None:
    """Reset cached backtest results."""
    with _cache_lock:
        _cache.clear()
