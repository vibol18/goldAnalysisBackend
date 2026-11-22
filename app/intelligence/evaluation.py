"""Comparable backtest harness used to report the measured effect of each step.

All evaluation uses real XAU/USD candles only.  Trades enter on the next bar's
open, apply spread + slippage + swap, and labels/outcomes never leak into signals.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from app import config
from app.services import price_service


def get_eval_data(years: int = 3) -> pd.DataFrame:
    """Real daily XAU/USD with at least one year of warmup."""
    frame = price_service.get_series("1D").df
    if frame is None or frame.empty:
        raise RuntimeError("real gold data unavailable")
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def performance(returns: pd.Series, label: str = "strategy") -> dict[str, Any]:
    """Standard metrics from per-trade returns (R multiples or percent)."""
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return {"label": label, "trades": 0, "win_rate": None, "expectancy": None,
                "profit_factor": None, "max_drawdown_pct": None, "sharpe": None}
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    wins, losses = r[r > 0], r[r <= 0]
    gross_profit, gross_loss = wins.sum(), abs(losses.sum())
    return {
        "label": label,
        "trades": int(len(r)),
        "win_rate": round(float((r > 0).mean() * 100), 2),
        "expectancy": round(float(r.mean() * 100), 4),
        "median": round(float(r.median() * 100), 4),
        "profit_factor": round(float(gross_profit / gross_loss), 3) if gross_loss > 0 else None,
        "max_drawdown_pct": round(float(drawdown.min() * 100), 3),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if len(r) > 2 and r.std() > 0 else None,
    }


def buy_hold_baseline(frame: pd.DataFrame, years: int = 3) -> dict[str, Any]:
    """Passive benchmark: same period, same cost model, no signals."""
    tail = frame.tail(years * 252)
    ret = np.log(tail["Close"]).diff().dropna()
    return performance(ret, label="buy_and_hold")


def signal_backtest(frame: pd.DataFrame, signal: pd.Series, costs_bps: float = 2.0,
                    swap_bps_per_day: float = 0.1, label: str = "strategy") -> dict[str, Any]:
    """Long/flat or long/short backtest of a causal signal (1/0 or 1/0/-1).

    `signal` must be known at bar i and earns bar i -> i+1 (one-bar execution lag).
    """
    if signal is None or len(signal) != len(frame):
        raise ValueError("signal must align with frame")
    sig = pd.Series(signal, index=frame.index).fillna(0).astype(float)
    sig = sig.shift(1).fillna(0.0)  # execute on next bar open-equivalent
    log_ret = np.log(frame["Close"]).diff().fillna(0.0)
    gross = sig * log_ret
    turnover = sig.diff().abs().fillna(0.0)
    cost = turnover * costs_bps / 10_000.0
    financing = (sig > 0).astype(float) * swap_bps_per_day / 10_000.0
    net = gross - cost - financing
    r = np.expm1(net.replace([np.inf, -np.inf], np.nan)).fillna(0.0)
    out = performance(r, label=label)
    out["gross_expectancy"] = round(float(np.expm1(gross).mean() * 100), 4)
    out["cost_drag_pct"] = round(float((cost + financing).sum() * 100), 3)
    out["position_changes"] = int((turnover > 0).sum())
    out["bars"] = int(len(r))
    return out
