"""Market-data providers, tried in priority order by :mod:`app.services.data`.

A provider either returns real OHLCV data or raises. No provider ever fabricates
prices: if every provider fails the caller serves the last real cached snapshot
and flags it as stale (or reports that no data is available).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import httpx

from app import config

logger = logging.getLogger(__name__)

TWELVE_INTERVALS = {"15m": "15m", "1h": "1h", "4h": "4h", "1D": "1day"}
YF_INTERVALS = {"15m": ("15m", "5d"), "1h": ("1h", "60d"),
                "4h": ("1h", "60d"), "1D": ("1d", "max")}
OUTPUT_SIZES = {"15m": 400, "1h": 500, "4h": 500, "1D": 800}
KRAKEN_INTERVALS = {"15m": 15, "1h": 60, "4h": 240, "1D": 1440}


def get_live_gold_quote() -> tuple[float, str | None]:
    """Fetch the latest real spot XAU price without keys or fabricated candles."""
    url = f"{config.GOLD_API_BASE_URL.rstrip('/')}/price/XAU"
    resp = httpx.get(url, timeout=5.0)
    resp.raise_for_status()
    payload = resp.json()
    price = float(payload["price"])
    if price <= 0:
        raise RuntimeError("gold-api returned an invalid XAU/USD price")
    updated_at = payload.get("updatedAt")
    return price, str(updated_at) if updated_at else None


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns and make the index tz-naive UTC (seconds-based)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.capitalize).copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    out = df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last",
         "Volume": "sum"}
    )
    return finalize(out.dropna(subset=["Close"]))


class TwelveDataProvider:
    """Primary source: Twelve Data spot XAU/USD (requires TWELVE_DATA_API_KEY)."""

    name = "twelvedata"

    def available(self) -> bool:
        return bool(config.TWELVE_DATA_API_KEY)

    def supports(self, ticker: str) -> bool:
        return self.available() and ticker in (
            config.GOLD_TICKER, config.GOLD_FALLBACK_TICKER
        )

    def fetch(self, ticker: str, timeframe: str) -> pd.DataFrame:
        if not self.available():
            raise RuntimeError("TWELVE_DATA_API_KEY is not configured")
        if ticker not in (config.GOLD_TICKER, config.GOLD_FALLBACK_TICKER):
            raise RuntimeError("Twelve Data is only configured for XAU/USD")

        import httpx

        params = {
            "symbol": config.TWELVE_DATA_SYMBOL,
            "interval": TWELVE_INTERVALS[timeframe],
            "outputsize": OUTPUT_SIZES[timeframe],
            "order": "ASC",
            "format": "JSON",
            "apikey": config.TWELVE_DATA_API_KEY,
        }
        url = f"{config.TWELVE_DATA_BASE_URL}/time_series"
        try:
            resp = httpx.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Twelve Data request failed: {exc}") from exc

        if str(payload.get("status", "ok")).lower() != "ok" or payload.get("code"):
            raise RuntimeError(
                f"Twelve Data error {payload.get('code', '')}: {payload.get('message', 'unknown')}"
            )
        values = payload.get("values") or []
        if not values:
            raise RuntimeError("Twelve Data returned no candles")

        df = pd.DataFrame(values)
        df["Date"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("Date")
        out = finalize(df[["open", "high", "low", "close", "volume"]])
        if timeframe == "4h":
            out = _resample_4h(out)
        if out.empty:
            raise RuntimeError("Twelve Data returned no usable candles")
        logger.info("Twelve Data returned %d %s bars for %s", len(out), timeframe,
                    config.TWELVE_DATA_SYMBOL)
        return out


class YFinanceProvider:
    """Fallback source: Yahoo Finance spot XAU/USD, then COMEX gold futures."""

    name = "yfinance"

    def supports(self, ticker: str) -> bool:
        return True

    def _download(self, ticker: str, timeframe: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("yfinance is not installed") from exc
        interval, period = YF_INTERVALS[timeframe]
        raw = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=False, progress=False, threads=False, timeout=5,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {ticker}/{timeframe}")
        return finalize(pd.DataFrame(raw))

    def _chart_api(self, ticker: str, timeframe: str) -> pd.DataFrame:
        """Yahoo chart API over curl_cffi (browser TLS) with a short timeout."""
        from curl_cffi import requests as curl_requests

        interval, period = YF_INTERVALS[timeframe]
        yahoo_interval = {"15m": "15m", "1h": "60m", "1d": "1d"}.get(interval, interval)
        yahoo_range = {"5d": "5d", "60d": "3mo", "max": "10y"}.get(period, "1mo")
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
               f"?range={yahoo_range}&interval={yahoo_interval}&includePrePost=false")
        resp = curl_requests.get(url, impersonate="chrome", timeout=5)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        result = (payload.get("chart", {}).get("result")) or []
        if not result:
            err = payload.get("chart", {}).get("error")
            raise RuntimeError(f"Yahoo chart API error: {err}")
        r0 = result[0]
        ts = r0.get("timestamp") or []
        quote = (r0.get("indicators", {}).get("quote") or [{}])[0]
        if not ts or not quote:
            raise RuntimeError(f"Yahoo chart API returned no candles for {ticker}")
        df = pd.DataFrame(
            {k: quote.get(k) for k in ("open", "high", "low", "close", "volume")},
            index=pd.to_datetime(ts, unit="s", utc=True),
        )
        out = finalize(df)
        if timeframe == "4h":
            out = _resample_4h(out)
        if out.empty:
            raise RuntimeError(f"Yahoo chart API returned no usable candles for {ticker}")
        return out

    def fetch(self, ticker: str, timeframe: str) -> pd.DataFrame:
        candidates = [ticker]
        if ticker in (config.GOLD_TICKER, config.GOLD_FALLBACK_TICKER):
            candidates = [config.GOLD_TICKER, config.GOLD_FALLBACK_TICKER]

        last_exc: Exception | None = None
        for tk in candidates:
            for getter in (self._download, self._chart_api):
                try:
                    df = getter(tk, timeframe)
                    if len(df) >= 2:
                        logger.info("%s returned %d %s bars for %s", getter.__name__,
                                    len(df), timeframe, tk)
                        return df
                    raise RuntimeError(f"only {len(df)} bar returned for {tk}")
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.debug("%s failed for %s/%s: %s", getter.__name__, tk,
                                 timeframe, exc)
        raise RuntimeError(f"all yfinance sources failed: {last_exc}")


class YahooChartProvider:
    """Reliable keyless Yahoo chart API via httpx (no crumb/cookie requirement)."""

    name = "yahoo-chart"

    def supports(self, ticker: str) -> bool:
        return True

    def fetch(self, ticker: str, timeframe: str) -> pd.DataFrame:
        interval, period = YF_INTERVALS[timeframe]
        yahoo_interval = {"15m": "15m", "1h": "60m", "1d": "1d"}.get(interval, interval)
        yahoo_range = {"5d": "5d", "60d": "3mo", "max": "10y"}.get(period, "1mo")
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?range={yahoo_range}&interval={yahoo_interval}&includePrePost=false"
        )
        resp = httpx.get(url, timeout=12.0, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        result = (payload.get("chart", {}).get("result")) or []
        if not result:
            raise RuntimeError(f"Yahoo chart error: {payload.get('chart', {}).get('error')}")
        r0 = result[0]
        ts = r0.get("timestamp") or []
        quote = (r0.get("indicators", {}).get("quote") or [{}])[0]
        if not ts or not quote:
            raise RuntimeError(f"Yahoo chart returned no candles for {ticker}")
        df = pd.DataFrame(
            {k: quote.get(k) for k in ("open", "high", "low", "close", "volume")},
            index=pd.to_datetime(ts, unit="s", utc=True),
        )
        out = finalize(df)
        if timeframe == "4h":
            out = _resample_4h(out)
        if len(out) < 2:
            raise RuntimeError(f"Yahoo chart returned {len(out)} usable bars for {ticker}")
        logger.info("yahoo-chart returned %d %s bars for %s", len(out), timeframe, ticker)
        return out


class KrakenGoldProvider:
    """Keyless real gold candles: Kraken XAUT/USD (1 token = 1 troy oz of gold)."""

    name = "kraken-xaut"

    def supports(self, ticker: str) -> bool:
        return ticker in (config.GOLD_TICKER, config.GOLD_FALLBACK_TICKER)

    def fetch(self, ticker: str, timeframe: str) -> pd.DataFrame:
        if ticker not in (config.GOLD_TICKER, config.GOLD_FALLBACK_TICKER):
            raise RuntimeError("Kraken provider is only used for gold")
        url = f"{config.KRAKEN_BASE_URL}/0/public/OHLC"
        params = {"pair": config.KRAKEN_GOLD_PAIR,
                  "interval": KRAKEN_INTERVALS[timeframe]}
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        if payload.get("error"):
            raise RuntimeError(f"Kraken error: {payload['error']}")
        rows = next((v for v in payload.get("result", {}).values()
                     if isinstance(v, list)), None)
        if not rows:
            raise RuntimeError("Kraken returned no candles")
        df = pd.DataFrame(rows).iloc[:, :8]
        df.columns = ["Date", "Open", "High", "Low", "Close",
                      "Vwap", "Volume", "Count"]
        df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=True)
        out = finalize(df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]])
        if out.empty:
            raise RuntimeError("Kraken returned no usable candles")
        logger.info("Kraken returned %d %s bars for %s", len(out), timeframe,
                    config.KRAKEN_GOLD_PAIR)
        return out


PROVIDERS = (
    TwelveDataProvider(),
    KrakenGoldProvider(),
    YahooChartProvider(),
    YFinanceProvider(),
)
