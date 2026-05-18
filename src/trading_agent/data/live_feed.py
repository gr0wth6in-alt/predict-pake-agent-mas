"""Live market data orchestration.

This module wraps the existing Binance and CoinGecko adapters into a single
"live" entry point that returns recent candles plus a snapshot of the latest
ticker. Binance is preferred (volume, fast 24h ticker, high rate limit). When
Binance is unreachable from the deploy host (HTTP 451 on PythonAnywhere is the
common case) we fall back to CoinGecko OHLC.

Two safety nets are layered on top of the providers:

1. **Result caching.** Each symbol's snapshot is stored briefly so a UI that
   polls every few seconds does not actually hit upstream every call. CoinGecko
   has a tighter free-tier rate limit so its TTL is longer than Binance's.
2. **Stale fallback.** When both providers fail right now but we still have a
   recent snapshot in memory, the cached snapshot is returned with a clear
   ``fallbacks=["stale-cache: ..."]`` marker. Better than a hard 400 every time
   CoinGecko throws 429 from a shared IP pool.
3. **Synthetic ticker from candles.** If we have candles but the ticker endpoint
   fails on its own (rare for Binance, common for CoinGecko free tier), the
   ticker is rebuilt from the last candle so the dashboard still moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic
from typing import Any

import httpx

from trading_agent.data.binance_feed import (
    BINANCE_BASE_URLS,
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    limit_for_days,
    load_binance_klines,
)
from trading_agent.data.coingecko_feed import (
    DEFAULT_DAYS,
    DEFAULT_VS_CURRENCY,
    coin_id_for_symbol,
    load_coingecko_ohlc,
)
from trading_agent.models import Candle


BINANCE_CACHE_TTL_SECONDS = 30.0
COINGECKO_CACHE_TTL_SECONDS = 90.0
STALE_FALLBACK_MAX_AGE_SECONDS = 300.0


@dataclass(frozen=True)
class LiveTicker:
    symbol: str
    last_price: float
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    volume_24h: float | None = None
    change_24h_pct: float | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "volume_24h": self.volume_24h,
            "change_24h_pct": self.change_24h_pct,
            "source": self.source,
        }


@dataclass(frozen=True)
class LiveMarketSnapshot:
    candles: list[Candle]
    ticker: LiveTicker
    source: str
    fallbacks: list[str] = field(default_factory=list)


_CACHE: dict[tuple[str, str, int | None], tuple[float, LiveMarketSnapshot]] = {}


def fetch_live_market(
    *,
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    days: int | None = None,
    vs_currency: str = DEFAULT_VS_CURRENCY,
    coin_id: str | None = None,
    timeout_seconds: float = 12.0,
) -> LiveMarketSnapshot:
    """Return recent candles plus a current ticker.

    Order of attempts: Binance (with caching), CoinGecko (with caching), stale
    cache. Raises ``ValueError`` only when nothing usable can be returned.
    """

    cache_key = (symbol.strip().upper(), interval, days)
    fallbacks: list[str] = []
    klines_limit = limit_for_days(days, interval) if days else limit
    days_for_coingecko = days if days else 30

    cached = _read_cache(cache_key)
    if cached is not None and cached[0] == "binance" and not _is_expired(
        cached[1], BINANCE_CACHE_TTL_SECONDS
    ):
        return cached[2]
    if cached is not None and cached[0] == "coingecko" and not _is_expired(
        cached[1], COINGECKO_CACHE_TTL_SECONDS
    ):
        return cached[2]

    binance_snapshot = _try_binance(
        symbol=symbol,
        interval=interval,
        klines_limit=klines_limit,
        timeout_seconds=timeout_seconds,
        fallbacks=fallbacks,
    )
    if binance_snapshot is not None:
        _write_cache(cache_key, "binance", binance_snapshot)
        return _with_fallbacks(binance_snapshot, fallbacks)

    coingecko_snapshot = _try_coingecko(
        symbol=symbol,
        coin_id=coin_id,
        vs_currency=vs_currency,
        days=days_for_coingecko,
        timeout_seconds=timeout_seconds,
        fallbacks=fallbacks,
    )
    if coingecko_snapshot is not None:
        _write_cache(cache_key, "coingecko", coingecko_snapshot)
        return _with_fallbacks(coingecko_snapshot, fallbacks)

    if cached is not None:
        cached_at, cached_snapshot = cached[1], cached[2]
        age = monotonic() - cached_at
        if age <= STALE_FALLBACK_MAX_AGE_SECONDS:
            stale_message = (
                f"stale-cache: {age:.0f}s old from {cached[0]}; both providers "
                "failed. Switch the backend to a host that allows Binance for fresh data."
            )
            return _with_fallbacks(cached_snapshot, fallbacks + [stale_message])

    raise ValueError(
        "live market data unavailable from configured providers: " + " | ".join(fallbacks)
    )


def _try_binance(
    *,
    symbol: str,
    interval: str,
    klines_limit: int,
    timeout_seconds: float,
    fallbacks: list[str],
) -> LiveMarketSnapshot | None:
    try:
        candles = load_binance_klines(symbol=symbol, interval=interval, limit=klines_limit)
    except (httpx.HTTPError, ValueError) as exc:
        fallbacks.append(f"binance: {exc}")
        return None

    try:
        ticker = _fetch_binance_ticker(symbol, timeout_seconds=timeout_seconds)
    except (httpx.HTTPError, ValueError) as exc:
        ticker = _ticker_from_candles(candles, source="binance-candles")
        fallbacks.append(f"binance-ticker: {exc}; using candle-derived ticker")
    return LiveMarketSnapshot(candles=candles, ticker=ticker, source="binance")


def _try_coingecko(
    *,
    symbol: str,
    coin_id: str | None,
    vs_currency: str,
    days: int,
    timeout_seconds: float,
    fallbacks: list[str],
) -> LiveMarketSnapshot | None:
    try:
        candles = load_coingecko_ohlc(
            symbol=symbol,
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
        )
    except (httpx.HTTPError, ValueError) as exc:
        fallbacks.append(f"coingecko: {exc}")
        return None

    try:
        ticker = _fetch_coingecko_ticker(
            symbol=symbol,
            coin_id=coin_id,
            vs_currency=vs_currency,
            timeout_seconds=timeout_seconds,
        )
    except (httpx.HTTPError, ValueError) as exc:
        ticker = _ticker_from_candles(candles, source="coingecko-candles")
        fallbacks.append(f"coingecko-ticker: {exc}; using candle-derived ticker")
    return LiveMarketSnapshot(candles=candles, ticker=ticker, source="coingecko")


def _ticker_from_candles(candles: list[Candle], *, source: str) -> LiveTicker:
    """Build a ``LiveTicker`` from the most recent candle.

    Used when a provider returns OHLC fine but the dedicated ticker endpoint is
    rate-limited or blocked. The 24h fields are best-effort: high/low come from
    the last 24 candles when we have hourly data, otherwise from the last candle.
    """

    if not candles:
        raise ValueError("cannot derive a ticker from an empty candle list")

    latest = candles[-1]
    window = candles[-min(24, len(candles)) :]
    high_24h = max(candle.high for candle in window)
    low_24h = min(candle.low for candle in window)
    open_24h = window[0].open
    change_24h_pct: float | None = None
    if open_24h > 0:
        change_24h_pct = (latest.close - open_24h) / open_24h * 100.0
    volume_24h = sum(candle.volume for candle in window) or None

    return LiveTicker(
        symbol=latest.symbol,
        last_price=latest.close,
        timestamp=latest.timestamp,
        high_24h=high_24h,
        low_24h=low_24h,
        volume_24h=volume_24h,
        change_24h_pct=change_24h_pct,
        source=source,
    )


def _fetch_binance_ticker(symbol: str, *, timeout_seconds: float) -> LiveTicker:
    normalized = symbol.strip().upper()
    last_error: Exception | None = None

    for base_url in BINANCE_BASE_URLS:
        try:
            response = httpx.get(
                f"{base_url}/api/v3/ticker/24hr",
                params={"symbol": normalized},
                headers={"accept": "application/json"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            close_time_ms = float(payload.get("closeTime", 0)) or float(payload.get("openTime", 0))
            timestamp = (
                datetime.fromtimestamp(close_time_ms / 1000)
                if close_time_ms
                else datetime.utcnow()
            )
            return LiveTicker(
                symbol=normalized,
                last_price=float(payload["lastPrice"]),
                timestamp=timestamp,
                bid=_safe_float(payload.get("bidPrice")),
                ask=_safe_float(payload.get("askPrice")),
                high_24h=_safe_float(payload.get("highPrice")),
                low_24h=_safe_float(payload.get("lowPrice")),
                volume_24h=_safe_float(payload.get("volume")),
                change_24h_pct=_safe_float(payload.get("priceChangePercent")),
                source="binance",
            )
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            last_error = exc

    raise ValueError(f"binance ticker unavailable: {last_error}")


def _fetch_coingecko_ticker(
    *,
    symbol: str,
    coin_id: str | None,
    vs_currency: str,
    timeout_seconds: float,
) -> LiveTicker:
    resolved_coin_id = coin_id or coin_id_for_symbol(symbol)
    response = httpx.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": resolved_coin_id,
            "vs_currencies": vs_currency.lower(),
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
            "include_last_updated_at": "true",
        },
        headers={"accept": "application/json"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or resolved_coin_id not in payload:
        raise ValueError(f"coingecko returned no ticker for {resolved_coin_id}")
    entry = payload[resolved_coin_id]
    last_price = _safe_float(entry.get(vs_currency.lower()))
    if last_price is None:
        raise ValueError(f"coingecko ticker missing price field for {resolved_coin_id}")

    last_updated = entry.get("last_updated_at")
    timestamp = (
        datetime.fromtimestamp(float(last_updated))
        if isinstance(last_updated, (int, float))
        else datetime.utcnow()
    )
    return LiveTicker(
        symbol=symbol.strip().upper(),
        last_price=last_price,
        timestamp=timestamp,
        volume_24h=_safe_float(entry.get(f"{vs_currency.lower()}_24h_vol")),
        change_24h_pct=_safe_float(entry.get(f"{vs_currency.lower()}_24h_change")),
        source="coingecko",
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _with_fallbacks(snapshot: LiveMarketSnapshot, fallbacks: list[str]) -> LiveMarketSnapshot:
    if not fallbacks:
        return snapshot
    return LiveMarketSnapshot(
        candles=snapshot.candles,
        ticker=snapshot.ticker,
        source=snapshot.source,
        fallbacks=list(snapshot.fallbacks) + fallbacks,
    )


def _read_cache(
    key: tuple[str, str, int | None],
) -> tuple[str, float, LiveMarketSnapshot] | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    cached_at, snapshot = entry
    return snapshot.source, cached_at, snapshot


def _write_cache(
    key: tuple[str, str, int | None],
    _source: str,
    snapshot: LiveMarketSnapshot,
) -> None:
    _CACHE[key] = (monotonic(), snapshot)


def _is_expired(cached_at: float, ttl_seconds: float) -> bool:
    return (monotonic() - cached_at) > ttl_seconds


__all__ = [
    "BINANCE_CACHE_TTL_SECONDS",
    "COINGECKO_CACHE_TTL_SECONDS",
    "DEFAULT_DAYS",
    "DEFAULT_INTERVAL",
    "DEFAULT_LIMIT",
    "DEFAULT_VS_CURRENCY",
    "LiveMarketSnapshot",
    "LiveTicker",
    "STALE_FALLBACK_MAX_AGE_SECONDS",
    "fetch_live_market",
]
