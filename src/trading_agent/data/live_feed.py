"""Live market data orchestration.

This module wraps the existing Binance and CoinGecko adapters to provide a single
"live" entry point that returns recent candles plus a snapshot of the latest ticker.
Binance is preferred because it includes volume and a 24h ticker; if that endpoint is
blocked from the deploy host (HTTP 451) we fall back to CoinGecko OHLC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    """Return recent candles plus a current ticker. Prefers Binance, falls back to CoinGecko."""

    fallbacks: list[str] = []
    klines_limit = limit_for_days(days, interval) if days else limit

    try:
        candles = load_binance_klines(symbol=symbol, interval=interval, limit=klines_limit)
        ticker = _fetch_binance_ticker(symbol, timeout_seconds=timeout_seconds)
        return LiveMarketSnapshot(candles=candles, ticker=ticker, source="binance")
    except (httpx.HTTPError, ValueError) as exc:
        fallbacks.append(f"binance: {exc}")

    days_to_use = days if days else 30
    try:
        candles = load_coingecko_ohlc(
            symbol=symbol,
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days_to_use,
        )
        ticker = _fetch_coingecko_ticker(
            symbol=symbol,
            coin_id=coin_id,
            vs_currency=vs_currency,
            timeout_seconds=timeout_seconds,
        )
        return LiveMarketSnapshot(
            candles=candles,
            ticker=ticker,
            source="coingecko",
            fallbacks=fallbacks,
        )
    except (httpx.HTTPError, ValueError) as exc:
        fallbacks.append(f"coingecko: {exc}")
        raise ValueError(
            "live market data unavailable from configured providers: " + " | ".join(fallbacks)
        ) from exc


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


__all__ = [
    "DEFAULT_DAYS",
    "DEFAULT_INTERVAL",
    "DEFAULT_LIMIT",
    "DEFAULT_VS_CURRENCY",
    "LiveMarketSnapshot",
    "LiveTicker",
    "fetch_live_market",
]
