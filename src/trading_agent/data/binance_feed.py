from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import monotonic

import httpx

from trading_agent.models import Candle


BINANCE_BASE_URLS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
DEFAULT_INTERVAL = "1h"
DEFAULT_LIMIT = 500
MAX_LIMIT = 1000

_CACHE: dict[tuple[str, str, int], tuple[float, list[Candle]]] = {}


def load_binance_klines(
    *,
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    timeout_seconds: float = 12.0,
    cache_ttl_seconds: float = 30.0,
) -> list[Candle]:
    normalized_symbol = symbol.strip().upper()
    normalized_limit = max(1, min(int(limit), MAX_LIMIT))
    cache_key = (normalized_symbol, interval, normalized_limit)
    cached = _CACHE.get(cache_key)

    if cached is not None:
        cached_at, candles = cached
        if monotonic() - cached_at <= cache_ttl_seconds:
            return list(candles)

    response = _get_with_fallback(
        "/api/v3/klines",
        params={
            "symbol": normalized_symbol,
            "interval": interval,
            "limit": normalized_limit,
        },
        timeout_seconds=timeout_seconds,
    )
    _raise_for_status(response)
    payload = response.json()

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Binance returned no kline data for {normalized_symbol}")

    candles = [_parse_kline_row(row, normalized_symbol) for row in payload]
    _CACHE[cache_key] = (monotonic(), candles)
    return list(candles)


def limit_for_days(days: int, interval: str = DEFAULT_INTERVAL) -> int:
    if days <= 0:
        return DEFAULT_LIMIT

    interval_hours = _interval_to_hours(interval)
    return min(MAX_LIMIT, max(1, int((timedelta(days=days).total_seconds() / 3600) / interval_hours)))


def _parse_kline_row(row: object, symbol: str) -> Candle:
    if not isinstance(row, list) or len(row) < 6:
        raise ValueError("Binance returned an invalid kline row")

    open_time_ms, open_price, high_price, low_price, close_price, volume = row[:6]
    return Candle(
        timestamp=datetime.fromtimestamp(float(open_time_ms) / 1000, tz=UTC),
        symbol=symbol,
        open=float(open_price),
        high=float(high_price),
        low=float(low_price),
        close=float(close_price),
        volume=float(volume),
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        raise ValueError("Binance rate limit reached (HTTP 429). Wait and retry with a smaller limit.")
    if response.status_code == 451:
        raise ValueError(
            "Binance rejected this server location (HTTP 451). "
            "Use CoinGecko or deploy the backend on a host allowed by Binance."
        )
    response.raise_for_status()


def _get_with_fallback(
    path: str,
    *,
    params: dict[str, object],
    timeout_seconds: float,
) -> httpx.Response:
    errors: list[str] = []

    for base_url in BINANCE_BASE_URLS:
        try:
            response = httpx.get(
                f"{base_url}{path}",
                params=params,
                headers={"accept": "application/json"},
                timeout=timeout_seconds,
            )
            _raise_for_status(response)
            return response
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"{base_url}: {exc}")

    raise ValueError(
        "Binance market data could not be reached from this server. "
        "If this happens on PythonAnywhere, use CoinGecko fallback or another Python host. "
        f"Last errors: {' | '.join(errors[-3:])}"
    )


def _interval_to_hours(interval: str) -> float:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value / 60
    if unit == "h":
        return float(value)
    if unit == "d":
        return value * 24
    raise ValueError(f"unsupported Binance interval: {interval}")
