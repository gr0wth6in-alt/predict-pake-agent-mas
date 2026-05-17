from __future__ import annotations

from datetime import UTC, datetime
from time import monotonic

import httpx

from trading_agent.models import Candle


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
DEFAULT_VS_CURRENCY = "usd"
DEFAULT_DAYS = 30

SYMBOL_TO_COIN_ID = {
    "BTC": "bitcoin",
    "BTCUSD": "bitcoin",
    "BTCUSDT": "bitcoin",
    "ETH": "ethereum",
    "ETHUSD": "ethereum",
    "ETHUSDT": "ethereum",
    "SOL": "solana",
    "SOLUSD": "solana",
    "SOLUSDT": "solana",
    "BNB": "binancecoin",
    "BNBUSD": "binancecoin",
    "BNBUSDT": "binancecoin",
    "XRP": "ripple",
    "XRPUSD": "ripple",
    "XRPUSDT": "ripple",
    "DOGE": "dogecoin",
    "DOGEUSD": "dogecoin",
    "DOGEUSDT": "dogecoin",
}

_CACHE: dict[tuple[str, str, int], tuple[float, list[Candle]]] = {}
_COIN_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}


def coin_id_for_symbol(symbol: str) -> str:
    return SYMBOL_TO_COIN_ID.get(symbol.strip().upper(), symbol.strip().lower())


def load_coingecko_ohlc(
    *,
    symbol: str,
    coin_id: str | None = None,
    vs_currency: str = DEFAULT_VS_CURRENCY,
    days: int = DEFAULT_DAYS,
    timeout_seconds: float = 12.0,
    cache_ttl_seconds: float = 60.0,
) -> list[Candle]:
    resolved_coin_id = coin_id or coin_id_for_symbol(symbol)
    cache_key = (resolved_coin_id, vs_currency.lower(), days)
    cached = _CACHE.get(cache_key)

    if cached is not None:
        cached_at, candles = cached
        if monotonic() - cached_at <= cache_ttl_seconds:
            return list(candles)

    url = f"{COINGECKO_BASE_URL}/coins/{resolved_coin_id}/ohlc"
    response = httpx.get(
        url,
        params={"vs_currency": vs_currency.lower(), "days": days},
        headers={"accept": "application/json"},
        timeout=timeout_seconds,
    )
    _raise_for_status(response)
    payload = response.json()

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"CoinGecko returned no OHLC data for {resolved_coin_id}")

    candles = [_parse_ohlc_row(row, symbol.strip().upper()) for row in payload]
    _CACHE[cache_key] = (monotonic(), candles)
    return list(candles)


def search_coins(
    query: str = "",
    *,
    limit: int = 100,
    timeout_seconds: float = 12.0,
    cache_ttl_seconds: float = 3600.0,
) -> list[dict[str, str]]:
    normalized_query = query.strip().lower()
    cache_key = f"search:{normalized_query}:{limit}"
    cached = _COIN_CACHE.get(cache_key)

    if cached is not None:
        cached_at, coins = cached
        if monotonic() - cached_at <= cache_ttl_seconds:
            return list(coins)

    if normalized_query:
        response = httpx.get(
            f"{COINGECKO_BASE_URL}/search",
            params={"query": normalized_query},
            headers={"accept": "application/json"},
            timeout=timeout_seconds,
        )
        _raise_for_status(response)
        payload = response.json()
        raw_coins = payload.get("coins", []) if isinstance(payload, dict) else []
        coins = [
            {
                "id": str(coin.get("id", "")),
                "symbol": str(coin.get("symbol", "")).upper(),
                "name": str(coin.get("name", "")),
            }
            for coin in raw_coins[:limit]
            if isinstance(coin, dict) and coin.get("id")
        ]
    else:
        response = httpx.get(
            f"{COINGECKO_BASE_URL}/coins/list",
            headers={"accept": "application/json"},
            timeout=timeout_seconds,
        )
        _raise_for_status(response)
        payload = response.json()
        raw_coins = payload if isinstance(payload, list) else []
        coins = [
            {
                "id": str(coin.get("id", "")),
                "symbol": str(coin.get("symbol", "")).upper(),
                "name": str(coin.get("name", "")),
            }
            for coin in raw_coins[:limit]
            if isinstance(coin, dict) and coin.get("id")
        ]

    _COIN_CACHE[cache_key] = (monotonic(), coins)
    return list(coins)


def _parse_ohlc_row(row: object, symbol: str) -> Candle:
    if not isinstance(row, list) or len(row) < 5:
        raise ValueError("CoinGecko returned an invalid OHLC row")

    timestamp_ms, open_price, high_price, low_price, close_price = row[:5]
    return Candle(
        timestamp=datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=UTC),
        symbol=symbol,
        open=float(open_price),
        high=float(high_price),
        low=float(low_price),
        close=float(close_price),
        volume=0.0,
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")
        wait_hint = f" Wait about {retry_after} seconds before retrying." if retry_after else ""
        raise ValueError(
            "CoinGecko rate limit reached (HTTP 429). "
            "Try again later, reduce --days to 30, or train fewer coins at once."
            f"{wait_hint}"
        )

    response.raise_for_status()
