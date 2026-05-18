"""Live ticker stream for all Binance USDT spot pairs.

To keep the project dependency-light we use REST polling instead of a WebSocket
connection. Binance exposes ``/api/v3/ticker/price`` which returns the latest price
for every spot symbol in one call. Polling that endpoint every few seconds gives
us a continuous "live" feed without adding ``websockets`` as a dependency.

Usage:

>>> from trading_agent.data.binance_stream import BinanceTickerStream
>>> stream = BinanceTickerStream(quote="USDT", interval_seconds=2.0)
>>> stream.start(on_tick=print)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Iterable

import httpx

from trading_agent.data.binance_feed import BINANCE_BASE_URLS


DEFAULT_INTERVAL_SECONDS = 2.0
TICKER_PATH = "/api/v3/ticker/price"


@dataclass(frozen=True)
class Tick:
    symbol: str
    price: float
    timestamp: datetime


TickHandler = Callable[[Tick], None]


class BinanceTickerStream:
    """Polls Binance for the latest price of every spot symbol on a fixed interval.

    The poller runs in a background thread when :meth:`start` is called. It filters
    by quote currency (default ``USDT``) and an optional ``symbols`` allowlist so the
    rest of the program only sees the markets it cares about.
    """

    def __init__(
        self,
        *,
        quote: str = "USDT",
        symbols: Iterable[str] | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        timeout_seconds: float = 8.0,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        self.quote = quote.upper()
        self.allowed_symbols = (
            {symbol.strip().upper() for symbol in symbols} if symbols is not None else None
        )
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._handlers: list[TickHandler] = []
        self._last_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, on_tick: TickHandler | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("stream already running")

        if on_tick is not None:
            self._handlers.append(on_tick)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="binance-stream", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_seconds)
            self._thread = None

    def add_handler(self, handler: TickHandler) -> None:
        self._handlers.append(handler)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def known_symbols(self) -> list[str]:
        return list(self._last_prices)

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                ticks = self._fetch_ticks()
                for tick in ticks:
                    self._last_prices[tick.symbol] = tick.price
                    self._dispatch(tick)
            except Exception as exc:  # noqa: BLE001 - log and keep streaming
                self._dispatch_error(exc)

            elapsed = time.monotonic() - started
            self._stop_event.wait(timeout=max(0.0, self.interval_seconds - elapsed))

    def _fetch_ticks(self) -> list[Tick]:
        last_error: Exception | None = None
        now = datetime.now(UTC)

        for base_url in BINANCE_BASE_URLS:
            try:
                response = httpx.get(
                    f"{base_url}{TICKER_PATH}",
                    headers={"accept": "application/json"},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise ValueError("unexpected ticker payload shape")
                ticks: list[Tick] = []
                for entry in payload:
                    if not isinstance(entry, dict):
                        continue
                    symbol = str(entry.get("symbol", "")).upper()
                    if not symbol or not symbol.endswith(self.quote):
                        continue
                    if self.allowed_symbols is not None and symbol not in self.allowed_symbols:
                        continue
                    try:
                        price = float(entry["price"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if price <= 0:
                        continue
                    ticks.append(Tick(symbol=symbol, price=price, timestamp=now))
                return ticks
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc

        raise RuntimeError(f"binance ticker stream unavailable: {last_error}")

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch(self, tick: Tick) -> None:
        for handler in list(self._handlers):
            try:
                handler(tick)
            except Exception as exc:  # noqa: BLE001 - never let one handler kill the loop
                self._dispatch_error(exc)

    def _dispatch_error(self, exc: BaseException) -> None:
        # Tag errors so they are obvious in logs but still keep the loop alive.
        message = f"[stream-error] {type(exc).__name__}: {exc}"
        for handler in list(self._handlers):
            try:
                # Handlers receive Tick instances normally; for errors we simply log
                # via the same callback contract by passing a synthetic tick is wrong,
                # so we just fall back to printing.
                pass
            except Exception:  # noqa: BLE001
                pass
        print(message)


__all__ = [
    "BinanceTickerStream",
    "DEFAULT_INTERVAL_SECONDS",
    "Tick",
    "TickHandler",
]
