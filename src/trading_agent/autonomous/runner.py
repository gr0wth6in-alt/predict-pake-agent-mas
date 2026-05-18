"""Autonomous trading runner.

Glues four pieces together:

1. :class:`~trading_agent.data.binance_stream.BinanceTickerStream` produces a live
   feed of every USDT spot pair.
2. :class:`~trading_agent.broker.exchange_simulator.ExchangeSimulator` keeps the
   paper account state (cash, holdings, fees) and runs the matching engine for
   limit orders on every tick.
3. A predictor (multi-indicator, ML model, or Claude) decides BUY / SELL / HOLD
   for each watched symbol on a fixed interval.
4. A retraining thread fetches recent klines and retrains the JSON Naive Bayes
   model, so the agent literally trains itself while running.

Each watched symbol can use its own candle interval (``BTCUSDT@1m, ETHUSDT@5m``)
so a trader can mix scalping and swing timeframes in one run.

The runner is intentionally a single class with a few small threads. No async, no
event loops. Beginners can read it top to bottom.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Deque, Iterable

from trading_agent.broker.exchange_simulator import (
    DEFAULT_FEE_RATE,
    ExchangeSimulator,
    SIDE_BUY,
    SIDE_SELL,
    parse_symbol,
)
from trading_agent.config import Settings, load_settings
from trading_agent.data.binance_feed import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    limit_for_days,
    load_binance_klines,
)
from trading_agent.data.binance_stream import (
    DEFAULT_INTERVAL_SECONDS,
    BinanceTickerStream,
    Tick,
)
from trading_agent.models import Candle
from trading_agent.prediction.factory import (
    PREDICTOR_AUTO,
    build_predictor,
)
from trading_agent.prediction.protocols import Predictor
from trading_agent.training.auto import AutoTrainConfig, auto_train_once


@dataclass
class AutonomousConfig:
    """Configuration for one autonomous run.

    ``symbols`` lists which markets to watch. ``symbol_intervals`` may override the
    candle interval for individual markets, e.g. ``{"BTCUSDT": "1m", "ETHUSDT": "5m"}``.
    Anything not listed falls back to ``candle_interval``.
    """

    symbols: list[str]
    starting_cash: float = 10_000.0
    fee_rate: float = DEFAULT_FEE_RATE
    predictor_name: str = PREDICTOR_AUTO
    model_path: str = "models/btcusd_auto_nb.json"
    decision_interval_seconds: float = 30.0
    stream_interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    candle_window: int = 200
    candle_interval: str = DEFAULT_INTERVAL
    symbol_intervals: dict[str, str] = field(default_factory=dict)
    klines_limit: int = DEFAULT_LIMIT
    buy_threshold: float = 0.2
    sell_threshold: float = -0.2
    quote_per_trade: float = 200.0
    max_position_quote: float = 2_500.0
    self_train_enabled: bool = True
    self_train_interval_minutes: int = 60
    self_train_days: int = 30
    self_train_lookback: int = 10
    self_train_horizon: int = 3
    self_train_label_threshold: float = 0.005

    def base_for(self, symbol: str) -> str:
        return parse_symbol(symbol)[0]

    def interval_for(self, symbol: str) -> str:
        return self.symbol_intervals.get(symbol.strip().upper(), self.candle_interval)


@dataclass
class _SymbolState:
    candles: Deque[Candle]
    interval: str = DEFAULT_INTERVAL
    bucket_seconds: int = 3600
    last_decision_at: float = 0.0
    last_price: float = 0.0


@dataclass
class AutonomousStatus:
    started_at: str
    running: bool
    cash: float
    equity: float
    fills: int
    open_orders: int
    holdings: dict[str, float]
    last_decisions: dict[str, str]
    intervals: dict[str, str]
    last_retrain_summary: dict[str, object] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "running": self.running,
            "cash": self.cash,
            "equity": self.equity,
            "fills": self.fills,
            "open_orders": self.open_orders,
            "holdings": self.holdings,
            "last_decisions": self.last_decisions,
            "intervals": self.intervals,
            "last_retrain_summary": self.last_retrain_summary,
            "errors": list(self.errors),
        }


class AutonomousRunner:
    """One paper account driven by a predictor on a live ticker feed."""

    def __init__(
        self,
        config: AutonomousConfig,
        *,
        settings: Settings | None = None,
        exchange: ExchangeSimulator | None = None,
        stream: BinanceTickerStream | None = None,
    ):
        if not config.symbols:
            raise ValueError("config.symbols must not be empty")

        normalized = [symbol.strip().upper() for symbol in config.symbols]
        normalized_intervals = {
            symbol.strip().upper(): interval.strip()
            for symbol, interval in config.symbol_intervals.items()
        }
        # Validate: every override must match a watched symbol so users learn early
        # if they made a typo like "BTUSDT@1m".
        unknown = sorted(set(normalized_intervals) - set(normalized))
        if unknown:
            raise ValueError(
                f"symbol_intervals references symbols that are not watched: {unknown}"
            )

        self.config = AutonomousConfig(
            **{
                **config.__dict__,
                "symbols": normalized,
                "symbol_intervals": normalized_intervals,
            }
        )
        self.settings = settings or load_settings()

        self.exchange = exchange or ExchangeSimulator(
            starting_cash=self.config.starting_cash,
            fee_rate=self.config.fee_rate,
            quote="USDT",
        )
        self.stream = stream or BinanceTickerStream(
            quote="USDT",
            symbols=normalized,
            interval_seconds=self.config.stream_interval_seconds,
        )

        self._states: dict[str, _SymbolState] = {}
        for symbol in normalized:
            interval = self.config.interval_for(symbol)
            self._states[symbol] = _SymbolState(
                candles=deque(maxlen=self.config.candle_window),
                interval=interval,
                bucket_seconds=_interval_to_seconds(interval),
            )
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._decision_thread: threading.Thread | None = None
        self._train_thread: threading.Thread | None = None
        self._started_at: str | None = None
        self._last_decisions: dict[str, str] = {}
        self._last_retrain_summary: dict[str, object] | None = None
        self._errors: list[str] = []

        self._predictor: Predictor = self._build_predictor()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._decision_thread is not None and self._decision_thread.is_alive():
            raise RuntimeError("runner already started")

        self._stop_event.clear()
        self._started_at = datetime.now(UTC).isoformat()

        for symbol in self.config.symbols:
            self._warm_up_symbol(symbol)

        self.stream.add_handler(self._on_tick)
        self.stream.start()

        self._decision_thread = threading.Thread(
            target=self._decision_loop,
            name="autonomous-decision",
            daemon=True,
        )
        self._decision_thread.start()

        if self.config.self_train_enabled:
            self._train_thread = threading.Thread(
                target=self._self_train_loop,
                name="autonomous-self-train",
                daemon=True,
            )
            self._train_thread.start()

        self._log(
            "runner started "
            f"symbols={self._format_symbols()} cash={self.exchange.cash:.2f}"
        )

    def stop(self) -> None:
        self._stop_event.set()
        self.stream.stop()
        for thread in (self._decision_thread, self._train_thread):
            if thread is not None:
                thread.join(timeout=5.0)
        self._decision_thread = None
        self._train_thread = None
        self._log("runner stopped")

    def status(self) -> AutonomousStatus:
        snapshot = self.exchange.snapshot()
        holdings = {
            base: holding.quantity
            for base, holding in snapshot.holdings.items()
            if holding.quantity > 0
        }
        intervals = {symbol: state.interval for symbol, state in self._states.items()}
        return AutonomousStatus(
            started_at=self._started_at or "",
            running=self._decision_thread is not None and self._decision_thread.is_alive(),
            cash=snapshot.cash,
            equity=self.exchange.equity(),
            fills=snapshot.fills,
            open_orders=snapshot.open_orders,
            holdings=holdings,
            last_decisions=dict(self._last_decisions),
            intervals=intervals,
            last_retrain_summary=self._last_retrain_summary,
            errors=list(self._errors[-10:]),
        )

    def join(self, timeout: float | None = None) -> None:
        """Block until the decision thread exits. Useful for ``cli`` runs."""

        if self._decision_thread is None:
            return
        self._decision_thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Tick handling
    # ------------------------------------------------------------------

    def _on_tick(self, tick: Tick) -> None:
        if tick.symbol not in self._states:
            return

        # 1) feed the matching engine so any pending limit orders can fire
        self.exchange.on_tick(tick.symbol, tick.price)

        # 2) update the latest synthetic candle for indicators
        with self._lock:
            state = self._states[tick.symbol]
            state.last_price = tick.price
            self._update_synthetic_candle(state, tick)

    def _update_synthetic_candle(self, state: _SymbolState, tick: Tick) -> None:
        """Treat each tick as the close of the current candle bucket.

        The bucket length follows the symbol's configured interval, so a 1m symbol
        rolls over each minute and a 1h symbol rolls over each hour. When the
        bucket boundary is crossed we close the candle and start a new one.
        """

        if not state.candles:
            return

        latest = state.candles[-1]
        latest_bucket = _bucket_floor(latest.timestamp, state.bucket_seconds)
        tick_bucket = _bucket_floor(tick.timestamp, state.bucket_seconds)
        if tick_bucket == latest_bucket:
            updated = Candle(
                timestamp=latest.timestamp,
                symbol=latest.symbol,
                open=latest.open,
                high=max(latest.high, tick.price),
                low=min(latest.low, tick.price),
                close=tick.price,
                volume=latest.volume,
            )
            state.candles[-1] = updated
        else:
            state.candles.append(
                Candle(
                    timestamp=tick_bucket,
                    symbol=latest.symbol,
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    volume=0.0,
                )
            )

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    def _decision_loop(self) -> None:
        while not self._stop_event.is_set():
            for symbol in self.config.symbols:
                try:
                    self._decide_for(symbol)
                except Exception as exc:  # noqa: BLE001 - keep the loop running
                    self._record_error(f"decide {symbol}: {exc}")
            self._stop_event.wait(timeout=self.config.decision_interval_seconds)

    def _decide_for(self, symbol: str) -> None:
        with self._lock:
            state = self._states[symbol]
            candles = list(state.candles)
            interval = state.interval

        if len(candles) < self._predictor.min_history:
            self._last_decisions[symbol] = (
                f"warming-up [{interval}] ({len(candles)}/{self._predictor.min_history} candles)"
            )
            return

        prediction = self._predictor.predict(candles)
        latest_price = candles[-1].close

        if prediction.direction_score >= self.config.buy_threshold:
            action = self._maybe_buy(symbol, latest_price, prediction.rationale)
        elif prediction.direction_score <= self.config.sell_threshold:
            action = self._maybe_sell(symbol, latest_price, prediction.rationale)
        else:
            action = "hold"

        self._last_decisions[symbol] = (
            f"[{interval}] score={prediction.direction_score:+.3f} "
            f"conf={prediction.confidence:.2f} -> {action}"
        )

    def _maybe_buy(self, symbol: str, price: float, reason: str) -> str:
        base = self.config.base_for(symbol)
        holding = self.exchange.get_holding(base)
        position_notional = holding.quantity * price
        if position_notional >= self.config.max_position_quote:
            return "buy-skipped:position-cap"

        max_remaining_notional = self.config.max_position_quote - position_notional
        budget = min(self.config.quote_per_trade, max_remaining_notional, self.exchange.cash * 0.95)
        if budget <= 1.0:
            return "buy-skipped:no-budget"

        try:
            self.exchange.place_market(
                symbol=symbol,
                side=SIDE_BUY,
                quote_amount=budget,
                reason=f"score-buy: {reason[:80]}",
            )
            return "buy"
        except ValueError as exc:
            self._record_error(f"buy {symbol}: {exc}")
            return f"buy-failed:{exc}"

    def _maybe_sell(self, symbol: str, price: float, reason: str) -> str:
        base = self.config.base_for(symbol)
        holding = self.exchange.get_holding(base)
        if holding.quantity <= 0:
            return "sell-skipped:no-position"

        target_qty = min(holding.quantity, max(self.config.quote_per_trade / max(price, 1e-9), 0.0))
        if target_qty <= 0:
            return "sell-skipped:zero-qty"

        try:
            self.exchange.place_market(
                symbol=symbol,
                side=SIDE_SELL,
                quantity=target_qty,
                reason=f"score-sell: {reason[:80]}",
            )
            return "sell"
        except ValueError as exc:
            self._record_error(f"sell {symbol}: {exc}")
            return f"sell-failed:{exc}"

    # ------------------------------------------------------------------
    # Self-training loop
    # ------------------------------------------------------------------

    def _self_train_loop(self) -> None:
        wait_seconds = max(60.0, self.config.self_train_interval_minutes * 60.0)
        self._stop_event.wait(timeout=wait_seconds)

        while not self._stop_event.is_set():
            try:
                summary = self._retrain_once()
                self._last_retrain_summary = summary
                self._reload_predictor()
                self._log(
                    f"self-train ok: test_accuracy={summary.get('test_accuracy')}"
                    f" samples={summary.get('samples')}"
                )
            except Exception as exc:  # noqa: BLE001
                self._record_error(f"self-train: {exc}")
            self._stop_event.wait(timeout=wait_seconds)

    def _retrain_once(self) -> dict[str, object]:
        primary_symbol = self.config.symbols[0]
        train_config = AutoTrainConfig(
            symbol=primary_symbol,
            output_path=self.config.model_path,
            data_source="binance",
            interval=self.config.interval_for(primary_symbol),
            limit=self.config.klines_limit,
            days=self.config.self_train_days,
            lookback=self.config.self_train_lookback,
            horizon=self.config.self_train_horizon,
            label_threshold=self.config.self_train_label_threshold,
        )
        report = auto_train_once(train_config)
        return report.summary()

    def _reload_predictor(self) -> None:
        with self._lock:
            self._predictor = self._build_predictor()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_predictor(self) -> Predictor:
        return build_predictor(
            self.config.predictor_name,
            model_path=self.config.model_path,
            symbol=self.config.symbols[0],
            settings=self.settings,
        )

    def _warm_up_symbol(self, symbol: str) -> None:
        interval = self.config.interval_for(symbol)
        try:
            candles = load_binance_klines(
                symbol=symbol,
                interval=interval,
                limit=limit_for_days(self.config.self_train_days, interval)
                or self.config.klines_limit,
            )
        except (ValueError, RuntimeError) as exc:
            self._record_error(f"warm-up {symbol}@{interval}: {exc}")
            return

        with self._lock:
            state = self._states[symbol]
            state.candles.clear()
            for candle in candles[-self.config.candle_window :]:
                state.candles.append(candle)
            if state.candles:
                state.last_price = state.candles[-1].close
                self.exchange.on_tick(symbol, state.candles[-1].close)

        self._log(
            f"warm-up {symbol}@{interval}: "
            f"{len(self._states[symbol].candles)} candles loaded"
        )

    def _record_error(self, message: str) -> None:
        timestamped = f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}"
        self._errors.append(timestamped)
        self._log(f"[ERROR ] {message}")

    def _log(self, message: str) -> None:
        print(f"[runner ] {datetime.now(UTC).isoformat(timespec='seconds')} {message}")

    def _format_symbols(self) -> str:
        return ", ".join(f"{symbol}@{state.interval}" for symbol, state in self._states.items())


def run_until_interrupt(config: AutonomousConfig) -> None:
    """Convenience helper used by the CLI: start, then sleep until Ctrl+C."""

    runner = AutonomousRunner(config)
    runner.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("autonomous runner interrupted by user")
    finally:
        runner.stop()


def split_symbols(raw: Iterable[str] | str) -> tuple[list[str], dict[str, str]]:
    """Parse a CLI symbol list with optional ``@interval`` suffixes.

    Examples
    --------
    ``"BTCUSDT,ETHUSDT"`` -> (``["BTCUSDT", "ETHUSDT"]``, ``{}``)
    ``"BTCUSDT@1m, ETHUSDT@5m"`` -> (``["BTCUSDT", "ETHUSDT"]``, ``{"BTCUSDT":"1m","ETHUSDT":"5m"}``)
    """

    if isinstance(raw, str):
        parts = [item for item in raw.replace(",", " ").split() if item]
    else:
        parts = [str(item) for item in raw if str(item).strip()]

    symbols: list[str] = []
    intervals: dict[str, str] = {}
    for raw_part in parts:
        if "@" in raw_part:
            symbol_part, _, interval_part = raw_part.partition("@")
        else:
            symbol_part, interval_part = raw_part, ""
        symbol = symbol_part.strip().upper()
        if not symbol:
            continue
        symbols.append(symbol)
        if interval_part:
            intervals[symbol] = interval_part.strip()
    return symbols, intervals


def _interval_to_seconds(interval: str) -> int:
    """Convert a Binance interval string like ``1m``, ``5m``, ``1h`` or ``1d`` to seconds."""

    cleaned = interval.strip().lower()
    if not cleaned:
        return 60
    unit = cleaned[-1]
    try:
        value = int(cleaned[:-1])
    except ValueError as exc:
        raise ValueError(f"unsupported interval: {interval!r}") from exc
    if value <= 0:
        raise ValueError(f"interval value must be positive: {interval!r}")
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    if unit == "w":
        return value * 604_800
    raise ValueError(f"unsupported interval unit: {interval!r}")


def _bucket_floor(timestamp: datetime, bucket_seconds: int) -> datetime:
    """Round ``timestamp`` down to the start of its candle bucket."""

    base = datetime(1970, 1, 1, tzinfo=timestamp.tzinfo or UTC)
    delta = (timestamp - base).total_seconds()
    bucket_start_seconds = int(delta // bucket_seconds) * bucket_seconds
    return base + timedelta(seconds=bucket_start_seconds)


__all__ = [
    "AutonomousConfig",
    "AutonomousRunner",
    "AutonomousStatus",
    "run_until_interrupt",
    "split_symbols",
]
