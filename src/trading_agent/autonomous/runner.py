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

The runner is intentionally a single class with a few small threads. No async, no
event loops. Beginners can read it top to bottom.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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
    """Configuration for one autonomous run."""

    symbols: list[str]                # which markets to watch and trade, e.g. ["BTCUSDT"]
    starting_cash: float = 10_000.0
    fee_rate: float = DEFAULT_FEE_RATE
    predictor_name: str = PREDICTOR_AUTO
    model_path: str = "models/btcusd_auto_nb.json"
    decision_interval_seconds: float = 30.0
    stream_interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    candle_window: int = 200          # how many recent candles we keep in memory
    candle_interval: str = DEFAULT_INTERVAL
    klines_limit: int = DEFAULT_LIMIT
    buy_threshold: float = 0.2        # direction_score > this => BUY
    sell_threshold: float = -0.2      # direction_score < this => SELL
    quote_per_trade: float = 200.0    # USD-equivalent size per market order
    max_position_quote: float = 2_500.0  # do not pile in past this notional per symbol
    self_train_enabled: bool = True
    self_train_interval_minutes: int = 60
    self_train_days: int = 30
    self_train_lookback: int = 10
    self_train_horizon: int = 3
    self_train_label_threshold: float = 0.005

    def base_for(self, symbol: str) -> str:
        return parse_symbol(symbol)[0]


@dataclass
class _SymbolState:
    candles: Deque[Candle]
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
        self.config = AutonomousConfig(**{**config.__dict__, "symbols": normalized})
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

        self._states: dict[str, _SymbolState] = {
            symbol: _SymbolState(candles=deque(maxlen=self.config.candle_window))
            for symbol in normalized
        }
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

        # Warm up: load some recent candles per symbol so indicators have history.
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
            f"runner started symbols={self.config.symbols} cash={self.exchange.cash:.2f}"
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
        return AutonomousStatus(
            started_at=self._started_at or "",
            running=self._decision_thread is not None and self._decision_thread.is_alive(),
            cash=snapshot.cash,
            equity=self.exchange.equity(),
            fills=snapshot.fills,
            open_orders=snapshot.open_orders,
            holdings=holdings,
            last_decisions=dict(self._last_decisions),
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
        """Treat each tick as the close of the current 1m candle.

        This keeps the live feed in sync with the indicator pipeline without waiting
        for a fresh klines fetch. When the minute rolls over, we close the candle
        and start a new one.
        """

        if not state.candles:
            return

        latest = state.candles[-1]
        bucket_start = latest.timestamp.replace(second=0, microsecond=0)
        tick_bucket = tick.timestamp.replace(second=0, microsecond=0)
        if tick_bucket == bucket_start:
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

        if len(candles) < self._predictor.min_history:
            self._last_decisions[symbol] = (
                f"warming-up ({len(candles)}/{self._predictor.min_history} candles)"
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
            f"score={prediction.direction_score:+.3f} conf={prediction.confidence:.2f} -> {action}"
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

        # Sell either a slice equal to quote_per_trade or the whole position when small.
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
        # First retrain happens after the first interval, so the agent is not
        # hammered at startup. A brief delay also lets the warm-up finish.
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
            interval=self.config.candle_interval,
            limit=self.config.klines_limit,
            days=self.config.self_train_days,
            lookback=self.config.self_train_lookback,
            horizon=self.config.self_train_horizon,
            label_threshold=self.config.self_train_label_threshold,
        )
        report = auto_train_once(train_config)
        return report.summary()

    def _reload_predictor(self) -> None:
        # Re-resolve the predictor so the trained-model branch picks up the new file.
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
        try:
            candles = load_binance_klines(
                symbol=symbol,
                interval=self.config.candle_interval,
                limit=limit_for_days(self.config.self_train_days, self.config.candle_interval)
                or self.config.klines_limit,
            )
        except (ValueError, RuntimeError) as exc:
            self._record_error(f"warm-up {symbol}: {exc}")
            return

        with self._lock:
            state = self._states[symbol]
            state.candles.clear()
            for candle in candles[-self.config.candle_window :]:
                state.candles.append(candle)
            if state.candles:
                state.last_price = state.candles[-1].close
                # seed the simulator's last price too so market orders can fill
                self.exchange.on_tick(symbol, state.candles[-1].close)

        self._log(f"warm-up {symbol}: {len(self._states[symbol].candles)} candles loaded")

    def _record_error(self, message: str) -> None:
        timestamped = f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}"
        self._errors.append(timestamped)
        self._log(f"[ERROR ] {message}")

    def _log(self, message: str) -> None:
        print(f"[runner ] {datetime.now(UTC).isoformat(timespec='seconds')} {message}")


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


def split_symbols(raw: Iterable[str] | str) -> list[str]:
    if isinstance(raw, str):
        parts = [item for item in raw.replace(",", " ").split() if item]
    else:
        parts = list(raw)
    return [Path(symbol).name.upper() for symbol in parts]


__all__ = [
    "AutonomousConfig",
    "AutonomousRunner",
    "AutonomousStatus",
    "run_until_interrupt",
    "split_symbols",
]
