"""Self-training orchestrator.

Pulls fresh historical data from a chosen feed, trains the Gaussian Naive Bayes model
and saves it as JSON. Designed to be runnable both as a one-shot CLI command and as a
periodic loop (the agent retraining itself on a schedule).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator

from trading_agent.data.binance_feed import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    limit_for_days,
    load_binance_klines,
)
from trading_agent.data.coingecko_feed import DEFAULT_DAYS, DEFAULT_VS_CURRENCY, load_coingecko_ohlc
from trading_agent.data.csv_feed import load_candles
from trading_agent.models import Candle
from trading_agent.training.dataset import TrainingConfig
from trading_agent.training.trainer import TrainingResult, train_and_save


@dataclass(frozen=True)
class AutoTrainConfig:
    symbol: str
    output_path: str
    data_source: str = "binance"  # one of: binance, coingecko, csv
    interval: str = DEFAULT_INTERVAL
    limit: int = DEFAULT_LIMIT
    days: int = DEFAULT_DAYS
    vs_currency: str = DEFAULT_VS_CURRENCY
    coin_id: str | None = None
    csv_path: str | None = None
    lookback: int = 10
    horizon: int = 3
    label_threshold: float = 0.005
    train_fraction: float = 0.8

    def to_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            lookback=self.lookback,
            horizon=self.horizon,
            label_threshold=self.label_threshold,
        )


@dataclass(frozen=True)
class AutoTrainReport:
    started_at: str
    finished_at: str
    config: AutoTrainConfig
    result: TrainingResult
    candle_count: int
    output_path: str

    def summary(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "symbol": self.config.symbol,
            "data_source": self.config.data_source,
            "candle_count": self.candle_count,
            "output_path": self.output_path,
            "samples": self.result.sample_count,
            "train_samples": self.result.train_count,
            "test_samples": self.result.test_count,
            "label_distribution": self.result.label_distribution,
            "train_accuracy": self.result.train_metrics.accuracy,
            "test_accuracy": self.result.test_metrics.accuracy,
            "warnings": list(self.result.warnings),
        }


def fetch_candles(config: AutoTrainConfig) -> list[Candle]:
    source = config.data_source.lower()
    if source == "binance":
        klines_limit = (
            limit_for_days(config.days, config.interval) if config.days else config.limit
        )
        return load_binance_klines(
            symbol=config.symbol,
            interval=config.interval,
            limit=klines_limit,
        )
    if source == "coingecko":
        return load_coingecko_ohlc(
            symbol=config.symbol,
            coin_id=config.coin_id,
            vs_currency=config.vs_currency,
            days=config.days,
        )
    if source == "csv":
        if not config.csv_path:
            raise ValueError("csv_path is required when data_source is 'csv'")
        return load_candles(Path(config.csv_path), config.symbol)
    raise ValueError(f"unsupported data_source: {config.data_source}")


def auto_train_once(config: AutoTrainConfig) -> AutoTrainReport:
    started_at = datetime.now(UTC).isoformat()
    candles = fetch_candles(config)
    if not candles:
        raise ValueError("no candles returned from data source")

    result = train_and_save(
        candles,
        symbol=config.symbol,
        config=config.to_training_config(),
        output_path=config.output_path,
        train_fraction=config.train_fraction,
    )
    finished_at = datetime.now(UTC).isoformat()
    return AutoTrainReport(
        started_at=started_at,
        finished_at=finished_at,
        config=config,
        result=result,
        candle_count=len(candles),
        output_path=str(config.output_path),
    )


def auto_train_loop(
    config: AutoTrainConfig,
    *,
    interval_seconds: float,
    iterations: int | None = None,
    sleep: object = time.sleep,
) -> Iterator[AutoTrainReport]:
    """Yield a report each time the model is retrained.

    ``iterations`` of ``None`` runs forever (use Ctrl+C). Tests can pass a fake
    ``sleep`` to avoid real waiting.
    """

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    count = 0
    while iterations is None or count < iterations:
        yield auto_train_once(config)
        count += 1
        if iterations is not None and count >= iterations:
            break
        sleep(interval_seconds)


def chain_reports(reports: Iterable[AutoTrainReport]) -> list[dict[str, object]]:
    """Materialize summaries from an iterator. Useful for batch CLI output."""

    return [report.summary() for report in reports]
