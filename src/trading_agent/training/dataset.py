from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from trading_agent.features.indicators import percentage_change
from trading_agent.models import Candle


LABEL_SELL = "sell"
LABEL_HOLD = "hold"
LABEL_BUY = "buy"
LABELS = (LABEL_SELL, LABEL_HOLD, LABEL_BUY)

FEATURE_NAMES = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "sma_5_distance",
    "sma_10_distance",
    "sma_cross_5_10",
    "volatility_5",
    "volatility_10",
    "volume_ratio_10",
    "candle_body_pct",
    "range_pct",
]


@dataclass(frozen=True)
class TrainingConfig:
    lookback: int = 10
    horizon: int = 3
    label_threshold: float = 0.01

    def validate(self) -> None:
        if self.lookback < 10:
            raise ValueError("lookback must be at least 10")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.label_threshold <= 0:
            raise ValueError("label_threshold must be positive")


@dataclass(frozen=True)
class TrainingSample:
    timestamp: str
    features: list[float]
    label: str
    future_return: float


def build_training_samples(candles: list[Candle], config: TrainingConfig) -> list[TrainingSample]:
    config.validate()
    if len(candles) <= config.lookback + config.horizon:
        raise ValueError("not enough candles for the configured lookback and horizon")

    samples: list[TrainingSample] = []
    last_feature_index = len(candles) - config.horizon - 1

    for index in range(config.lookback, last_feature_index + 1):
        future_return = percentage_change(candles[index].close, candles[index + config.horizon].close)
        if future_return > config.label_threshold:
            label = LABEL_BUY
        elif future_return < -config.label_threshold:
            label = LABEL_SELL
        else:
            label = LABEL_HOLD

        samples.append(
            TrainingSample(
                timestamp=candles[index].timestamp.isoformat(),
                features=build_feature_vector(candles, index, config),
                label=label,
                future_return=future_return,
            )
        )

    return samples


def build_feature_vector(candles: list[Candle], index: int, config: TrainingConfig) -> list[float]:
    config.validate()
    if index < config.lookback:
        raise ValueError("index does not have enough historical candles")

    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]
    current = candles[index]

    returns = [
        _return_over(closes, index, 1),
        _return_over(closes, index, 3),
        _return_over(closes, index, 5),
        _return_over(closes, index, 10),
    ]
    sma_5 = _mean(closes[index - 4 : index + 1])
    sma_10 = _mean(closes[index - 9 : index + 1])
    volume_mean_10 = _mean(volumes[index - 9 : index + 1])
    rolling_returns_5 = [_return_over(closes, point, 1) for point in range(index - 4, index + 1)]
    rolling_returns_10 = [_return_over(closes, point, 1) for point in range(index - 9, index + 1)]

    return [
        *returns,
        _safe_ratio(current.close - sma_5, sma_5),
        _safe_ratio(current.close - sma_10, sma_10),
        _safe_ratio(sma_5 - sma_10, sma_10),
        pstdev(rolling_returns_5),
        pstdev(rolling_returns_10),
        _safe_ratio(current.volume, volume_mean_10) - 1.0,
        _safe_ratio(current.close - current.open, current.open),
        _safe_ratio(current.high - current.low, current.close),
    ]


def label_counts(samples: list[TrainingSample]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for sample in samples:
        counts[sample.label] += 1
    return counts


def _return_over(values: list[float], index: int, periods: int) -> float:
    return percentage_change(values[index - periods], values[index])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
