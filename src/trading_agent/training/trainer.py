from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading_agent.models import Candle
from trading_agent.training.dataset import (
    TrainingConfig,
    TrainingSample,
    build_training_samples,
    label_counts,
)
from trading_agent.training.naive_bayes import (
    EvaluationMetrics,
    GaussianNaiveBayesModel,
    evaluate_model,
)


@dataclass(frozen=True)
class TrainingResult:
    model: GaussianNaiveBayesModel
    train_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    sample_count: int
    train_count: int
    test_count: int
    label_distribution: dict[str, int]
    warnings: list[str]


def train_supervised_model(
    candles: list[Candle],
    *,
    symbol: str,
    config: TrainingConfig,
    train_fraction: float = 0.8,
) -> TrainingResult:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    samples = build_training_samples(candles, config)
    if len(samples) < 5:
        raise ValueError("need at least 5 supervised samples to train")

    split_index = max(1, min(len(samples) - 1, int(len(samples) * train_fraction)))
    train_samples = samples[:split_index]
    test_samples = samples[split_index:]
    warnings = _build_warnings(samples, train_samples, test_samples)

    model = GaussianNaiveBayesModel.fit(
        train_samples,
        symbol=symbol,
        config=config,
    )
    train_metrics = evaluate_model(model, train_samples)
    test_metrics = evaluate_model(model, test_samples)

    metrics_payload = {
        "train": _metrics_to_dict(train_metrics),
        "test": _metrics_to_dict(test_metrics),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "label_distribution": label_counts(samples),
        "warnings": warnings,
    }
    model.metrics.update(metrics_payload)

    return TrainingResult(
        model=model,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        sample_count=len(samples),
        train_count=len(train_samples),
        test_count=len(test_samples),
        label_distribution=label_counts(samples),
        warnings=warnings,
    )


def train_and_save(
    candles: list[Candle],
    *,
    symbol: str,
    config: TrainingConfig,
    output_path: str | Path,
    train_fraction: float = 0.8,
) -> TrainingResult:
    result = train_supervised_model(
        candles,
        symbol=symbol,
        config=config,
        train_fraction=train_fraction,
    )
    result.model.save(output_path)
    return result


def _build_warnings(
    samples: list[TrainingSample],
    train_samples: list[TrainingSample],
    test_samples: list[TrainingSample],
) -> list[str]:
    all_counts = label_counts(samples)
    train_counts = label_counts(train_samples)
    test_counts = label_counts(test_samples)
    warnings: list[str] = []

    if sum(1 for count in all_counts.values() if count > 0) < 2:
        warnings.append("Only one label exists in the full dataset; use more varied market data.")
    if sum(1 for count in train_counts.values() if count > 0) < 2:
        warnings.append("Only one label exists in the training split; model will be biased.")
    if sum(1 for count in test_counts.values() if count > 0) < 2:
        warnings.append("Only one label exists in the test split; accuracy is not very informative.")

    return warnings


def _metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, object]:
    return {
        "accuracy": metrics.accuracy,
        "total": metrics.total,
        "correct": metrics.correct,
        "confusion_matrix": metrics.confusion_matrix,
    }
