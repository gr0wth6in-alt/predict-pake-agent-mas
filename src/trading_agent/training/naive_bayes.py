from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trading_agent.training.dataset import FEATURE_NAMES, LABELS, TrainingConfig, TrainingSample


MODEL_KIND = "gaussian_naive_bayes_v1"
MIN_VARIANCE = 1e-9


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    total: int
    correct: int
    confusion_matrix: dict[str, dict[str, int]]


class GaussianNaiveBayesModel:
    def __init__(
        self,
        *,
        symbol: str,
        feature_names: list[str],
        labels: list[str],
        priors: dict[str, float],
        means: dict[str, list[float]],
        variances: dict[str, list[float]],
        config: TrainingConfig,
        metrics: dict[str, object],
        trained_at: str | None = None,
    ):
        self.symbol = symbol
        self.feature_names = feature_names
        self.labels = labels
        self.priors = priors
        self.means = means
        self.variances = variances
        self.config = config
        self.metrics = metrics
        self.trained_at = trained_at or datetime.now(UTC).isoformat()

    @property
    def min_history(self) -> int:
        return self.config.lookback

    @classmethod
    def fit(
        cls,
        samples: list[TrainingSample],
        *,
        symbol: str,
        config: TrainingConfig,
        metrics: dict[str, object] | None = None,
    ) -> GaussianNaiveBayesModel:
        if not samples:
            raise ValueError("cannot train model without samples")

        labels = [label for label in LABELS if any(sample.label == label for sample in samples)]
        feature_count = len(samples[0].features)
        priors: dict[str, float] = {}
        means: dict[str, list[float]] = {}
        variances: dict[str, list[float]] = {}

        for label in labels:
            label_samples = [sample for sample in samples if sample.label == label]
            priors[label] = len(label_samples) / len(samples)
            columns = [
                [sample.features[feature_index] for sample in label_samples]
                for feature_index in range(feature_count)
            ]
            means[label] = [_mean(column) for column in columns]
            variances[label] = [_variance(column) for column in columns]

        return cls(
            symbol=symbol,
            feature_names=list(FEATURE_NAMES),
            labels=labels,
            priors=priors,
            means=means,
            variances=variances,
            config=config,
            metrics=metrics or {},
        )

    def predict_proba(self, features: list[float]) -> dict[str, float]:
        if len(features) != len(self.feature_names):
            raise ValueError("feature vector length does not match the trained model")

        log_scores = {
            label: math.log(self.priors[label]) + self._log_likelihood(label, features)
            for label in self.labels
        }
        max_log_score = max(log_scores.values())
        exp_scores = {
            label: math.exp(log_score - max_log_score) for label, log_score in log_scores.items()
        }
        total = sum(exp_scores.values())
        probabilities = {label: exp_scores[label] / total for label in self.labels}

        for label in LABELS:
            probabilities.setdefault(label, 0.0)
        return probabilities

    def predict_label(self, features: list[float]) -> str:
        probabilities = self.predict_proba(features)
        return max(probabilities, key=probabilities.get)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GaussianNaiveBayesModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config_payload = payload["training_config"]
        return cls(
            symbol=payload["symbol"],
            feature_names=list(payload["feature_names"]),
            labels=list(payload["labels"]),
            priors={label: float(value) for label, value in payload["priors"].items()},
            means={label: list(values) for label, values in payload["means"].items()},
            variances={label: list(values) for label, values in payload["variances"].items()},
            config=TrainingConfig(
                lookback=int(config_payload["lookback"]),
                horizon=int(config_payload["horizon"]),
                label_threshold=float(config_payload["label_threshold"]),
            ),
            metrics=dict(payload.get("metrics", {})),
            trained_at=payload["trained_at"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": MODEL_KIND,
            "symbol": self.symbol,
            "trained_at": self.trained_at,
            "feature_names": self.feature_names,
            "labels": self.labels,
            "priors": self.priors,
            "means": self.means,
            "variances": self.variances,
            "training_config": {
                "lookback": self.config.lookback,
                "horizon": self.config.horizon,
                "label_threshold": self.config.label_threshold,
            },
            "metrics": self.metrics,
        }

    def _log_likelihood(self, label: str, features: list[float]) -> float:
        log_likelihood = 0.0
        for value, mean, variance in zip(features, self.means[label], self.variances[label]):
            variance = max(variance, MIN_VARIANCE)
            log_likelihood += -0.5 * math.log(2 * math.pi * variance)
            log_likelihood += -((value - mean) ** 2) / (2 * variance)
        return log_likelihood


def evaluate_model(
    model: GaussianNaiveBayesModel,
    samples: list[TrainingSample],
) -> EvaluationMetrics:
    confusion_matrix = {
        actual: {predicted: 0 for predicted in LABELS}
        for actual in LABELS
    }
    correct = 0

    for sample in samples:
        predicted = model.predict_label(sample.features)
        confusion_matrix[sample.label][predicted] += 1
        if predicted == sample.label:
            correct += 1

    total = len(samples)
    accuracy = correct / total if total else 0.0
    return EvaluationMetrics(
        accuracy=accuracy,
        total=total,
        correct=correct,
        confusion_matrix=confusion_matrix,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return MIN_VARIANCE
    mean = _mean(values)
    return max(sum((value - mean) ** 2 for value in values) / len(values), MIN_VARIANCE)
