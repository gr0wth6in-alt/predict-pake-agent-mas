from __future__ import annotations

from pathlib import Path

from trading_agent.features.indicators import clamp
from trading_agent.models import Candle, Prediction
from trading_agent.training.dataset import LABEL_BUY, LABEL_SELL, build_feature_vector
from trading_agent.training.naive_bayes import GaussianNaiveBayesModel


class TrainedModelPredictor:
    def __init__(self, model: GaussianNaiveBayesModel):
        self.model = model

    @property
    def min_history(self) -> int:
        return self.model.min_history

    @classmethod
    def load(cls, path: str | Path) -> TrainedModelPredictor:
        return cls(GaussianNaiveBayesModel.load(path))

    def predict(self, candles: list[Candle]) -> Prediction:
        if len(candles) < self.min_history + 1:
            raise ValueError(f"need at least {self.min_history + 1} candles")

        latest = candles[-1]
        features = build_feature_vector(candles, len(candles) - 1, self.model.config)
        probabilities = self.model.predict_proba(features)
        buy_probability = probabilities[LABEL_BUY]
        sell_probability = probabilities[LABEL_SELL]
        direction_score = clamp(buy_probability - sell_probability, -1.0, 1.0)
        confidence = max(probabilities.values())
        rationale = (
            f"ml_probs sell={sell_probability:.3f}, "
            f"hold={probabilities['hold']:.3f}, buy={buy_probability:.3f}"
        )

        return Prediction(
            symbol=latest.symbol,
            direction_score=direction_score,
            confidence=confidence,
            horizon_candles=self.model.config.horizon,
            rationale=rationale,
        )
