from __future__ import annotations

from typing import Protocol

from trading_agent.models import Candle, Prediction


class Predictor(Protocol):
    @property
    def min_history(self) -> int:
        raise NotImplementedError

    def predict(self, candles: list[Candle]) -> Prediction:
        raise NotImplementedError
