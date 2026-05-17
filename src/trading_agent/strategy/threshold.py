from __future__ import annotations

from trading_agent.models import Prediction, Side, Signal


class ThresholdStrategy:
    def __init__(self, threshold: float = 0.02):
        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        self.threshold = threshold

    def generate_signal(self, prediction: Prediction) -> Signal:
        if prediction.direction_score > self.threshold:
            side = Side.BUY
        elif prediction.direction_score < -self.threshold:
            side = Side.SELL
        else:
            side = Side.HOLD

        return Signal(
            symbol=prediction.symbol,
            side=side,
            strength=prediction.confidence,
            reason=prediction.rationale,
        )
