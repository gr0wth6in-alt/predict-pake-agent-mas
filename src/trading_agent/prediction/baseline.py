from __future__ import annotations

from trading_agent.features.indicators import clamp, simple_moving_average
from trading_agent.models import Candle, Prediction


class MovingAverageMomentumPredictor:
    """Small deterministic predictor that is easy to replace with an ML model."""

    def __init__(self, short_window: int = 5, long_window: int = 20, horizon_candles: int = 1):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("windows must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")

        self.short_window = short_window
        self.long_window = long_window
        self.horizon_candles = horizon_candles

    @property
    def min_history(self) -> int:
        return self.long_window

    def predict(self, candles: list[Candle]) -> Prediction:
        if len(candles) < self.min_history:
            raise ValueError(f"need at least {self.min_history} candles")

        closes = [candle.close for candle in candles]
        short_ma = simple_moving_average(closes, self.short_window)[-1]
        long_ma = simple_moving_average(closes, self.long_window)[-1]

        if short_ma is None or long_ma is None or long_ma == 0:
            raw_score = 0.0
        else:
            raw_score = (short_ma - long_ma) / long_ma

        direction_score = clamp(raw_score * 5.0, -1.0, 1.0)
        confidence = clamp(abs(direction_score), 0.0, 1.0)
        symbol = candles[-1].symbol
        rationale = (
            f"short_ma={short_ma:.4f}, long_ma={long_ma:.4f}, "
            f"score={direction_score:.4f}"
        )

        return Prediction(
            symbol=symbol,
            direction_score=direction_score,
            confidence=confidence,
            horizon_candles=self.horizon_candles,
            rationale=rationale,
        )
