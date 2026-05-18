from __future__ import annotations

from dataclasses import dataclass

from trading_agent.features.indicators import (
    IndicatorSnapshot,
    clamp,
    compute_indicator_snapshot,
)
from trading_agent.models import Candle, Prediction


@dataclass(frozen=True)
class IndicatorVote:
    name: str
    score: float
    detail: str


class MultiIndicatorPredictor:
    """Combines RSI, MACD, EMA cross and Bollinger %B into a single weighted score.

    Each indicator votes in [-1, 1]. The final direction score is the weighted average,
    so the predictor stays inspectable and dependency-light while reacting to more than
    just a moving-average cross.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "rsi": 1.0,
        "macd": 1.0,
        "ema_cross": 1.0,
        "bollinger": 0.75,
        "momentum": 0.75,
    }

    def __init__(
        self,
        *,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        weights: dict[str, float] | None = None,
        horizon_candles: int = 1,
    ):
        if not 0 < rsi_oversold < rsi_overbought < 100:
            raise ValueError("rsi thresholds must satisfy 0 < oversold < overbought < 100")

        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.weights = dict(weights) if weights is not None else dict(self.DEFAULT_WEIGHTS)
        self.horizon_candles = horizon_candles

    @property
    def min_history(self) -> int:
        # Bollinger and EMA cross want 26 + a buffer; MACD wants 26 + 9. Use the larger.
        return 35

    def predict(self, candles: list[Candle]) -> Prediction:
        if len(candles) < self.min_history:
            raise ValueError(f"need at least {self.min_history} candles")

        closes = [candle.close for candle in candles]
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]
        snapshot = compute_indicator_snapshot(closes, highs, lows)
        votes = self._collect_votes(snapshot)

        weighted_sum = 0.0
        weight_total = 0.0
        for vote in votes:
            weight = self.weights.get(vote.name, 0.0)
            if weight <= 0:
                continue
            weighted_sum += weight * vote.score
            weight_total += weight

        direction_score = clamp(weighted_sum / weight_total, -1.0, 1.0) if weight_total > 0 else 0.0
        confidence = clamp(abs(direction_score), 0.0, 1.0)

        rationale = "; ".join(f"{vote.name}={vote.detail}" for vote in votes) or "no indicator votes"

        return Prediction(
            symbol=candles[-1].symbol,
            direction_score=direction_score,
            confidence=confidence,
            horizon_candles=self.horizon_candles,
            rationale=rationale,
        )

    def _collect_votes(self, snapshot: IndicatorSnapshot) -> list[IndicatorVote]:
        votes: list[IndicatorVote] = []

        if snapshot.rsi_14 is not None:
            if snapshot.rsi_14 <= self.rsi_oversold:
                score = clamp((self.rsi_oversold - snapshot.rsi_14) / self.rsi_oversold, 0.0, 1.0)
            elif snapshot.rsi_14 >= self.rsi_overbought:
                top_room = max(100.0 - self.rsi_overbought, 1e-9)
                score = -clamp((snapshot.rsi_14 - self.rsi_overbought) / top_room, 0.0, 1.0)
            else:
                # Linear scale across the neutral zone.
                midpoint = (self.rsi_overbought + self.rsi_oversold) / 2
                half_range = (self.rsi_overbought - self.rsi_oversold) / 2
                score = clamp((snapshot.rsi_14 - midpoint) / half_range * 0.3, -1.0, 1.0)
            votes.append(IndicatorVote("rsi", score, f"{snapshot.rsi_14:.1f}->score={score:.2f}"))

        if snapshot.macd_histogram is not None and snapshot.close > 0:
            normalized = clamp(snapshot.macd_histogram / snapshot.close * 200.0, -1.0, 1.0)
            votes.append(IndicatorVote("macd", normalized, f"hist={snapshot.macd_histogram:.4f}"))

        if snapshot.ema_12 is not None and snapshot.ema_26 is not None and snapshot.ema_26 > 0:
            spread = (snapshot.ema_12 - snapshot.ema_26) / snapshot.ema_26
            score = clamp(spread * 25.0, -1.0, 1.0)
            votes.append(IndicatorVote("ema_cross", score, f"spread={spread:.4f}"))

        if snapshot.bollinger_percent is not None:
            # Inverse: low %B (near lower band) is bullish, high %B is bearish.
            score = clamp((0.5 - snapshot.bollinger_percent) * 2.0, -1.0, 1.0)
            votes.append(
                IndicatorVote("bollinger", score, f"%B={snapshot.bollinger_percent:.2f}")
            )

        if snapshot.return_5 is not None:
            score = clamp(snapshot.return_5 * 10.0, -1.0, 1.0)
            votes.append(IndicatorVote("momentum", score, f"return_5={snapshot.return_5:.4f}"))

        return votes
