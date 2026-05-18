from __future__ import annotations

import math
import unittest
from datetime import UTC, datetime, timedelta

from trading_agent.models import Candle
from trading_agent.prediction.factory import (
    PREDICTOR_BASELINE,
    PREDICTOR_MULTI,
    build_predictor,
)
from trading_agent.prediction.llm import ClaudePredictor, LLMConfigurationError
from trading_agent.prediction.multi_indicator import MultiIndicatorPredictor


def _make_candles(closes: list[float], symbol: str = "BTCUSD") -> list[Candle]:
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    for index, close in enumerate(closes):
        candles.append(
            Candle(
                timestamp=base_time + timedelta(hours=index),
                symbol=symbol,
                open=close * 0.999,
                high=close * 1.002,
                low=close * 0.998,
                close=close,
                volume=100.0,
            )
        )
    return candles


def _noisy_trend(start: float, slope: float, length: int, noise_amplitude: float = 1.5) -> list[float]:
    """Trend with small oscillations so RSI and Bollinger %B don't saturate."""

    return [start + slope * i + noise_amplitude * math.sin(i / 2.5) for i in range(length)]


class MultiIndicatorPredictorTests(unittest.TestCase):
    def test_uptrend_yields_bounded_score(self) -> None:
        closes = _noisy_trend(start=100.0, slope=0.3, length=80)
        prediction = MultiIndicatorPredictor().predict(_make_candles(closes))
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)
        self.assertGreaterEqual(prediction.direction_score, -1.0)
        self.assertLessEqual(prediction.direction_score, 1.0)
        self.assertIn("rsi", prediction.rationale)
        self.assertIn("ema_cross", prediction.rationale)

    def test_uptrend_with_pullback_resolves_bullish(self) -> None:
        # Strong overall uptrend with a small recent pullback so RSI stays in the
        # 50-70 range and EMA/momentum dominate the vote toward a positive score.
        closes = [100.0 + 0.4 * i for i in range(60)] + [124.0 - 0.05 * i for i in range(20)]
        prediction = MultiIndicatorPredictor().predict(_make_candles(closes))
        self.assertGreater(prediction.direction_score, 0.0)

    def test_downtrend_with_relief_resolves_bearish(self) -> None:
        closes = [200.0 - 0.4 * i for i in range(60)] + [176.0 + 0.05 * i for i in range(20)]
        prediction = MultiIndicatorPredictor().predict(_make_candles(closes))
        self.assertLess(prediction.direction_score, 0.0)

    def test_oscillating_market_stays_near_zero(self) -> None:
        closes = [100.0 + 5 * math.sin(i / 3) for i in range(80)]
        prediction = MultiIndicatorPredictor().predict(_make_candles(closes))
        self.assertLess(abs(prediction.direction_score), 0.6)

    def test_rejects_too_short_history(self) -> None:
        with self.assertRaises(ValueError):
            MultiIndicatorPredictor().predict(_make_candles([100.0] * 10))


class FactoryTests(unittest.TestCase):
    def test_baseline_selection(self) -> None:
        predictor = build_predictor(PREDICTOR_BASELINE)
        self.assertEqual(type(predictor).__name__, "MovingAverageMomentumPredictor")

    def test_multi_selection(self) -> None:
        predictor = build_predictor(PREDICTOR_MULTI)
        self.assertEqual(type(predictor).__name__, "MultiIndicatorPredictor")

    def test_unknown_predictor_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_predictor("does-not-exist")

    def test_auto_falls_back_when_no_model(self) -> None:
        predictor = build_predictor("auto", model_path="models/_does_not_exist_.json")
        self.assertEqual(type(predictor).__name__, "MultiIndicatorPredictor")


class ClaudePredictorTests(unittest.TestCase):
    def test_missing_api_key_raises_configuration_error(self) -> None:
        predictor = ClaudePredictor(api_key=None)
        # Force the api_key to be missing regardless of environment.
        predictor._api_key = None  # type: ignore[attr-defined]
        with self.assertRaises(LLMConfigurationError):
            predictor.predict(_make_candles([100.0 + i for i in range(80)]))

    def test_parse_response_extracts_json(self) -> None:
        text = (
            "Some chatter before the JSON. "
            '{"direction": "up", "confidence": 0.62, "rationale": "trend up"}'
        )
        parsed = ClaudePredictor._parse_response(text)
        self.assertEqual(parsed.direction, "up")
        self.assertAlmostEqual(parsed.confidence, 0.62)
        self.assertIn("trend", parsed.rationale)

    def test_parse_response_invalid_direction_falls_back_to_flat(self) -> None:
        parsed = ClaudePredictor._parse_response('{"direction": "sideways", "confidence": 0.4}')
        self.assertEqual(parsed.direction, "flat")
        self.assertAlmostEqual(parsed.confidence, 0.4)


if __name__ == "__main__":
    unittest.main()
