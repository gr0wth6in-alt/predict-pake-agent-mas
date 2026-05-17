from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from trading_agent.models import Candle
from trading_agent.prediction.ml import TrainedModelPredictor
from trading_agent.training.dataset import TrainingConfig, build_training_samples
from trading_agent.training.naive_bayes import GaussianNaiveBayesModel
from trading_agent.training.trainer import train_and_save


def _mixed_candles(count: int) -> list[Candle]:
    start = datetime(2026, 1, 1)
    closes = []
    price = 100.0
    for index in range(count):
        cycle = index % 18
        if cycle < 6:
            price *= 1.012
        elif cycle < 12:
            price *= 0.988
        else:
            price *= 1.001
        closes.append(price)

    return [
        Candle(
            timestamp=start + timedelta(hours=index),
            symbol="BTCUSD",
            open=close * 0.998,
            high=close * 1.004,
            low=close * 0.996,
            close=close,
            volume=1000 + (index % 7) * 25,
        )
        for index, close in enumerate(closes)
    ]


class TrainingPipelineTests(unittest.TestCase):
    def test_build_training_samples(self) -> None:
        samples = build_training_samples(
            _mixed_candles(40),
            TrainingConfig(lookback=10, horizon=3, label_threshold=0.005),
        )

        self.assertGreater(len(samples), 0)
        self.assertEqual(len(samples[0].features), 12)

    def test_train_save_load_and_predict(self) -> None:
        candles = _mixed_candles(80)
        config = TrainingConfig(lookback=10, horizon=3, label_threshold=0.005)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.json"
            result = train_and_save(
                candles,
                symbol="BTCUSD",
                config=config,
                output_path=model_path,
            )

            self.assertTrue(model_path.exists())
            self.assertGreater(result.sample_count, 0)

            model = GaussianNaiveBayesModel.load(model_path)
            predictor = TrainedModelPredictor(model)
            prediction = predictor.predict(candles[:30])

            self.assertEqual(prediction.symbol, "BTCUSD")
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
