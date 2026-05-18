from __future__ import annotations

import unittest

from trading_agent.features.indicators import (
    average_true_range,
    bollinger_bands,
    compute_indicator_snapshot,
    exponential_moving_average,
    macd,
    percentage_change,
    relative_strength_index,
    simple_moving_average,
)


class IndicatorTests(unittest.TestCase):
    def test_simple_moving_average(self) -> None:
        self.assertEqual(
            simple_moving_average([1, 2, 3, 4], 2),
            [None, 1.5, 2.5, 3.5],
        )

    def test_percentage_change(self) -> None:
        self.assertEqual(percentage_change(0, 10), 0.0)
        self.assertAlmostEqual(percentage_change(100, 110), 0.10)

    def test_exponential_moving_average_seeds_with_sma(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ema = exponential_moving_average(values, 3)
        self.assertEqual(ema[:2], [None, None])
        # SMA of first three values is 2.0; EMA then continues from there.
        self.assertAlmostEqual(ema[2] or 0.0, 2.0)
        self.assertGreater(ema[3] or 0.0, ema[2] or 0.0)
        self.assertGreater(ema[4] or 0.0, ema[3] or 0.0)

    def test_relative_strength_index_bounds(self) -> None:
        rising = list(range(1, 30))
        rsi = relative_strength_index([float(value) for value in rising], 14)
        last = rsi[-1]
        assert last is not None
        self.assertGreater(last, 70.0)
        self.assertLessEqual(last, 100.0)

    def test_macd_returns_three_aligned_series(self) -> None:
        values = [float(i) + (i % 5) for i in range(60)]
        macd_line, signal_line, histogram = macd(values)
        self.assertEqual(len(macd_line), len(values))
        self.assertEqual(len(signal_line), len(values))
        self.assertEqual(len(histogram), len(values))
        self.assertIsNotNone(macd_line[-1])
        self.assertIsNotNone(signal_line[-1])

    def test_bollinger_bands_envelope(self) -> None:
        values = [float(value) for value in range(1, 41)]
        lower, middle, upper = bollinger_bands(values, 20)
        self.assertEqual(len(lower), len(values))
        self.assertIsNotNone(middle[-1])
        assert lower[-1] is not None and middle[-1] is not None and upper[-1] is not None
        self.assertLess(lower[-1], middle[-1])
        self.assertLess(middle[-1], upper[-1])

    def test_average_true_range_positive(self) -> None:
        highs = [10.0 + i for i in range(20)]
        lows = [9.0 + i for i in range(20)]
        closes = [9.5 + i for i in range(20)]
        atr = average_true_range(highs, lows, closes, 14)
        last = atr[-1]
        self.assertIsNotNone(last)
        assert last is not None
        self.assertGreater(last, 0.0)

    def test_compute_indicator_snapshot_matches_lengths(self) -> None:
        closes = [100.0 + (i * 0.5) for i in range(60)]
        highs = [value + 1 for value in closes]
        lows = [value - 1 for value in closes]
        snapshot = compute_indicator_snapshot(closes, highs, lows)
        self.assertEqual(snapshot.close, closes[-1])
        self.assertIsNotNone(snapshot.rsi_14)
        self.assertIsNotNone(snapshot.macd)
        self.assertIsNotNone(snapshot.atr_14)
        self.assertIsNotNone(snapshot.bollinger_percent)


if __name__ == "__main__":
    unittest.main()
