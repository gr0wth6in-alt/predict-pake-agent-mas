from __future__ import annotations

import unittest

from trading_agent.features.indicators import percentage_change, simple_moving_average


class IndicatorTests(unittest.TestCase):
    def test_simple_moving_average(self) -> None:
        self.assertEqual(
            simple_moving_average([1, 2, 3, 4], 2),
            [None, 1.5, 2.5, 3.5],
        )

    def test_percentage_change(self) -> None:
        self.assertEqual(percentage_change(0, 10), 0.0)
        self.assertAlmostEqual(percentage_change(100, 110), 0.10)


if __name__ == "__main__":
    unittest.main()
