from __future__ import annotations

import unittest

from trading_agent.data.binance_feed import _parse_kline_row, limit_for_days


class BinanceFeedTests(unittest.TestCase):
    def test_parse_kline_row(self) -> None:
        candle = _parse_kline_row(
            [
                1_704_067_200_000,
                "100.0",
                "110.0",
                "90.0",
                "105.0",
                "123.45",
                1_704_070_799_999,
            ],
            "BTCUSDT",
        )

        self.assertEqual(candle.symbol, "BTCUSDT")
        self.assertEqual(candle.open, 100.0)
        self.assertEqual(candle.high, 110.0)
        self.assertEqual(candle.low, 90.0)
        self.assertEqual(candle.close, 105.0)
        self.assertEqual(candle.volume, 123.45)

    def test_limit_for_days(self) -> None:
        self.assertEqual(limit_for_days(1, "1h"), 24)
        self.assertEqual(limit_for_days(2, "4h"), 12)
        self.assertEqual(limit_for_days(100, "1h"), 1000)


if __name__ == "__main__":
    unittest.main()
