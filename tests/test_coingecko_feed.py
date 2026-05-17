from __future__ import annotations

import unittest

from trading_agent.data.coingecko_feed import _parse_ohlc_row, coin_id_for_symbol


class CoinGeckoFeedTests(unittest.TestCase):
    def test_coin_id_for_symbol(self) -> None:
        self.assertEqual(coin_id_for_symbol("BTCUSD"), "bitcoin")
        self.assertEqual(coin_id_for_symbol("ethusdt"), "ethereum")
        self.assertEqual(coin_id_for_symbol("custom-coin"), "custom-coin")

    def test_parse_ohlc_row(self) -> None:
        candle = _parse_ohlc_row([1_704_067_200_000, 100, 110, 90, 105], "BTCUSD")

        self.assertEqual(candle.symbol, "BTCUSD")
        self.assertEqual(candle.open, 100.0)
        self.assertEqual(candle.high, 110.0)
        self.assertEqual(candle.low, 90.0)
        self.assertEqual(candle.close, 105.0)
        self.assertEqual(candle.volume, 0.0)


if __name__ == "__main__":
    unittest.main()
