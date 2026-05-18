from __future__ import annotations

import unittest

from trading_agent.broker.exchange_simulator import (
    DEFAULT_FEE_RATE,
    ExchangeSimulator,
    SIDE_BUY,
    SIDE_SELL,
    parse_symbol,
)


class ExchangeSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logs: list[str] = []
        self.exchange = ExchangeSimulator(
            starting_cash=10_000.0,
            fee_rate=DEFAULT_FEE_RATE,
            quote="USDT",
            logger=self.logs.append,
        )

    def test_parse_symbol(self) -> None:
        self.assertEqual(parse_symbol("BTCUSDT"), ("BTC", "USDT"))
        self.assertEqual(parse_symbol("ethusdt"), ("ETH", "USDT"))
        with self.assertRaises(ValueError):
            parse_symbol("BTCUSD", quote="USDT")

    def test_initial_state(self) -> None:
        snapshot = self.exchange.snapshot()
        self.assertEqual(snapshot.cash, 10_000.0)
        self.assertEqual(snapshot.fills, 0)
        self.assertEqual(snapshot.holdings, {})

    def test_market_buy_charges_fee_and_updates_holding(self) -> None:
        self.exchange.on_tick("BTCUSDT", 50_000.0)
        fill = self.exchange.place_market("BTCUSDT", SIDE_BUY, quote_amount=1_000.0)
        self.assertEqual(fill.symbol, "BTCUSDT")
        self.assertEqual(fill.side, SIDE_BUY)
        self.assertGreater(fill.quantity, 0.0)
        self.assertGreater(fill.fee, 0.0)
        # cash should drop by gross + fee
        expected_cash = 10_000.0 - (fill.notional + fill.fee)
        self.assertAlmostEqual(self.exchange.cash, expected_cash, places=6)
        # 0.1% fee
        self.assertAlmostEqual(fill.fee / fill.notional, DEFAULT_FEE_RATE, places=8)
        holding = self.exchange.get_holding("BTC")
        self.assertAlmostEqual(holding.quantity, fill.quantity)
        self.assertAlmostEqual(holding.average_price, fill.price)

    def test_market_sell_returns_cash_minus_fee(self) -> None:
        self.exchange.on_tick("BTCUSDT", 50_000.0)
        self.exchange.place_market("BTCUSDT", SIDE_BUY, quote_amount=1_000.0)
        cash_after_buy = self.exchange.cash

        # Price moves up and we sell everything.
        self.exchange.on_tick("BTCUSDT", 55_000.0)
        holding = self.exchange.get_holding("BTC")
        sell_fill = self.exchange.place_market("BTCUSDT", SIDE_SELL, quantity=holding.quantity)

        self.assertEqual(sell_fill.side, SIDE_SELL)
        # The fee on the sell side comes out of the proceeds.
        expected_cash = cash_after_buy + sell_fill.notional - sell_fill.fee
        self.assertAlmostEqual(self.exchange.cash, expected_cash, places=6)
        self.assertAlmostEqual(self.exchange.get_holding("BTC").quantity, 0.0)

    def test_limit_buy_fills_when_price_drops_below_limit(self) -> None:
        self.exchange.on_tick("BTCUSDT", 50_000.0)
        self.exchange.place_limit(
            "BTCUSDT", SIDE_BUY, quantity=0.01, limit_price=48_000.0
        )
        # Price is still above the limit -> no fill
        fills = self.exchange.on_tick("BTCUSDT", 49_500.0)
        self.assertEqual(fills, [])
        # Price crosses below the limit -> filled at the limit price
        fills = self.exchange.on_tick("BTCUSDT", 47_500.0)
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].price, 48_000.0)
        self.assertEqual(self.exchange.open_orders(), [])

    def test_limit_sell_fills_when_price_rises_above_limit(self) -> None:
        self.exchange.on_tick("BTCUSDT", 50_000.0)
        self.exchange.place_market("BTCUSDT", SIDE_BUY, quote_amount=1_000.0)
        holding_qty = self.exchange.get_holding("BTC").quantity

        self.exchange.place_limit(
            "BTCUSDT",
            SIDE_SELL,
            quantity=holding_qty,
            limit_price=55_000.0,
        )
        self.assertEqual(self.exchange.on_tick("BTCUSDT", 54_000.0), [])
        fills = self.exchange.on_tick("BTCUSDT", 56_000.0)
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].price, 55_000.0)

    def test_buy_rejected_when_cash_insufficient(self) -> None:
        self.exchange.on_tick("BTCUSDT", 50_000.0)
        with self.assertRaises(ValueError):
            self.exchange.place_market("BTCUSDT", SIDE_BUY, quote_amount=20_000.0)

    def test_sell_rejected_when_no_holding(self) -> None:
        self.exchange.on_tick("BTCUSDT", 50_000.0)
        with self.assertRaises(ValueError):
            self.exchange.place_market("BTCUSDT", SIDE_SELL, quantity=0.001)

    def test_logger_records_fill_and_account_lines(self) -> None:
        self.exchange.on_tick("BTCUSDT", 50_000.0)
        self.exchange.place_market("BTCUSDT", SIDE_BUY, quote_amount=500.0)
        self.assertTrue(any("[FILL  ]" in line for line in self.logs))
        self.assertTrue(any("[ACCOUNT]" in line for line in self.logs))


if __name__ == "__main__":
    unittest.main()
