from __future__ import annotations

import unittest

from trading_agent.autonomous.runner import (
    AutonomousConfig,
    split_symbols,
    _bucket_floor,
    _interval_to_seconds,
)
from datetime import UTC, datetime


class SplitSymbolsTests(unittest.TestCase):
    def test_basic_comma_list(self) -> None:
        symbols, intervals = split_symbols("BTCUSDT,ETHUSDT")
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(intervals, {})

    def test_per_symbol_interval(self) -> None:
        symbols, intervals = split_symbols("BTCUSDT@1m, ETHUSDT@5m,SOLUSDT")
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.assertEqual(intervals, {"BTCUSDT": "1m", "ETHUSDT": "5m"})

    def test_normalizes_case(self) -> None:
        symbols, intervals = split_symbols("btcusdt@15m")
        self.assertEqual(symbols, ["BTCUSDT"])
        self.assertEqual(intervals, {"BTCUSDT": "15m"})

    def test_iterable_input(self) -> None:
        symbols, intervals = split_symbols(["BTCUSDT@4h", "ETHUSDT"])
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(intervals, {"BTCUSDT": "4h"})


class IntervalHelpersTests(unittest.TestCase):
    def test_interval_to_seconds(self) -> None:
        self.assertEqual(_interval_to_seconds("30s"), 30)
        self.assertEqual(_interval_to_seconds("5m"), 300)
        self.assertEqual(_interval_to_seconds("1h"), 3600)
        self.assertEqual(_interval_to_seconds("1d"), 86_400)
        self.assertEqual(_interval_to_seconds("1w"), 604_800)

    def test_interval_to_seconds_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _interval_to_seconds("0m")
        with self.assertRaises(ValueError):
            _interval_to_seconds("xy")
        with self.assertRaises(ValueError):
            _interval_to_seconds("5z")

    def test_bucket_floor_aligns_to_interval(self) -> None:
        ts = datetime(2026, 5, 17, 14, 23, 45, tzinfo=UTC)
        self.assertEqual(
            _bucket_floor(ts, 60),
            datetime(2026, 5, 17, 14, 23, tzinfo=UTC),
        )
        self.assertEqual(
            _bucket_floor(ts, 300),
            datetime(2026, 5, 17, 14, 20, tzinfo=UTC),
        )
        self.assertEqual(
            _bucket_floor(ts, 3600),
            datetime(2026, 5, 17, 14, 0, tzinfo=UTC),
        )


class AutonomousConfigTests(unittest.TestCase):
    def test_interval_for_falls_back_to_default(self) -> None:
        config = AutonomousConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            symbol_intervals={"BTCUSDT": "1m"},
            candle_interval="1h",
        )
        self.assertEqual(config.interval_for("BTCUSDT"), "1m")
        self.assertEqual(config.interval_for("ETHUSDT"), "1h")
        self.assertEqual(config.interval_for("solusdt"), "1h")


if __name__ == "__main__":
    unittest.main()
