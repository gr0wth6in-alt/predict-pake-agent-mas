from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from trading_agent.backtest.engine import BacktestEngine
from trading_agent.models import Candle
from trading_agent.prediction.baseline import MovingAverageMomentumPredictor
from trading_agent.risk.manager import RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy


def _candles(count: int) -> list[Candle]:
    start = datetime(2026, 1, 1)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            symbol="BTCUSD",
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100 + index,
            volume=1000,
        )
        for index in range(count)
    ]


class BacktestSmokeTests(unittest.TestCase):
    def test_backtest_runs(self) -> None:
        engine = BacktestEngine(
            predictor=MovingAverageMomentumPredictor(short_window=3, long_window=5),
            strategy=ThresholdStrategy(threshold=0.01),
            risk_manager=RiskManager(max_order_notional=1000),
            starting_cash=10_000,
        )

        result = engine.run(_candles(12))

        self.assertGreater(len(result.equity_curve), 0)
        self.assertGreater(result.ending_equity, 0)


if __name__ == "__main__":
    unittest.main()
