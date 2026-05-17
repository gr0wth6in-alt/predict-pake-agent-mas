from __future__ import annotations

import unittest

from trading_agent.models import Position, Side, Signal
from trading_agent.risk.manager import RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_blocks_short_when_disabled(self) -> None:
        manager = RiskManager(allow_short=False)
        signal = Signal(symbol="BTCUSD", side=Side.SELL, strength=1.0, reason="test")
        decision = manager.evaluate(signal, cash=10_000, price=100, position=Position("BTCUSD"))

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "short selling disabled")

    def test_caps_buy_order_notional(self) -> None:
        manager = RiskManager(max_position_fraction=1.0, max_order_notional=500)
        signal = Signal(symbol="BTCUSD", side=Side.BUY, strength=1.0, reason="test")
        decision = manager.evaluate(signal, cash=10_000, price=100, position=Position("BTCUSD"))

        self.assertTrue(decision.approved)
        self.assertIsNotNone(decision.order)
        assert decision.order is not None
        self.assertAlmostEqual(decision.order.notional, 500)

    def test_protective_exit_can_override_hold(self) -> None:
        manager = RiskManager(stop_loss_pct=0.03)
        signal = Signal(symbol="BTCUSD", side=Side.HOLD, strength=0.0, reason="test")
        position = Position(symbol="BTCUSD", quantity=2.0, average_price=100.0)
        decision = manager.evaluate(signal, cash=10_000, price=95.0, position=position)

        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "stop loss")
        self.assertIsNotNone(decision.order)
        assert decision.order is not None
        self.assertEqual(decision.order.side, Side.SELL)


if __name__ == "__main__":
    unittest.main()
