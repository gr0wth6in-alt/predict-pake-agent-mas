from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from trading_agent.api import app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_uses_default_demo_data(self) -> None:
        response = self.client.post("/predict", json={"symbol": "BTCUSD"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["symbol"], "BTCUSD")
        self.assertIn("confidence", payload)

    def test_backtest_uses_default_demo_data(self) -> None:
        response = self.client.post("/backtest", json={"symbol": "BTCUSD"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(payload["ending_equity"], 0)
        self.assertGreater(payload["equity_points"], 0)


if __name__ == "__main__":
    unittest.main()
