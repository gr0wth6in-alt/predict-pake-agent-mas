from __future__ import annotations

import unittest

from trading_agent.wsgi_app import app


class WsgiAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    def test_predict(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "symbol": "BTCUSD",
                "data_source": "csv",
                "csv_path": "examples/mixed_training_prices.csv",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["symbol"], "BTCUSD")
        self.assertIn("confidence", response.json)


if __name__ == "__main__":
    unittest.main()
