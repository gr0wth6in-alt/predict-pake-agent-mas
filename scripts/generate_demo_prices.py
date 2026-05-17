from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


def main() -> None:
    output_path = Path("examples/mixed_training_prices.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    price = 100.0
    timestamp = datetime(2026, 1, 1)

    for index in range(180):
        regime = (index // 30) % 3
        if regime == 0:
            drift = 0.006
        elif regime == 1:
            drift = -0.005
        else:
            drift = 0.0005

        cycle = math.sin(index / 4.0) * 0.006
        shock = ((index % 11) - 5) * 0.0008
        previous = price
        price = max(1.0, price * (1.0 + drift + cycle + shock))
        high = max(previous, price) * 1.004
        low = min(previous, price) * 0.996
        volume = 1000 + regime * 150 + (index % 12) * 18

        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": f"{previous:.4f}",
                "high": f"{high:.4f}",
                "low": f"{low:.4f}",
                "close": f"{price:.4f}",
                "volume": f"{volume:.2f}",
            }
        )
        timestamp += timedelta(hours=1)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)


if __name__ == "__main__":
    main()
