from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from trading_agent.models import Candle


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_candles(path: str | Path, symbol: str) -> list[Candle]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or missing a header row")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing columns: {missing_cols}")

        candles = [
            Candle(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in reader
        ]

    if not candles:
        raise ValueError(f"{csv_path} did not contain any candles")
    return candles
