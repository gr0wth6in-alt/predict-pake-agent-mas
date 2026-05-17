from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    trading_mode: str = os.getenv("TRADING_MODE", "paper")
    symbol: str = os.getenv("SYMBOL", "BTCUSD")
    initial_cash: float = _env_float("INITIAL_CASH", 10_000.0)
    prediction_short_window: int = _env_int("PREDICTION_SHORT_WINDOW", 5)
    prediction_long_window: int = _env_int("PREDICTION_LONG_WINDOW", 20)
    prediction_threshold: float = _env_float("PREDICTION_THRESHOLD", 0.02)
    max_position_fraction: float = _env_float("MAX_POSITION_FRACTION", 0.25)
    max_order_notional: float = _env_float("MAX_ORDER_NOTIONAL", 2_500.0)
    stop_loss_pct: float = _env_float("STOP_LOSS_PCT", 0.03)
    take_profit_pct: float = _env_float("TAKE_PROFIT_PCT", 0.06)
    allow_short: bool = _env_bool("ALLOW_SHORT", False)


def load_settings() -> Settings:
    return Settings()
