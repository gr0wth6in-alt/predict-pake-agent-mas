from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Prediction:
    symbol: str
    direction_score: float
    confidence: float
    horizon_candles: int
    rationale: str


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Side
    strength: float
    reason: str


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    limit_price: float
    reason: str

    @property
    def notional(self) -> float:
        return self.quantity * self.limit_price


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.average_price
