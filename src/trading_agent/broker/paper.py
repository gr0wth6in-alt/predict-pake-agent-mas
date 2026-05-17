from __future__ import annotations

from dataclasses import dataclass, field

from trading_agent.models import Order, Position, Side


@dataclass
class Fill:
    symbol: str
    side: Side
    quantity: float
    price: float
    notional: float
    reason: str


@dataclass
class PaperBroker:
    cash: float = 10_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol=symbol))

    def execute(self, order: Order) -> Fill:
        if order.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if order.limit_price <= 0:
            raise ValueError("order price must be positive")

        position = self.get_position(order.symbol)
        notional = order.notional

        if order.side == Side.BUY:
            if notional > self.cash:
                raise ValueError("insufficient paper cash")
            total_cost = position.average_price * position.quantity + notional
            new_quantity = position.quantity + order.quantity
            position.quantity = new_quantity
            position.average_price = total_cost / new_quantity
            self.cash -= notional
        elif order.side == Side.SELL:
            if order.quantity > position.quantity:
                raise ValueError("cannot sell more than current paper position")
            position.quantity -= order.quantity
            self.cash += notional
            if position.quantity == 0:
                position.average_price = 0.0
        else:
            raise ValueError("hold orders cannot be executed")

        self.positions[order.symbol] = position
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.limit_price,
            notional=notional,
            reason=order.reason,
        )
        self.fills.append(fill)
        return fill

    def equity(self, last_prices: dict[str, float]) -> float:
        position_value = 0.0
        for symbol, position in self.positions.items():
            position_value += position.quantity * last_prices.get(symbol, position.average_price)
        return self.cash + position_value
