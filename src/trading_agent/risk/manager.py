from __future__ import annotations

from dataclasses import dataclass

from trading_agent.features.indicators import clamp
from trading_agent.models import Order, Position, Side, Signal


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    order: Order | None
    reason: str


class RiskManager:
    def __init__(
        self,
        max_position_fraction: float = 0.25,
        max_order_notional: float = 2_500.0,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        allow_short: bool = False,
    ):
        if not 0 < max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be between 0 and 1")
        if max_order_notional <= 0:
            raise ValueError("max_order_notional must be positive")
        if stop_loss_pct <= 0 or take_profit_pct <= 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be positive")

        self.max_position_fraction = max_position_fraction
        self.max_order_notional = max_order_notional
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.allow_short = allow_short

    def evaluate(
        self,
        signal: Signal,
        cash: float,
        price: float,
        position: Position | None = None,
    ) -> RiskDecision:
        if price <= 0:
            return RiskDecision(False, None, "invalid price")

        current_position = position or Position(symbol=signal.symbol)

        exit_decision = self._maybe_exit_for_protection(signal, price, current_position)
        if exit_decision is not None:
            return exit_decision

        if signal.side == Side.HOLD:
            return RiskDecision(False, None, "hold signal")

        if signal.side == Side.SELL and current_position.quantity <= 0 and not self.allow_short:
            return RiskDecision(False, None, "short selling disabled")

        strength = clamp(signal.strength, 0.0, 1.0)
        budget = min(cash * self.max_position_fraction * strength, self.max_order_notional)

        if signal.side == Side.SELL and current_position.quantity > 0:
            quantity = min(current_position.quantity, self.max_order_notional / price)
        else:
            quantity = budget / price

        if quantity <= 0:
            return RiskDecision(False, None, "calculated quantity was zero")

        order = Order(
            symbol=signal.symbol,
            side=signal.side,
            quantity=quantity,
            limit_price=price,
            reason=signal.reason,
        )
        return RiskDecision(True, order, "approved")

    def _maybe_exit_for_protection(
        self,
        signal: Signal,
        price: float,
        position: Position,
    ) -> RiskDecision | None:
        if position.quantity <= 0 or position.average_price <= 0:
            return None

        stop_price = position.average_price * (1 - self.stop_loss_pct)
        take_profit_price = position.average_price * (1 + self.take_profit_pct)
        should_exit = price <= stop_price or price >= take_profit_price

        if not should_exit:
            return None

        reason = "stop loss" if price <= stop_price else "take profit"
        quantity = min(position.quantity, self.max_order_notional / price)
        order = Order(
            symbol=signal.symbol,
            side=Side.SELL,
            quantity=quantity,
            limit_price=price,
            reason=reason,
        )
        return RiskDecision(True, order, reason)
