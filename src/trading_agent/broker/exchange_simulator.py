"""Paper exchange simulator with a matching engine.

This module simulates a fake crypto exchange. It is intentionally written in a very
straightforward style so beginners can follow what happens on every fill:

* the simulator holds a USD-like cash balance plus one balance entry per coin
  (for example "BTC", "ETH"). All trading pairs are quoted in USDT and we treat
  USDT == USD;
* :meth:`place_market` fills instantly at the latest known price;
* :meth:`place_limit` adds the order to a resting list. The matching engine will
  fill it later, when ``on_tick`` is called with a price that crosses the limit;
* every fill pays a 0.1% fee by default and prints a small log line plus the
  current account snapshot;
* state is protected by a single :class:`threading.RLock` so the streamer thread
  and the decision thread can safely call into the simulator at the same time.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Iterable


DEFAULT_FEE_RATE = 0.001  # 0.1%
DEFAULT_QUOTE = "USDT"
SIDE_BUY = "buy"
SIDE_SELL = "sell"
KIND_MARKET = "market"
KIND_LIMIT = "limit"


@dataclass
class Holding:
    """Quantity of a single base asset (for example BTC) plus its average buy price."""

    quantity: float = 0.0
    average_price: float = 0.0


@dataclass
class SimFill:
    """Result of an executed order. ``notional`` is the gross quote value (qty * price)."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    notional: float
    timestamp: datetime
    kind: str
    reason: str


@dataclass
class RestingOrder:
    """A limit order that is waiting to be filled by the matching engine."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    limit_price: float
    placed_at: datetime
    reason: str


@dataclass(frozen=True)
class AccountSnapshot:
    """Plain-data view of the account state. Easy to print or send through an API."""

    cash: float
    holdings: dict[str, Holding]
    open_orders: int
    total_fees_paid: float
    fills: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cash": self.cash,
            "holdings": {
                base: {"quantity": h.quantity, "average_price": h.average_price}
                for base, h in self.holdings.items()
                if h.quantity > 0
            },
            "open_orders": self.open_orders,
            "total_fees_paid": self.total_fees_paid,
            "fills": self.fills,
        }


def parse_symbol(symbol: str, quote: str = DEFAULT_QUOTE) -> tuple[str, str]:
    """Return (base, quote). Example: ``parse_symbol("BTCUSDT")`` -> ``("BTC", "USDT")``."""

    upper = symbol.strip().upper()
    if not upper.endswith(quote):
        raise ValueError(f"symbol {symbol!r} does not use {quote} as the quote currency")
    base = upper[: -len(quote)]
    if not base:
        raise ValueError(f"symbol {symbol!r} has an empty base asset")
    return base, quote


class ExchangeSimulator:
    """A small paper exchange. One instance simulates one trading account."""

    def __init__(
        self,
        *,
        starting_cash: float = 10_000.0,
        fee_rate: float = DEFAULT_FEE_RATE,
        quote: str = DEFAULT_QUOTE,
        logger: Callable[[str], None] | None = print,
    ):
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if not 0 <= fee_rate < 0.5:
            raise ValueError("fee_rate must be between 0 and 0.5")

        self._lock = threading.RLock()
        self._cash = starting_cash
        self._fee_rate = fee_rate
        self._quote = quote
        self._holdings: dict[str, Holding] = {}
        self._last_price: dict[str, float] = {}
        self._resting: list[RestingOrder] = []
        self._fills: list[SimFill] = []
        self._total_fees = 0.0
        self._order_id_counter = itertools.count(start=1)
        self._log = logger or (lambda _msg: None)

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        with self._lock:
            return self._cash

    @property
    def fee_rate(self) -> float:
        return self._fee_rate

    def get_holding(self, base: str) -> Holding:
        with self._lock:
            holding = self._holdings.get(base.upper())
            if holding is None:
                return Holding()
            return Holding(quantity=holding.quantity, average_price=holding.average_price)

    def last_price(self, symbol: str) -> float | None:
        with self._lock:
            return self._last_price.get(symbol.strip().upper())

    def open_orders(self) -> list[RestingOrder]:
        with self._lock:
            return list(self._resting)

    def fills(self) -> list[SimFill]:
        with self._lock:
            return list(self._fills)

    def equity(self) -> float:
        """Cash + market value of holdings using the latest prices we have seen."""

        with self._lock:
            value = self._cash
            for base, holding in self._holdings.items():
                price = self._price_for_base(base)
                if price is None:
                    price = holding.average_price
                value += holding.quantity * price
            return value

    def snapshot(self) -> AccountSnapshot:
        with self._lock:
            return AccountSnapshot(
                cash=self._cash,
                holdings={base: Holding(h.quantity, h.average_price) for base, h in self._holdings.items()},
                open_orders=len(self._resting),
                total_fees_paid=self._total_fees,
                fills=len(self._fills),
            )

    # ------------------------------------------------------------------
    # Order entry
    # ------------------------------------------------------------------

    def place_market(
        self,
        symbol: str,
        side: str,
        *,
        quantity: float | None = None,
        quote_amount: float | None = None,
        reason: str = "",
    ) -> SimFill:
        """Fill immediately at the latest known price.

        Provide either ``quantity`` (in base units) or ``quote_amount`` (in USD-like quote
        units). One of the two is required.
        """

        side = _normalize_side(side)
        with self._lock:
            price = self._require_price(symbol)
            qty = self._resolve_quantity(symbol, quantity, quote_amount, price)
            order_id = self._next_order_id(symbol, KIND_MARKET)
            fill = self._fill(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                kind=KIND_MARKET,
                reason=reason or "market order",
            )
            return fill

    def place_limit(
        self,
        symbol: str,
        side: str,
        *,
        quantity: float,
        limit_price: float,
        reason: str = "",
    ) -> RestingOrder:
        """Add a limit order to the resting list. Filled later by ``on_tick``."""

        side = _normalize_side(side)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if limit_price <= 0:
            raise ValueError("limit_price must be positive")

        with self._lock:
            base, _ = parse_symbol(symbol, self._quote)
            if side == SIDE_BUY:
                required_cash = quantity * limit_price * (1.0 + self._fee_rate)
                if required_cash > self._cash:
                    raise ValueError("insufficient cash to reserve this limit BUY")
            else:
                holding = self._holdings.get(base, Holding())
                if quantity > holding.quantity:
                    raise ValueError("insufficient holdings to reserve this limit SELL")

            order_id = self._next_order_id(symbol, KIND_LIMIT)
            order = RestingOrder(
                order_id=order_id,
                symbol=symbol.strip().upper(),
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                placed_at=_now(),
                reason=reason or "limit order",
            )
            self._resting.append(order)
            self._log(
                f"[ORDER ] {order.symbol} {order.side.upper()} LIMIT "
                f"qty={order.quantity:.8f} @ {order.limit_price:.4f} (id={order.order_id})"
            )
            return order

    def cancel(self, order_id: str) -> bool:
        with self._lock:
            for index, order in enumerate(self._resting):
                if order.order_id == order_id:
                    self._resting.pop(index)
                    self._log(f"[CANCEL] {order.symbol} order_id={order_id}")
                    return True
            return False

    # ------------------------------------------------------------------
    # Matching engine
    # ------------------------------------------------------------------

    def on_tick(self, symbol: str, price: float) -> list[SimFill]:
        """Update the latest price for ``symbol`` and try to fill any resting orders.

        Returns the list of fills produced by this tick (often empty).
        """

        if price <= 0:
            return []

        normalized = symbol.strip().upper()
        with self._lock:
            self._last_price[normalized] = price

            still_resting: list[RestingOrder] = []
            new_fills: list[SimFill] = []
            for order in self._resting:
                if order.symbol != normalized:
                    still_resting.append(order)
                    continue
                if not _crosses(order, price):
                    still_resting.append(order)
                    continue
                # Fill at the better of (limit price, current price). Beginner-friendly
                # rule: a buy fills at the limit price if the market drops below it,
                # and a sell fills at the limit price if the market rises above it.
                fill_price = order.limit_price
                try:
                    fill = self._fill(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        kind=KIND_LIMIT,
                        reason=order.reason,
                    )
                except ValueError as exc:
                    self._log(f"[REJECT] {order.symbol} limit fill skipped: {exc}")
                    still_resting.append(order)
                    continue
                new_fills.append(fill)

            self._resting = still_resting
            return new_fills

    def update_prices(self, prices: Iterable[tuple[str, float]]) -> list[SimFill]:
        """Apply many ticks at once. Returns the union of all fills produced."""

        all_fills: list[SimFill] = []
        for symbol, price in prices:
            all_fills.extend(self.on_tick(symbol, price))
        return all_fills

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill(
        self,
        *,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        kind: str,
        reason: str,
    ) -> SimFill:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if price <= 0:
            raise ValueError("price must be positive")

        base, _ = parse_symbol(symbol, self._quote)
        gross = quantity * price
        fee = gross * self._fee_rate
        holding = self._holdings.setdefault(base, Holding())

        if side == SIDE_BUY:
            total_cost = gross + fee
            if total_cost > self._cash + 1e-9:
                raise ValueError(
                    f"insufficient cash: need {total_cost:.4f}, have {self._cash:.4f}"
                )
            new_qty = holding.quantity + quantity
            new_cost_basis = holding.quantity * holding.average_price + gross
            holding.quantity = new_qty
            holding.average_price = new_cost_basis / new_qty if new_qty > 0 else 0.0
            self._cash -= total_cost
        else:
            if quantity > holding.quantity + 1e-9:
                raise ValueError(
                    f"insufficient holdings: need {quantity:.8f} {base}, have {holding.quantity:.8f}"
                )
            holding.quantity -= quantity
            if holding.quantity <= 1e-12:
                holding.quantity = 0.0
                holding.average_price = 0.0
            self._cash += gross - fee

        self._total_fees += fee
        fill = SimFill(
            order_id=order_id,
            symbol=symbol.strip().upper(),
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            notional=gross,
            timestamp=_now(),
            kind=kind,
            reason=reason,
        )
        self._fills.append(fill)
        self._log_fill(fill)
        return fill

    def _resolve_quantity(
        self,
        symbol: str,
        quantity: float | None,
        quote_amount: float | None,
        price: float,
    ) -> float:
        if quantity is not None and quantity > 0:
            return quantity
        if quote_amount is not None and quote_amount > 0:
            # Reserve a small slice for the fee so a BUY does not fail by a few cents.
            net = quote_amount / (1.0 + self._fee_rate)
            return net / price
        raise ValueError("provide either quantity or quote_amount > 0")

    def _require_price(self, symbol: str) -> float:
        normalized = symbol.strip().upper()
        price = self._last_price.get(normalized)
        if price is None or price <= 0:
            raise ValueError(
                f"no last price for {normalized}; feed at least one tick before placing market orders"
            )
        return price

    def _price_for_base(self, base: str) -> float | None:
        symbol = f"{base}{self._quote}"
        return self._last_price.get(symbol)

    def _next_order_id(self, symbol: str, kind: str) -> str:
        return f"{symbol.upper()}-{kind[0].upper()}-{next(self._order_id_counter):06d}"

    def _log_fill(self, fill: SimFill) -> None:
        snapshot = self.snapshot()
        holdings_view = ", ".join(
            f"{base}={h.quantity:.6f}@{h.average_price:.4f}"
            for base, h in snapshot.holdings.items()
            if h.quantity > 0
        ) or "no positions"
        self._log(
            f"[FILL  ] {fill.timestamp.isoformat(timespec='seconds')} "
            f"{fill.symbol} {fill.side.upper()} {fill.kind.upper()} "
            f"qty={fill.quantity:.8f} @ {fill.price:.4f} "
            f"notional={fill.notional:.2f} fee={fill.fee:.4f} "
            f"reason={fill.reason or '-'}"
        )
        self._log(
            f"[ACCOUNT] cash={snapshot.cash:.2f} {self._quote} | {holdings_view} "
            f"| equity={self.equity():.2f} | total_fees={snapshot.total_fees_paid:.4f}"
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _crosses(order: RestingOrder, price: float) -> bool:
    """Return True when the market price has reached this limit order."""

    if order.side == SIDE_BUY:
        return price <= order.limit_price
    return price >= order.limit_price


def _normalize_side(side: str) -> str:
    normalized = side.strip().lower()
    if normalized not in {SIDE_BUY, SIDE_SELL}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return normalized


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "AccountSnapshot",
    "DEFAULT_FEE_RATE",
    "ExchangeSimulator",
    "Holding",
    "RestingOrder",
    "SIDE_BUY",
    "SIDE_SELL",
    "SimFill",
    "parse_symbol",
]
