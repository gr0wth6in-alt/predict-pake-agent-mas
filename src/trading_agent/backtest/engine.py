from __future__ import annotations

from dataclasses import dataclass

from trading_agent.broker.paper import Fill, PaperBroker
from trading_agent.models import Candle
from trading_agent.prediction.baseline import MovingAverageMomentumPredictor
from trading_agent.risk.manager import RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    equity: float
    close: float


@dataclass(frozen=True)
class BacktestResult:
    starting_cash: float
    ending_equity: float
    total_return_pct: float
    fills: list[Fill]
    equity_curve: list[EquityPoint]


class BacktestEngine:
    def __init__(
        self,
        predictor: MovingAverageMomentumPredictor,
        strategy: ThresholdStrategy,
        risk_manager: RiskManager,
        starting_cash: float = 10_000.0,
    ):
        self.predictor = predictor
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.starting_cash = starting_cash

    def run(self, candles: list[Candle]) -> BacktestResult:
        if len(candles) < self.predictor.min_history:
            raise ValueError("not enough candles to run backtest")

        broker = PaperBroker(cash=self.starting_cash)
        equity_curve: list[EquityPoint] = []

        for index in range(self.predictor.min_history, len(candles)):
            history = candles[: index + 1]
            latest = history[-1]
            prediction = self.predictor.predict(history)
            signal = self.strategy.generate_signal(prediction)
            position = broker.get_position(latest.symbol)
            decision = self.risk_manager.evaluate(
                signal=signal,
                cash=broker.cash,
                price=latest.close,
                position=position,
            )
            if decision.approved and decision.order is not None:
                broker.execute(decision.order)

            equity_curve.append(
                EquityPoint(
                    timestamp=latest.timestamp.isoformat(),
                    equity=broker.equity({latest.symbol: latest.close}),
                    close=latest.close,
                )
            )

        ending_equity = equity_curve[-1].equity
        total_return_pct = (ending_equity - self.starting_cash) / self.starting_cash * 100
        return BacktestResult(
            starting_cash=self.starting_cash,
            ending_equity=ending_equity,
            total_return_pct=total_return_pct,
            fills=list(broker.fills),
            equity_curve=equity_curve,
        )
