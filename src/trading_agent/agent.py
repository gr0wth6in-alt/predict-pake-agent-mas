from __future__ import annotations

from dataclasses import dataclass

from trading_agent.broker.paper import Fill, PaperBroker
from trading_agent.models import Candle, Signal
from trading_agent.prediction.protocols import Predictor
from trading_agent.risk.manager import RiskDecision, RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy


@dataclass(frozen=True)
class AgentDecision:
    signal: Signal
    risk_decision: RiskDecision
    fill: Fill | None


class TradingAgent:
    def __init__(
        self,
        predictor: Predictor,
        strategy: ThresholdStrategy,
        risk_manager: RiskManager,
        broker: PaperBroker,
    ):
        self.predictor = predictor
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.broker = broker

    def run_once(self, candles: list[Candle]) -> AgentDecision:
        if len(candles) < self.predictor.min_history:
            raise ValueError("not enough candles for a prediction")

        latest = candles[-1]
        prediction = self.predictor.predict(candles)
        signal = self.strategy.generate_signal(prediction)
        risk_decision = self.risk_manager.evaluate(
            signal=signal,
            cash=self.broker.cash,
            price=latest.close,
            position=self.broker.get_position(latest.symbol),
        )
        fill = None
        if risk_decision.approved and risk_decision.order is not None:
            fill = self.broker.execute(risk_decision.order)

        return AgentDecision(signal=signal, risk_decision=risk_decision, fill=fill)
