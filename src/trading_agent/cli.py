from __future__ import annotations

import argparse
from pathlib import Path

from trading_agent.agent import TradingAgent
from trading_agent.backtest.engine import BacktestEngine
from trading_agent.broker.paper import PaperBroker
from trading_agent.config import load_settings
from trading_agent.data.csv_feed import load_candles
from trading_agent.prediction.baseline import MovingAverageMomentumPredictor
from trading_agent.risk.manager import RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy


def build_components() -> tuple[MovingAverageMomentumPredictor, ThresholdStrategy, RiskManager]:
    settings = load_settings()
    predictor = MovingAverageMomentumPredictor(
        short_window=settings.prediction_short_window,
        long_window=settings.prediction_long_window,
    )
    strategy = ThresholdStrategy(threshold=settings.prediction_threshold)
    risk_manager = RiskManager(
        max_position_fraction=settings.max_position_fraction,
        max_order_notional=settings.max_order_notional,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        allow_short=settings.allow_short,
    )
    return predictor, strategy, risk_manager


def run_backtest(args: argparse.Namespace) -> None:
    settings = load_settings()
    candles = load_candles(Path(args.csv), args.symbol)
    predictor, strategy, risk_manager = build_components()
    engine = BacktestEngine(
        predictor=predictor,
        strategy=strategy,
        risk_manager=risk_manager,
        starting_cash=args.cash if args.cash is not None else settings.initial_cash,
    )
    result = engine.run(candles)
    print(f"starting_cash={result.starting_cash:.2f}")
    print(f"ending_equity={result.ending_equity:.2f}")
    print(f"total_return_pct={result.total_return_pct:.2f}")
    print(f"fills={len(result.fills)}")


def run_paper_once(args: argparse.Namespace) -> None:
    settings = load_settings()
    candles = load_candles(Path(args.csv), args.symbol)
    predictor, strategy, risk_manager = build_components()
    broker = PaperBroker(cash=args.cash if args.cash is not None else settings.initial_cash)
    agent = TradingAgent(
        predictor=predictor,
        strategy=strategy,
        risk_manager=risk_manager,
        broker=broker,
    )
    decision = agent.run_once(candles)
    print(f"signal={decision.signal.side.value}")
    print(f"strength={decision.signal.strength:.4f}")
    print(f"risk={decision.risk_decision.reason}")
    if decision.fill is None:
        print("fill=none")
    else:
        print(
            "fill="
            f"{decision.fill.side.value} "
            f"{decision.fill.quantity:.8f} "
            f"{decision.fill.symbol} "
            f"@ {decision.fill.price:.2f}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous trading agent starter CLI")
    subparsers = parser.add_subparsers(required=True)

    backtest = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    backtest.add_argument("--symbol", default=load_settings().symbol)
    backtest.add_argument("--cash", type=float, default=None)
    backtest.set_defaults(func=run_backtest)

    paper_once = subparsers.add_parser("paper-once", help="Run one paper trading decision")
    paper_once.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    paper_once.add_argument("--symbol", default=load_settings().symbol)
    paper_once.add_argument("--cash", type=float, default=None)
    paper_once.set_defaults(func=run_paper_once)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
