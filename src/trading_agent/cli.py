from __future__ import annotations

import argparse
from pathlib import Path

from trading_agent.agent import TradingAgent
from trading_agent.backtest.engine import BacktestEngine
from trading_agent.broker.paper import PaperBroker
from trading_agent.config import load_settings
from trading_agent.data.binance_feed import DEFAULT_INTERVAL, DEFAULT_LIMIT, limit_for_days, load_binance_klines
from trading_agent.data.coingecko_feed import DEFAULT_DAYS, DEFAULT_VS_CURRENCY, load_coingecko_ohlc
from trading_agent.data.csv_feed import load_candles
from trading_agent.prediction.baseline import MovingAverageMomentumPredictor
from trading_agent.prediction.ml import TrainedModelPredictor
from trading_agent.prediction.protocols import Predictor
from trading_agent.risk.manager import RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy
from trading_agent.training.dataset import TrainingConfig
from trading_agent.training.trainer import train_and_save


def build_components(args: argparse.Namespace | None = None) -> tuple[Predictor, ThresholdStrategy, RiskManager]:
    settings = load_settings()
    model_path = getattr(args, "model_path", None) if args is not None else None
    if model_path is None:
        predictor: Predictor = MovingAverageMomentumPredictor(
            short_window=settings.prediction_short_window,
            long_window=settings.prediction_long_window,
        )
    else:
        predictor = TrainedModelPredictor.load(model_path)

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
    predictor, strategy, risk_manager = build_components(args)
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
    predictor, strategy, risk_manager = build_components(args)
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


def run_train(args: argparse.Namespace) -> None:
    try:
        if args.data_source == "binance":
            candles = load_binance_klines(
                symbol=args.symbol,
                interval=args.interval,
                limit=limit_for_days(args.days, args.interval) if args.days else args.limit,
            )
        elif args.data_source == "coingecko":
            candles = load_coingecko_ohlc(
                symbol=args.symbol,
                coin_id=args.coin_id,
                vs_currency=args.vs_currency,
                days=args.days,
            )
        else:
            if args.csv is None:
                raise SystemExit("--csv is required when --data-source csv")
            candles = load_candles(Path(args.csv), args.symbol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    config = TrainingConfig(
        lookback=args.lookback,
        horizon=args.horizon,
        label_threshold=args.label_threshold,
    )
    result = train_and_save(
        candles,
        symbol=args.symbol,
        config=config,
        output_path=args.output,
        train_fraction=args.train_fraction,
    )

    print(f"model_path={Path(args.output)}")
    print(f"samples={result.sample_count}")
    print(f"train_samples={result.train_count}")
    print(f"test_samples={result.test_count}")
    print(f"labels={result.label_distribution}")
    print(f"train_accuracy={result.train_metrics.accuracy:.4f}")
    print(f"test_accuracy={result.test_metrics.accuracy:.4f}")
    for warning in result.warnings:
        print(f"warning={warning}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous trading agent starter CLI")
    subparsers = parser.add_subparsers(required=True)

    train = subparsers.add_parser("train", help="Train a supervised prediction model")
    train.add_argument("--data-source", choices=["csv", "coingecko", "binance"], default="csv")
    train.add_argument("--csv", default=None, help="Path to OHLCV CSV")
    train.add_argument("--symbol", default=load_settings().symbol)
    train.add_argument("--coin-id", default=None, help="CoinGecko coin id, for example bitcoin")
    train.add_argument("--vs-currency", default=DEFAULT_VS_CURRENCY)
    train.add_argument("--days", type=int, default=DEFAULT_DAYS)
    train.add_argument("--interval", default=DEFAULT_INTERVAL)
    train.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    train.add_argument("--output", default="models/latest_model.json")
    train.add_argument("--lookback", type=int, default=10)
    train.add_argument("--horizon", type=int, default=3)
    train.add_argument("--label-threshold", type=float, default=0.01)
    train.add_argument("--train-fraction", type=float, default=0.8)
    train.set_defaults(func=run_train)

    backtest = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    backtest.add_argument("--symbol", default=load_settings().symbol)
    backtest.add_argument("--cash", type=float, default=None)
    backtest.add_argument("--model-path", default=None, help="Use a trained JSON model")
    backtest.set_defaults(func=run_backtest)

    paper_once = subparsers.add_parser("paper-once", help="Run one paper trading decision")
    paper_once.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    paper_once.add_argument("--symbol", default=load_settings().symbol)
    paper_once.add_argument("--cash", type=float, default=None)
    paper_once.add_argument("--model-path", default=None, help="Use a trained JSON model")
    paper_once.set_defaults(func=run_paper_once)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
