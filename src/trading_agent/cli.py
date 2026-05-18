from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trading_agent.agent import TradingAgent
from trading_agent.autonomous.runner import (
    AutonomousConfig,
    run_until_interrupt,
    split_symbols,
)
from trading_agent.backtest.engine import BacktestEngine
from trading_agent.broker.exchange_simulator import DEFAULT_FEE_RATE
from trading_agent.broker.paper import PaperBroker
from trading_agent.config import load_settings
from trading_agent.data.binance_feed import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    limit_for_days,
    load_binance_klines,
)
from trading_agent.data.coingecko_feed import DEFAULT_DAYS, DEFAULT_VS_CURRENCY, load_coingecko_ohlc
from trading_agent.data.csv_feed import load_candles
from trading_agent.data.live_feed import fetch_live_market
from trading_agent.prediction.factory import (
    PREDICTOR_AUTO,
    PREDICTOR_NAMES,
    build_predictor,
)
from trading_agent.prediction.protocols import Predictor
from trading_agent.risk.manager import RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy
from trading_agent.training.auto import (
    AutoTrainConfig,
    auto_train_loop,
    auto_train_once,
)
from trading_agent.training.dataset import TrainingConfig
from trading_agent.training.trainer import train_and_save


def build_components(args: argparse.Namespace | None = None) -> tuple[Predictor, ThresholdStrategy, RiskManager]:
    settings = load_settings()
    predictor_name = getattr(args, "predictor", PREDICTOR_AUTO) if args is not None else PREDICTOR_AUTO
    model_path = getattr(args, "model_path", None) if args is not None else None
    symbol = getattr(args, "symbol", None) if args is not None else None
    predictor = build_predictor(
        predictor_name,
        model_path=model_path,
        symbol=symbol,
        settings=settings,
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
    candles = _load_candles_for_runtime(args)
    predictor, strategy, risk_manager = build_components(args)
    engine = BacktestEngine(
        predictor=predictor,
        strategy=strategy,
        risk_manager=risk_manager,
        starting_cash=args.cash if args.cash is not None else settings.initial_cash,
    )
    result = engine.run(candles)
    print(f"predictor={type(predictor).__name__}")
    print(f"starting_cash={result.starting_cash:.2f}")
    print(f"ending_equity={result.ending_equity:.2f}")
    print(f"total_return_pct={result.total_return_pct:.2f}")
    print(f"fills={len(result.fills)}")


def run_paper_once(args: argparse.Namespace) -> None:
    settings = load_settings()
    candles = _load_candles_for_runtime(args)
    predictor, strategy, risk_manager = build_components(args)
    broker = PaperBroker(cash=args.cash if args.cash is not None else settings.initial_cash)
    agent = TradingAgent(
        predictor=predictor,
        strategy=strategy,
        risk_manager=risk_manager,
        broker=broker,
    )
    decision = agent.run_once(candles)
    print(f"predictor={type(predictor).__name__}")
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


def run_auto_train(args: argparse.Namespace) -> None:
    config = AutoTrainConfig(
        symbol=args.symbol,
        output_path=args.output,
        data_source=args.data_source,
        interval=args.interval,
        limit=args.limit,
        days=args.days,
        vs_currency=args.vs_currency,
        coin_id=args.coin_id,
        csv_path=args.csv,
        lookback=args.lookback,
        horizon=args.horizon,
        label_threshold=args.label_threshold,
        train_fraction=args.train_fraction,
    )

    if not args.loop:
        try:
            report = auto_train_once(config)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(report.summary(), indent=2, default=str))
        return

    interval_seconds = max(1, args.interval_minutes) * 60
    iterations = args.iterations if args.iterations and args.iterations > 0 else None
    try:
        for report in auto_train_loop(
            config,
            interval_seconds=interval_seconds,
            iterations=iterations,
            sleep=time.sleep,
        ):
            print(json.dumps(report.summary(), indent=2, default=str))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("auto-train loop interrupted by user")


def run_live_snapshot(args: argparse.Namespace) -> None:
    try:
        snapshot = fetch_live_market(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            days=args.days,
            vs_currency=args.vs_currency,
            coin_id=args.coin_id,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    summary = {
        "source": snapshot.source,
        "fallbacks": snapshot.fallbacks,
        "candles": len(snapshot.candles),
        "ticker": snapshot.ticker.to_dict(),
    }
    print(json.dumps(summary, indent=2, default=str))


def run_autonomous(args: argparse.Namespace) -> None:
    symbols, symbol_intervals = split_symbols(args.symbols)
    if not symbols:
        raise SystemExit("--symbols must list at least one market, e.g. BTCUSDT@1m")
    config = AutonomousConfig(
        symbols=symbols,
        symbol_intervals=symbol_intervals,
        starting_cash=args.cash,
        fee_rate=args.fee_rate,
        predictor_name=args.predictor,
        model_path=args.model_path,
        decision_interval_seconds=args.decision_interval_seconds,
        stream_interval_seconds=args.stream_interval_seconds,
        candle_window=args.candle_window,
        candle_interval=args.candle_interval,
        klines_limit=args.klines_limit,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        quote_per_trade=args.quote_per_trade,
        max_position_quote=args.max_position_quote,
        self_train_enabled=not args.no_self_train,
        self_train_interval_minutes=args.self_train_interval_minutes,
        self_train_days=args.self_train_days,
    )
    run_until_interrupt(config)


def _load_candles_for_runtime(args: argparse.Namespace) -> list:
    source = getattr(args, "data_source", "csv").lower()
    if source == "binance":
        return load_binance_klines(
            symbol=args.symbol,
            interval=args.interval,
            limit=limit_for_days(args.days, args.interval) if args.days else args.limit,
        )
    if source == "coingecko":
        return load_coingecko_ohlc(
            symbol=args.symbol,
            coin_id=args.coin_id,
            vs_currency=args.vs_currency,
            days=args.days,
        )
    if source == "live":
        return fetch_live_market(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            days=args.days,
            vs_currency=args.vs_currency,
            coin_id=args.coin_id,
        ).candles
    if not getattr(args, "csv", None):
        raise SystemExit("--csv is required when --data-source csv")
    return load_candles(Path(args.csv), args.symbol)


def _add_data_source_args(sub: argparse.ArgumentParser, *, default: str = "csv") -> None:
    sub.add_argument(
        "--data-source",
        choices=["csv", "coingecko", "binance", "live"],
        default=default,
    )
    sub.add_argument("--csv", default=None, help="Path to OHLCV CSV (when data-source=csv)")
    sub.add_argument("--coin-id", default=None, help="CoinGecko coin id, for example bitcoin")
    sub.add_argument("--vs-currency", default=DEFAULT_VS_CURRENCY)
    sub.add_argument("--days", type=int, default=DEFAULT_DAYS)
    sub.add_argument("--interval", default=DEFAULT_INTERVAL)
    sub.add_argument("--limit", type=int, default=DEFAULT_LIMIT)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous trading agent starter CLI")
    subparsers = parser.add_subparsers(required=True)

    train = subparsers.add_parser("train", help="Train a supervised prediction model")
    _add_data_source_args(train, default="csv")
    train.add_argument("--symbol", default=load_settings().symbol)
    train.add_argument("--output", default="models/latest_model.json")
    train.add_argument("--lookback", type=int, default=10)
    train.add_argument("--horizon", type=int, default=3)
    train.add_argument("--label-threshold", type=float, default=0.01)
    train.add_argument("--train-fraction", type=float, default=0.8)
    train.set_defaults(func=run_train)

    auto = subparsers.add_parser(
        "auto-train",
        help="Self-training: fetch live data, retrain, save. Optional --loop for periodic retraining.",
    )
    _add_data_source_args(auto, default="binance")
    auto.add_argument("--symbol", default=load_settings().symbol)
    auto.add_argument("--output", default="models/latest_auto_nb.json")
    auto.add_argument("--lookback", type=int, default=10)
    auto.add_argument("--horizon", type=int, default=3)
    auto.add_argument("--label-threshold", type=float, default=0.005)
    auto.add_argument("--train-fraction", type=float, default=0.8)
    auto.add_argument("--loop", action="store_true", help="Retrain on a schedule")
    auto.add_argument("--interval-minutes", type=int, default=60)
    auto.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="0 means run forever when --loop is set",
    )
    auto.set_defaults(func=run_auto_train)

    backtest = subparsers.add_parser("backtest", help="Run a historical backtest")
    _add_data_source_args(backtest, default="csv")
    backtest.add_argument("--symbol", default=load_settings().symbol)
    backtest.add_argument("--cash", type=float, default=None)
    backtest.add_argument("--model-path", default=None, help="Use a trained JSON model")
    backtest.add_argument("--predictor", choices=list(PREDICTOR_NAMES), default=PREDICTOR_AUTO)
    backtest.set_defaults(func=run_backtest)

    paper_once = subparsers.add_parser("paper-once", help="Run one paper trading decision")
    _add_data_source_args(paper_once, default="live")
    paper_once.add_argument("--symbol", default=load_settings().symbol)
    paper_once.add_argument("--cash", type=float, default=None)
    paper_once.add_argument("--model-path", default=None, help="Use a trained JSON model")
    paper_once.add_argument("--predictor", choices=list(PREDICTOR_NAMES), default=PREDICTOR_AUTO)
    paper_once.set_defaults(func=run_paper_once)

    live = subparsers.add_parser("live", help="Print a live market snapshot for a symbol")
    live.add_argument("--symbol", default=load_settings().symbol)
    live.add_argument("--interval", default=DEFAULT_INTERVAL)
    live.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    live.add_argument("--days", type=int, default=None)
    live.add_argument("--vs-currency", default=DEFAULT_VS_CURRENCY)
    live.add_argument("--coin-id", default=None)
    live.set_defaults(func=run_live_snapshot)

    autonomous = subparsers.add_parser(
        "autonomous",
        help=(
            "Run the autonomous agent: live USDT ticker stream, paper exchange with "
            "market+limit orders, predictor decisions, and periodic self-training."
        ),
    )
    autonomous.add_argument(
        "--symbols",
        default="BTCUSDT",
        help=(
            "Comma- or space-separated USDT symbols. Append @<interval> to override the "
            "candle interval per coin, e.g. BTCUSDT@1m,ETHUSDT@5m,SOLUSDT@1h."
        ),
    )
    autonomous.add_argument("--cash", type=float, default=10_000.0)
    autonomous.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    autonomous.add_argument("--predictor", choices=list(PREDICTOR_NAMES), default=PREDICTOR_AUTO)
    autonomous.add_argument("--model-path", default="models/btcusd_auto_nb.json")
    autonomous.add_argument("--decision-interval-seconds", type=float, default=30.0)
    autonomous.add_argument("--stream-interval-seconds", type=float, default=2.0)
    autonomous.add_argument("--candle-window", type=int, default=200)
    autonomous.add_argument(
        "--candle-interval",
        default=DEFAULT_INTERVAL,
        help="Default candle interval for symbols without an @<interval> override.",
    )
    autonomous.add_argument("--klines-limit", type=int, default=DEFAULT_LIMIT)
    autonomous.add_argument("--buy-threshold", type=float, default=0.2)
    autonomous.add_argument("--sell-threshold", type=float, default=-0.2)
    autonomous.add_argument("--quote-per-trade", type=float, default=200.0)
    autonomous.add_argument("--max-position-quote", type=float, default=2_500.0)
    autonomous.add_argument(
        "--no-self-train",
        action="store_true",
        help="Disable the periodic retraining thread",
    )
    autonomous.add_argument("--self-train-interval-minutes", type=int, default=60)
    autonomous.add_argument("--self-train-days", type=int, default=30)
    autonomous.set_defaults(func=run_autonomous)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
