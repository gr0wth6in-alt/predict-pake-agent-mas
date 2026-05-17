from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from trading_agent.agent import TradingAgent
from trading_agent.backtest.engine import BacktestEngine
from trading_agent.broker.paper import PaperBroker
from trading_agent.config import load_settings
from trading_agent.data.binance_feed import DEFAULT_INTERVAL, DEFAULT_LIMIT, limit_for_days, load_binance_klines
from trading_agent.data.coingecko_feed import (
    DEFAULT_DAYS,
    DEFAULT_VS_CURRENCY,
    load_coingecko_ohlc,
    search_coins,
)
from trading_agent.data.csv_feed import load_candles
from trading_agent.prediction.baseline import MovingAverageMomentumPredictor
from trading_agent.prediction.ml import TrainedModelPredictor
from trading_agent.prediction.protocols import Predictor
from trading_agent.risk.manager import RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy


app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")}},
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)


@app.after_request
def add_cors_headers(response: Any) -> Any:
    response.headers["Access-Control-Allow-Origin"] = os.getenv("ALLOWED_ORIGINS", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_preflight(path: str) -> Any:
    return "", 204


@app.get("/")
def root() -> Any:
    return jsonify(
        {
            "name": "autonomous-trading-agent-wsgi",
            "status": "ok",
            "docs": "Use /health, /model/status, /predict, /paper/run-once, /backtest",
        }
    )


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/model/status")
def model_status() -> Any:
    model_path = _default_model_path()
    return jsonify(
        {
            "model_path": str(model_path),
            "exists": model_path.exists(),
            "message": "model is ready" if model_path.exists() else "model file is missing",
        }
    )


@app.get("/market/ohlc")
def market_ohlc() -> Any:
    symbol = request.args.get("symbol", os.getenv("SYMBOL", "BTCUSDT"))
    data_source = request.args.get("data_source", "binance").lower()
    interval = request.args.get("interval", DEFAULT_INTERVAL)
    limit = int(request.args.get("limit", DEFAULT_LIMIT))
    days = int(request.args.get("days", DEFAULT_DAYS))

    if data_source == "binance":
        try:
            candles = load_binance_klines(
                symbol=symbol,
                interval=interval,
                limit=limit_for_days(days, interval) if days else limit,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        latest = candles[-1]
        return jsonify(
            {
                "source": "binance",
                "symbol": latest.symbol,
                "interval": interval,
                "candles": len(candles),
                "latest": {
                    "timestamp": latest.timestamp.isoformat(),
                    "open": latest.open,
                    "high": latest.high,
                    "low": latest.low,
                    "close": latest.close,
                    "volume": latest.volume,
                },
            }
        )

    coin_id = request.args.get("coin_id")
    vs_currency = request.args.get("vs_currency", DEFAULT_VS_CURRENCY)

    try:
        candles = load_coingecko_ohlc(
            symbol=symbol,
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    latest = candles[-1]
    return jsonify(
        {
            "source": "coingecko",
            "symbol": latest.symbol,
            "coin_id": coin_id,
            "vs_currency": vs_currency,
            "days": days,
            "candles": len(candles),
            "latest": {
                "timestamp": latest.timestamp.isoformat(),
                "open": latest.open,
                "high": latest.high,
                "low": latest.low,
                "close": latest.close,
                "volume": latest.volume,
            },
        }
    )


@app.get("/market/coins")
def market_coins() -> Any:
    query = request.args.get("query", "")
    limit = max(1, min(int(request.args.get("limit", 100)), 250))

    try:
        coins = search_coins(query=query, limit=limit)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "source": "coingecko",
            "query": query,
            "count": len(coins),
            "coins": coins,
        }
    )


@app.post("/predict")
def predict() -> Any:
    payload = _json_payload()
    try:
        candles = _resolve_candles(payload)
        prediction = _build_predictor(payload.get("model_path")).predict(candles)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "symbol": prediction.symbol,
            "direction_score": prediction.direction_score,
            "confidence": prediction.confidence,
            "horizon_candles": prediction.horizon_candles,
            "rationale": prediction.rationale,
        }
    )


@app.post("/paper/run-once")
def paper_run_once() -> Any:
    payload = _json_payload()
    settings = load_settings()

    try:
        candles = _resolve_candles(payload)
        broker = PaperBroker(cash=float(payload.get("cash") or settings.initial_cash))
        agent = TradingAgent(
            predictor=_build_predictor(payload.get("model_path")),
            strategy=_build_strategy(),
            risk_manager=_build_risk_manager(),
            broker=broker,
        )
        decision = agent.run_once(candles)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    fill = None
    if decision.fill is not None:
        fill = {
            "symbol": decision.fill.symbol,
            "side": decision.fill.side.value,
            "quantity": decision.fill.quantity,
            "price": decision.fill.price,
            "notional": decision.fill.notional,
            "reason": decision.fill.reason,
        }

    return jsonify(
        {
            "signal": {
                "symbol": decision.signal.symbol,
                "side": decision.signal.side.value,
                "strength": decision.signal.strength,
                "reason": decision.signal.reason,
            },
            "risk": {
                "approved": decision.risk_decision.approved,
                "reason": decision.risk_decision.reason,
            },
            "fill": fill,
            "paper_cash": broker.cash,
        }
    )


@app.post("/backtest")
def backtest() -> Any:
    payload = _json_payload()
    settings = load_settings()

    try:
        candles = _resolve_candles(payload)
        engine = BacktestEngine(
            predictor=_build_predictor(payload.get("model_path")),
            strategy=_build_strategy(),
            risk_manager=_build_risk_manager(),
            starting_cash=float(payload.get("cash") or settings.initial_cash),
        )
        result = engine.run(candles)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "starting_cash": result.starting_cash,
            "ending_equity": result.ending_equity,
            "total_return_pct": result.total_return_pct,
            "fills": len(result.fills),
            "equity_points": len(result.equity_curve),
            "last_equity_point": None
            if not result.equity_curve
            else result.equity_curve[-1].__dict__,
        }
    )


def _json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _resolve_candles(payload: dict[str, Any]) -> Any:
    symbol = str(payload.get("symbol") or os.getenv("SYMBOL", "BTCUSDT"))
    data_source = str(payload.get("data_source") or "binance").lower()

    if data_source == "binance" and not payload.get("csv_path"):
        interval = str(payload.get("interval") or DEFAULT_INTERVAL)
        days = int(payload.get("days") or DEFAULT_DAYS)
        limit = int(payload.get("limit") or DEFAULT_LIMIT)
        return load_binance_klines(
            symbol=symbol,
            interval=interval,
            limit=limit_for_days(days, interval) if days else limit,
        )

    if data_source == "coingecko" and not payload.get("csv_path"):
        return load_coingecko_ohlc(
            symbol=symbol,
            coin_id=payload.get("coin_id"),
            vs_currency=str(payload.get("vs_currency") or DEFAULT_VS_CURRENCY),
            days=int(payload.get("days") or DEFAULT_DAYS),
        )

    csv_path = Path(str(payload.get("csv_path"))) if payload.get("csv_path") else _default_csv_path()
    return load_candles(csv_path, symbol)


def _build_predictor(model_path: str | None) -> Predictor:
    settings = load_settings()
    resolved_model_path = Path(model_path) if model_path else _default_model_path()
    if resolved_model_path.exists():
        return TrainedModelPredictor.load(resolved_model_path)

    return MovingAverageMomentumPredictor(
        short_window=settings.prediction_short_window,
        long_window=settings.prediction_long_window,
    )


def _build_strategy() -> ThresholdStrategy:
    settings = load_settings()
    return ThresholdStrategy(threshold=settings.prediction_threshold)


def _build_risk_manager() -> RiskManager:
    settings = load_settings()
    return RiskManager(
        max_position_fraction=settings.max_position_fraction,
        max_order_notional=settings.max_order_notional,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        allow_short=settings.allow_short,
    )


def _default_model_path() -> Path:
    return Path(os.getenv("DEFAULT_MODEL_PATH", "models/btcusd_demo_nb.json"))


def _default_csv_path() -> Path:
    return Path(os.getenv("DEFAULT_CSV_PATH", "examples/mixed_training_prices.csv"))
