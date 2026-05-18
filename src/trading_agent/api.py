from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trading_agent.agent import TradingAgent
from trading_agent.backtest.engine import BacktestEngine
from trading_agent.broker.paper import PaperBroker
from trading_agent.config import load_settings
from trading_agent.data.binance_feed import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    limit_for_days,
    load_binance_klines,
)
from trading_agent.data.coingecko_feed import (
    DEFAULT_DAYS,
    DEFAULT_VS_CURRENCY,
    load_coingecko_ohlc,
    search_coins,
)
from trading_agent.data.csv_feed import load_candles
from trading_agent.data.live_feed import fetch_live_market
from trading_agent.features.indicators import compute_indicator_snapshot
from trading_agent.models import Candle
from trading_agent.prediction.factory import (
    PREDICTOR_AUTO,
    PREDICTOR_NAMES,
    build_predictor,
)
from trading_agent.prediction.llm import LLMConfigurationError
from trading_agent.prediction.protocols import Predictor
from trading_agent.risk.manager import RiskManager
from trading_agent.strategy.threshold import ThresholdStrategy
from trading_agent.training.auto import AutoTrainConfig, auto_train_once


app = FastAPI(
    title="Autonomous Trading Agent API",
    version="0.2.0",
    description="Paper-trading and backtesting API for the autonomous trading agent starter.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CandlePayload(BaseModel):
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class MarketRequest(BaseModel):
    symbol: str = "BTCUSD"
    candles: list[CandlePayload] | None = None
    csv_path: str | None = None
    data_source: str = "binance"
    coin_id: str | None = None
    vs_currency: str = DEFAULT_VS_CURRENCY
    days: int = Field(default=DEFAULT_DAYS, ge=1, le=365)
    interval: str = DEFAULT_INTERVAL
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=1000)
    model_path: str | None = None
    cash: float | None = Field(default=None, gt=0)
    predictor: str = PREDICTOR_AUTO


class AutoTrainRequest(BaseModel):
    symbol: str = "BTCUSD"
    output_path: str = "models/btcusd_auto_nb.json"
    data_source: str = "binance"
    interval: str = DEFAULT_INTERVAL
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=1000)
    days: int = Field(default=DEFAULT_DAYS, ge=1, le=365)
    vs_currency: str = DEFAULT_VS_CURRENCY
    coin_id: str | None = None
    csv_path: str | None = None
    lookback: int = Field(default=10, ge=10)
    horizon: int = Field(default=3, ge=1)
    label_threshold: float = Field(default=0.005, gt=0)
    train_fraction: float = Field(default=0.8, gt=0, lt=1)


class ModelStatusResponse(BaseModel):
    model_path: str
    exists: bool
    message: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "autonomous-trading-agent-api",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    model_path = _default_model_path()
    exists = model_path.exists()
    message = "model is ready" if exists else "model file is missing; train before using ML mode"
    return ModelStatusResponse(model_path=str(model_path), exists=exists, message=message)


@app.get("/predictors")
def list_predictors() -> dict[str, object]:
    return {
        "predictors": list(PREDICTOR_NAMES),
        "default": PREDICTOR_AUTO,
        "llm_available": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


@app.get("/market/live")
def market_live(
    symbol: str = "BTCUSD",
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    days: int | None = None,
    vs_currency: str = DEFAULT_VS_CURRENCY,
    coin_id: str | None = None,
) -> dict[str, object]:
    try:
        snapshot = fetch_live_market(
            symbol=symbol,
            interval=interval,
            limit=limit,
            days=days,
            vs_currency=vs_currency,
            coin_id=coin_id,
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    closes = [candle.close for candle in snapshot.candles]
    highs = [candle.high for candle in snapshot.candles]
    lows = [candle.low for candle in snapshot.candles]
    indicators = compute_indicator_snapshot(closes, highs, lows).to_dict() if closes else {}

    return {
        "source": snapshot.source,
        "fallbacks": snapshot.fallbacks,
        "symbol": symbol,
        "interval": interval,
        "candles": len(snapshot.candles),
        "ticker": snapshot.ticker.to_dict(),
        "indicators": indicators,
    }


@app.get("/market/ohlc")
def market_ohlc(
    symbol: str = "BTCUSD",
    coin_id: str | None = None,
    vs_currency: str = DEFAULT_VS_CURRENCY,
    days: int = DEFAULT_DAYS,
    data_source: str = "binance",
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, object]:
    if data_source.lower() == "binance":
        try:
            candles = load_binance_klines(
                symbol=symbol,
                interval=interval,
                limit=limit_for_days(days, interval) if days else limit,
            )
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        latest = candles[-1]
        return {
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

    try:
        candles = load_coingecko_ohlc(
            symbol=symbol,
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latest = candles[-1]
    return {
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


@app.get("/market/coins")
def market_coins(query: str = "", limit: int = 100) -> dict[str, object]:
    try:
        coins = search_coins(query=query, limit=max(1, min(limit, 250)))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "source": "coingecko",
        "query": query,
        "count": len(coins),
        "coins": coins,
    }


@app.post("/predict")
def predict(request: MarketRequest) -> dict[str, object]:
    candles = _resolve_candles(request)
    predictor = _build_predictor_for_request(request)

    try:
        prediction = predictor.predict(candles)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latest = candles[-1]
    return {
        "market_source": request.data_source.lower(),
        "predictor": _predictor_label(predictor),
        "candle_count": len(candles),
        "latest_price": latest.close,
        "latest_timestamp": latest.timestamp.isoformat(),
        "symbol": prediction.symbol,
        "direction_score": prediction.direction_score,
        "confidence": prediction.confidence,
        "horizon_candles": prediction.horizon_candles,
        "rationale": prediction.rationale,
    }


@app.post("/paper/run-once")
def paper_run_once(request: MarketRequest) -> dict[str, object]:
    settings = load_settings()
    candles = _resolve_candles(request)
    predictor = _build_predictor_for_request(request)
    broker = PaperBroker(cash=request.cash if request.cash is not None else settings.initial_cash)
    agent = TradingAgent(
        predictor=predictor,
        strategy=_build_strategy(),
        risk_manager=_build_risk_manager(),
        broker=broker,
    )

    try:
        decision = agent.run_once(candles)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latest = candles[-1]
    return {
        "market_source": request.data_source.lower(),
        "predictor": _predictor_label(predictor),
        "candle_count": len(candles),
        "latest_price": latest.close,
        "latest_timestamp": latest.timestamp.isoformat(),
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
        "fill": None
        if decision.fill is None
        else {
            "symbol": decision.fill.symbol,
            "side": decision.fill.side.value,
            "quantity": decision.fill.quantity,
            "price": decision.fill.price,
            "notional": decision.fill.notional,
            "reason": decision.fill.reason,
        },
        "paper_cash": broker.cash,
    }


@app.post("/backtest")
def backtest(request: MarketRequest) -> dict[str, object]:
    settings = load_settings()
    candles = _resolve_candles(request)
    engine = BacktestEngine(
        predictor=_build_predictor_for_request(request),
        strategy=_build_strategy(),
        risk_manager=_build_risk_manager(),
        starting_cash=request.cash if request.cash is not None else settings.initial_cash,
    )

    try:
        result = engine.run(candles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latest = candles[-1]
    return {
        "market_source": request.data_source.lower(),
        "candle_count": len(candles),
        "latest_price": latest.close,
        "latest_timestamp": latest.timestamp.isoformat(),
        "starting_cash": result.starting_cash,
        "ending_equity": result.ending_equity,
        "total_return_pct": result.total_return_pct,
        "fills": len(result.fills),
        "equity_points": len(result.equity_curve),
        "last_equity_point": None if not result.equity_curve else result.equity_curve[-1].__dict__,
    }


@app.post("/train/auto")
def train_auto(request: AutoTrainRequest) -> dict[str, object]:
    config = AutoTrainConfig(
        symbol=request.symbol,
        output_path=request.output_path,
        data_source=request.data_source,
        interval=request.interval,
        limit=request.limit,
        days=request.days,
        vs_currency=request.vs_currency,
        coin_id=request.coin_id,
        csv_path=request.csv_path,
        lookback=request.lookback,
        horizon=request.horizon,
        label_threshold=request.label_threshold,
        train_fraction=request.train_fraction,
    )
    try:
        report = auto_train_once(config)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report.summary()


def _resolve_candles(request: MarketRequest) -> list[Candle]:
    if request.candles:
        return [
            Candle(
                timestamp=candle.timestamp,
                symbol=request.symbol,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            for candle in request.candles
        ]

    source = request.data_source.lower()
    if source == "binance" and request.csv_path is None:
        try:
            return load_binance_klines(
                symbol=request.symbol,
                interval=request.interval,
                limit=limit_for_days(request.days, request.interval)
                if request.days
                else request.limit,
            )
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if source == "coingecko" and request.csv_path is None:
        try:
            return load_coingecko_ohlc(
                symbol=request.symbol,
                coin_id=request.coin_id,
                vs_currency=request.vs_currency,
                days=request.days,
            )
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if source == "live" and request.csv_path is None:
        try:
            snapshot = fetch_live_market(
                symbol=request.symbol,
                interval=request.interval,
                limit=request.limit,
                days=request.days,
                vs_currency=request.vs_currency,
                coin_id=request.coin_id,
            )
            return snapshot.candles
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    csv_path = Path(request.csv_path) if request.csv_path else _default_csv_path()
    try:
        return load_candles(csv_path, request.symbol)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_predictor_for_request(request: MarketRequest) -> Predictor:
    model_path = Path(request.model_path) if request.model_path else _default_model_path()
    try:
        return build_predictor(
            request.predictor,
            model_path=model_path,
            symbol=request.symbol,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _predictor_label(predictor: Predictor) -> str:
    return type(predictor).__name__


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
    return Path(os.getenv("DEFAULT_MODEL_PATH", "models/btcusd_auto_nb.json"))


def _default_csv_path() -> Path:
    return Path(os.getenv("DEFAULT_CSV_PATH", "examples/mixed_training_prices.csv"))
