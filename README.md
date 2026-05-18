# Autonomous AI Trading Agent Starter

Safety-first starter for building an autonomous crypto trading research agent. Wired
exclusively for **backtesting** and **paper trading**. Live broker execution is
intentionally not included.

## What is included

- CSV market data loader plus CoinGecko, Binance, and a "live" Binance + CoinGecko fallback feed.
- Multiple predictors, all sharing the same `predict(candles)` shape:
  - moving-average momentum baseline
  - multi-indicator predictor combining RSI, MACD, EMA cross, Bollinger %B, momentum
  - trainable pure-Python Gaussian Naive Bayes model (JSON artifacts in `models/`)
  - optional Anthropic Claude predictor (`llm` extra)
- Threshold strategy that turns predictions into BUY / SELL / HOLD signals.
- Risk manager with position sizing, order caps, long-only default.
- **Paper exchange simulator** (`broker/exchange_simulator.py`) with multi-asset
  cash + holdings, market and limit orders, a matching engine, and a 0.1% fee.
- **Autonomous runner** (`autonomous/runner.py`) that streams every USDT spot pair
  from Binance, decides on a fixed cadence, places orders into the simulator, and
  retrains the JSON model in a background thread.
- Backtest engine with an equity curve.
- CLI commands: `train`, `auto-train`, `backtest`, `paper-once`, `live`, `autonomous`.
- FastAPI service for Render and website integration.
- Unit tests for indicators, predictors, training, paper broker, exchange simulator, and the API.

## Quick start

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

### Run the autonomous agent

Live USDT ticker stream + paper exchange + predictor + periodic self-training, all in one command:

```powershell
$env:PYTHONPATH="src"
python -m trading_agent.cli autonomous --symbols BTCUSDT,ETHUSDT --cash 10000
```

Useful flags:

- `--predictor auto|baseline|multi|ml|llm` (default `auto`).
- `--decision-interval-seconds 30` how often the agent re-evaluates each symbol.
- `--quote-per-trade 200` USD-equivalent size per market order.
- `--no-self-train` to disable the retraining thread.
- `--self-train-interval-minutes 60` retraining cadence.

### Use the Anthropic Claude predictor

```powershell
pip install -e ".[llm]"
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:PYTHONPATH="src"
python -m trading_agent.cli autonomous --symbols BTCUSDT --predictor llm
```

`ANTHROPIC_MODEL` (default `claude-haiku-4-5`) lets you switch to Sonnet or Opus.

### Other CLI commands

```powershell
python scripts/generate_demo_prices.py
python -m trading_agent.cli train --csv examples/mixed_training_prices.csv --symbol BTCUSD --output models/btcusd_demo_nb.json --label-threshold 0.005
python -m trading_agent.cli auto-train --data-source binance --symbol BTCUSDT --output models/btcusd_auto_nb.json --loop --interval-minutes 60
python -m trading_agent.cli backtest --csv examples/sample_prices.csv --symbol BTCUSD --predictor multi
python -m trading_agent.cli paper-once --data-source live --symbol BTCUSDT --predictor auto
python -m trading_agent.cli live --symbol BTCUSDT
```

## Run as an API

```powershell
$env:PYTHONPATH="src"
uvicorn trading_agent.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API. The Render start command is:

```bash
uvicorn trading_agent.api:app --host 0.0.0.0 --port $PORT
```

New endpoints:

- `GET /predictors` — list available predictors and whether the LLM is available.
- `GET /market/live` — current ticker plus indicator snapshot.
- `POST /train/auto` — train a new JSON model from a chosen feed.

## Project layout

```text
src/trading_agent/
  agent.py                          Autonomous decision loop for one cycle
  api.py                            FastAPI service for Render
  autonomous/runner.py              Live stream + simulator + predictor + self-training
  backtest/engine.py                Historical simulation
  broker/paper.py                   Single-symbol paper broker
  broker/exchange_simulator.py      Multi-asset paper exchange with matching engine and fees
  config.py                         Environment-based settings
  data/csv_feed.py                  CSV candle loader
  data/coingecko_feed.py            CoinGecko OHLC fetcher
  data/binance_feed.py              Binance klines fetcher
  data/binance_stream.py            REST polling stream for every USDT spot pair
  data/live_feed.py                 Unified live feed (Binance, falls back to CoinGecko)
  features/indicators.py            SMA, EMA, RSI, MACD, ATR, Bollinger, snapshot helper
  prediction/baseline.py            Moving-average momentum
  prediction/multi_indicator.py     Combined RSI/MACD/EMA/Bollinger predictor
  prediction/ml.py                  JSON-trained Naive Bayes adapter
  prediction/llm.py                 Anthropic Claude predictor (optional, llm extra)
  prediction/factory.py             Pick a predictor by name
  risk/manager.py
  strategy/threshold.py
  training/                         Feature, label, train, evaluate, save
  training/auto.py                  Self-training orchestrator
```

## Safety notes

This is research code, not financial advice. Backtest performance does not imply
live performance. Start with historical validation, then paper trading. Live broker
execution is out of scope for this starter.
