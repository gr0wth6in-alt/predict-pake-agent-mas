# Autonomous AI Trading Agent Starter

This is a safety-first starter project for building an autonomous trading research agent. It is intentionally wired for backtesting and paper trading only. Add real broker execution later only after you have strong tests, monitoring, and risk controls.

## What is included

- CSV market data loader
- Baseline moving-average momentum predictor
- Trainable pure-Python Gaussian Naive Bayes predictor
- Threshold strategy that turns predictions into signals
- Risk manager with position sizing, order caps, and long-only default
- Paper broker simulator
- Backtest engine with an equity curve
- CLI commands for backtest and a single paper decision
- FastAPI service for Render and website integration
- Unit tests for the core trading loop

## Quick start

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m unittest discover -s tests
python scripts/generate_demo_prices.py
python -m trading_agent.cli train --csv examples/mixed_training_prices.csv --symbol BTCUSD --output models/btcusd_demo_nb.json --label-threshold 0.005
python -m trading_agent.cli train --data-source coingecko --symbol BTCUSD --coin-id bitcoin --days 365 --output models/btcusd_coingecko_nb.json --label-threshold 0.005
python -m trading_agent.cli backtest --csv examples/sample_prices.csv --symbol BTCUSD
python -m trading_agent.cli backtest --csv examples/mixed_training_prices.csv --symbol BTCUSD --model-path models/btcusd_demo_nb.json
python -m trading_agent.cli paper-once --csv examples/sample_prices.csv --symbol BTCUSD
```

The trainable model is deliberately simple and inspectable before you add larger ML libraries or exchange integrations.
For real crypto data, the app can fetch CoinGecko OHLC data through the Python backend. Live broker execution is intentionally not included.

## Run as an API

```powershell
$env:PYTHONPATH="src"
uvicorn trading_agent.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API. The Render start command is:

```bash
uvicorn trading_agent.api:app --host 0.0.0.0 --port $PORT
```

## Project layout

```text
src/trading_agent/
  agent.py              Autonomous decision loop for one cycle
  api.py                FastAPI service for Render
  backtest/engine.py    Historical simulation
  broker/paper.py       Paper trading broker
  config.py             Environment-based settings
  data/csv_feed.py      CSV candle loader
  features/indicators.py
  prediction/baseline.py
  prediction/ml.py       JSON-trained model adapter
  risk/manager.py
  strategy/threshold.py
  training/              Feature, label, train, evaluate, save
```

## How to continue

1. Train the included model with more realistic historical data.
2. Add richer features in `src/trading_agent/features/`.
3. Keep every model behind the `predict(candles)` shape used by `TradingAgent`.
4. Use `BacktestEngine` before paper trading any new strategy.
5. Keep live broker execution in a separate adapter and default it to disabled.

## Safety notes

This project is not financial advice. Prediction quality in a backtest does not guarantee live performance. Start with historical validation, then paper trading, then tiny supervised live tests only if you fully understand the risks.
