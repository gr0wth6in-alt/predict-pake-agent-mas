# Autonomous AI Trading Agent Starter

This is a safety-first starter project for building an autonomous trading research agent. It is intentionally wired for backtesting and paper trading only. Add real broker execution later only after you have strong tests, monitoring, and risk controls.

## What is included

- CSV market data loader
- Baseline moving-average momentum predictor
- Threshold strategy that turns predictions into signals
- Risk manager with position sizing, order caps, and long-only default
- Paper broker simulator
- Backtest engine with an equity curve
- CLI commands for backtest and a single paper decision
- Unit tests for the core trading loop

## Quick start

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
python -m unittest discover -s tests
python -m trading_agent.cli backtest --csv examples/sample_prices.csv --symbol BTCUSD
python -m trading_agent.cli paper-once --csv examples/sample_prices.csv --symbol BTCUSD
```

No third-party packages are required for the starter. That keeps the core loop easy to inspect before you add ML libraries or exchange integrations.

## Project layout

```text
src/trading_agent/
  agent.py              Autonomous decision loop for one cycle
  backtest/engine.py    Historical simulation
  broker/paper.py       Paper trading broker
  config.py             Environment-based settings
  data/csv_feed.py      CSV candle loader
  features/indicators.py
  prediction/baseline.py
  risk/manager.py
  strategy/threshold.py
```

## How to continue

1. Replace `MovingAverageMomentumPredictor` with your model in `src/trading_agent/prediction/`.
2. Add richer features in `src/trading_agent/features/`.
3. Keep every model behind the `predict(candles)` shape used by `TradingAgent`.
4. Use `BacktestEngine` before paper trading any new strategy.
5. Keep live broker execution in a separate adapter and default it to disabled.

## Safety notes

This project is not financial advice. Prediction quality in a backtest does not guarantee live performance. Start with historical validation, then paper trading, then tiny supervised live tests only if you fully understand the risks.
