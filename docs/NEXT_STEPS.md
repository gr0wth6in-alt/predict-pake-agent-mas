# Next Steps

Use this checklist to turn the starter into your own autonomous AI trading agent.

## 1. Data

- Add a real historical data adapter under `src/trading_agent/data/`.
- Normalize every source into the `Candle` model.
- Store raw data separately from processed features.
- Add tests for missing candles, duplicate timestamps, bad prices, and time zone handling.

## 2. Prediction model

- Add your model under `src/trading_agent/prediction/`.
- Keep the public shape as `predict(candles) -> Prediction`.
- Track model version, training data window, feature list, and validation metrics.
- Never tune only against one asset or one market regime.

## 3. Strategy

- Keep strategy logic separate from prediction logic.
- Add slippage, fees, spread, and latency assumptions to backtests.
- Compare the strategy against a no-trade baseline and a buy-and-hold baseline.

## 4. Risk

- Keep `ALLOW_SHORT=false` until you have explicit short-selling tests.
- Add max daily loss, max drawdown, cooldown after losses, and per-symbol exposure limits.
- Require manual approval before moving from paper trading to live trading.

## 5. Operations

- Log every prediction, signal, risk decision, and fill.
- Add alerts for rejected orders, data gaps, model errors, and drawdown.
- Run the agent in paper mode for a meaningful period before any live execution.
