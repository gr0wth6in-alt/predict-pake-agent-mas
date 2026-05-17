# Real Data + Paper Trading

The intended setup is:

```text
CoinGecko real OHLC data
  -> train supervised model
  -> predict buy/sell/hold
  -> produce a short rationale
  -> risk manager
  -> paper broker only
```

No live order execution is included.

## Why CoinGecko first

- CoinGecko public API can be used without an API key.
- PythonAnywhere Free allowlists `api.coingecko.com`.
- The `/coins/{id}/ohlc` endpoint returns real candlestick data.

Sources:

- CoinGecko keyless API: https://docs.coingecko.com/docs/keyless-public-api
- CoinGecko OHLC support: https://support.coingecko.com/hc/en-us/articles/4538892425113-How-to-get-Candlestick-OHLC-Kline-data-using-API
- PythonAnywhere allowlist: https://www.pythonanywhere.com/whitelist/

## Train with real CoinGecko data

```bash
PYTHONPATH=src python -m trading_agent.cli train \
  --data-source coingecko \
  --symbol BTCUSD \
  --coin-id bitcoin \
  --days 365 \
  --output models/btcusd_coingecko_nb.json \
  --label-threshold 0.005
```

## Use real CoinGecko data from the API

Frontend request body:

```json
{
  "symbol": "BTCUSD",
  "coin_id": "bitcoin",
  "vs_currency": "usd",
  "days": 30,
  "data_source": "coingecko"
}
```

Endpoints:

```text
GET  /market/coins?query=bitcoin
GET  /market/ohlc?symbol=BTCUSD&coin_id=bitcoin&days=30
POST /predict
POST /paper/run-once
POST /backtest
```

`coin_id` can be any CoinGecko coin id. Use `/market/coins?query=solana` or `/market/coins?query=pepe` to discover ids.

## Reasoning

The current model returns a short rationale such as class probabilities and confidence. That is an audit summary for the decision, not a hidden chain-of-thought and not financial advice.
