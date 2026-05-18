# Connect Lovable to the Python Agent

The Python agent is the backend. Lovable or this React app is the frontend.

## Architecture

```text
Lovable / React UI
  -> HTTPS fetch
Python API (Render or PythonAnywhere)
  -> trained model JSON (auto-trained from live Binance klines)
  -> paper trading and backtest logic
```

## Backend defaults

The agent prefers Binance for market data (live ticker plus klines). CoinGecko
is kept only as a fallback for hosts that Binance blocks (HTTP 451). The frontend
should default to `data_source: "live"` so the backend can pick the best feed
automatically. `data_source: "binance"` and `data_source: "coingecko"` are
available when you want to pin a specific provider.

## Sanity checks

```text
https://YOUR_HOST/health
https://YOUR_HOST/model/status
https://YOUR_HOST/predictors
https://YOUR_HOST/market/live?symbol=BTCUSDT
```

If `/model/status` returns `exists: true`, an auto-trained model is available.
If it returns `exists: false`, the predictor falls back to the multi-indicator
predictor (RSI / MACD / EMA / Bollinger %B / momentum) until the next train.

## Lovable prompt

Paste this into Lovable:

```text
Build a sleek autonomous trading dashboard that connects to my external Python API.

Use this base URL:
https://YOUR_HOST

Create controls for:
- GET /health and /model/status sanity checks
- GET /predictors so users can pick auto / baseline / multi / ml / llm
- GET /market/live?symbol=<SYMBOL> for the live ticker plus indicator snapshot
- POST /predict with JSON body {"symbol":"BTCUSDT","data_source":"live","predictor":"auto"}
- POST /paper/run-once with JSON body {"symbol":"BTCUSDT","data_source":"live","predictor":"auto"}
- POST /backtest with JSON body {"symbol":"BTCUSDT","data_source":"live","predictor":"auto"}
- POST /train/auto with JSON body {"symbol":"BTCUSDT","data_source":"binance","days":30}

Show prediction confidence, direction_score, buy/sell/hold signal, risk approval,
paper fill, and backtest return. Show RSI, MACD histogram, EMA 12/26 and Bollinger %B
from the live snapshot. Do not store broker keys in the frontend.

Allow users to override per-coin candle intervals when calling auto-train, e.g.
BTCUSDT@1m, ETHUSDT@5m, SOLUSDT@1h.

Do not keep stale previous results after a failed fetch.
Clear prediction, paper, and backtest panels when a new request starts.
```

## If Lovable asks for code

```ts
const API_BASE_URL = "https://YOUR_HOST";

async function predict(symbol: string) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol,
      data_source: "live",
      predictor: "auto",
      days: 30,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}
```

## CORS

The Python backend ships with browser CORS headers. For early demos it allows
all origins. For production, set:

```text
ALLOWED_ORIGINS=https://your-lovable-domain.lovable.app
```

on Render or your Python host.

## Deployment note

Lovable is a frontend builder. Keep Python model training, paper trading,
broker keys and private logic on Render or another Python host.
