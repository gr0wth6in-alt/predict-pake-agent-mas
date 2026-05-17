# Connect Lovable to the Python Agent

The Python agent is the backend. Lovable or this React app is the frontend.

## Architecture

```text
Lovable / React UI
  -> HTTPS fetch
PythonAnywhere Flask API
  -> trained model JSON
  -> paper trading and backtest logic
```

## PythonAnywhere first

Make sure these URLs work:

```text
https://YOUR_USERNAME.pythonanywhere.com/health
https://YOUR_USERNAME.pythonanywhere.com/model/status
```

If `/model/status` returns `exists: true`, the model file is available.

## Lovable prompt

Paste this into Lovable:

```text
Build a sleek trading dashboard frontend that connects to my external Python API.

Use this base URL:
https://YOUR_USERNAME.pythonanywhere.com

Create buttons for:
- GET /model/status
- GET /market/coins?query=bitcoin
- GET /market/ohlc?symbol=BTCUSD&coin_id=bitcoin&days=30
- POST /predict with JSON body {"symbol":"BTCUSD","coin_id":"bitcoin","vs_currency":"usd","days":30,"data_source":"coingecko"}
- POST /paper/run-once with JSON body {"symbol":"BTCUSD","coin_id":"bitcoin","vs_currency":"usd","days":30,"data_source":"coingecko"}
- POST /backtest with JSON body {"symbol":"BTCUSD","coin_id":"bitcoin","vs_currency":"usd","days":30,"data_source":"coingecko"}

Show prediction confidence, direction_score, buy/sell/hold signal, risk approval, paper fill, and backtest return.
Do not store broker keys in the frontend.
Use CoinGecko as the only market data source.
Allow users to choose common coins and enter any CoinGecko coin id manually.
```

## If Lovable asks for code

Use this fetch pattern:

```ts
const API_BASE_URL = "https://YOUR_USERNAME.pythonanywhere.com";

async function predict(symbol: string) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol,
      coin_id: "bitcoin",
      vs_currency: "usd",
      days: 30,
      data_source: "coingecko",
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}
```

## CORS

The Python backend now sends browser CORS headers. For early demos it allows all origins. For production, set:

```text
ALLOWED_ORIGINS=https://your-lovable-domain.lovable.app
```

in PythonAnywhere or your Python host.

## Deployment note

Lovable is a frontend builder. Keep Python model training, paper trading, broker keys, and private logic on PythonAnywhere or another Python host.
