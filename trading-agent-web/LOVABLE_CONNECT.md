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
- GET /market/ohlc?symbol=BTCUSDT&data_source=binance&days=30&interval=1h
- POST /predict with JSON body {"symbol":"BTCUSDT","days":30,"interval":"1h","data_source":"binance"}
- POST /paper/run-once with JSON body {"symbol":"BTCUSDT","days":30,"interval":"1h","data_source":"binance"}
- POST /backtest with JSON body {"symbol":"BTCUSDT","days":30,"interval":"1h","data_source":"binance"}

Show prediction confidence, direction_score, buy/sell/hold signal, risk approval, paper fill, and backtest return.
Do not store broker keys in the frontend.
Use Binance klines as the only market data source in the frontend.
Do not keep stale previous results after a failed fetch.
Clear prediction, paper, and backtest panels when a new request starts.
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
      symbol: "BTCUSDT",
      data_source: "binance",
      interval: "1h",
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

The Python backend now sends browser CORS headers. For early demos it allows all origins. For production, set:

```text
ALLOWED_ORIGINS=https://your-lovable-domain.lovable.app
```

in PythonAnywhere or your Python host.

## Deployment note

Lovable is a frontend builder. Keep Python model training, paper trading, broker keys, and private logic on PythonAnywhere or another Python host.
