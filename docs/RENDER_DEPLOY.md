# Deploy to Render

This project is ready for Render as a Python Web Service.

## Option 1: Blueprint

1. Push this repository to GitHub.
2. In Render, choose **New +** then **Blueprint**.
3. Select the GitHub repository.
4. Render will read `render.yaml`.
5. Deploy the service.

## Option 2: Manual Web Service

Use these values:

```text
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn trading_agent.api:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
PYTHONPATH=src
TRADING_MODE=paper
SYMBOL=BTCUSD
DEFAULT_MODEL_PATH=models/btcusd_auto_nb.json
DEFAULT_CSV_PATH=examples/mixed_training_prices.csv
```

## Test after deploy

Open these URLs after Render gives you a domain:

```text
https://your-service.onrender.com/health
https://your-service.onrender.com/docs
```

Then test prediction with:

```bash
curl -X POST https://your-service.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d "{\"symbol\":\"BTCUSD\"}"
```

## Notes

The API is paper-trading only. Do not add live broker keys until you have authentication, encrypted secrets, monitoring, and risk limits.
