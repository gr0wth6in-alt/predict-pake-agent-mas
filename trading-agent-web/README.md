# Signal Deck Frontend

React + TypeScript dashboard for the Python autonomous trading agent API.

## Local setup

```bash
npm install
cp .env.example .env
npm run dev
```

Set `.env` to your PythonAnywhere API URL:

```text
VITE_AGENT_API_BASE_URL=https://YOUR_USERNAME.pythonanywhere.com
```

## Build

```bash
npm run build
```

Deploy output:

```text
dist
```

## API endpoints used

```text
GET  /model/status
POST /predict
POST /paper/run-once
POST /backtest
```
