# Website Architecture

If your website is TypeScript and the AI trading engine is Python, the clean architecture is to run them as separate services.

## Recommended shape

```text
TypeScript frontend
  -> TypeScript backend / API route
  -> Python trading-agent service
  -> broker/data APIs
  -> database/logs
```

## Why separate services

- TypeScript is good for UI, dashboard, authentication, forms, and charts.
- Python is better for model training, inference, backtesting, data science, and broker adapters.
- The trading agent should not run inside the browser.
- The frontend should never hold broker API keys.

## Hosting options

- Frontend only: Vercel, Netlify, Cloudflare Pages.
- TypeScript backend: Vercel serverless, Railway, Render, Fly.io, VPS.
- Python agent: Railway, Render, Fly.io, VPS, Docker container, or a private server.
- Database/logs: Postgres is a good first choice.

For early development, run both locally:

```text
localhost:3000  TypeScript website
localhost:8000  Python FastAPI trading-agent API
```

For production, host the Python agent separately from the website and communicate through authenticated HTTP or a queue. Keep live trading behind manual approval, risk limits, and a kill switch.
