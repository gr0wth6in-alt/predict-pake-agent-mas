# Frontend Deploy

The frontend lives in:

```text
trading-agent-web/
```

It is separate from the Python agent backend.

## Lovable

Use the files in `trading-agent-web/` as the frontend source. The most important file is:

```text
trading-agent-web/LOVABLE_CONNECT.md
```

It contains the prompt and API details for connecting Lovable to the PythonAnywhere backend.

## Netlify

If you deploy this frontend to Netlify:

```text
Base directory: trading-agent-web
Build command: npm run build
Publish directory: trading-agent-web/dist
```

If Netlify asks for environment variables:

```text
VITE_AGENT_API_BASE_URL=https://YOUR_USERNAME.pythonanywhere.com
```

Replace `YOUR_USERNAME` with your PythonAnywhere username.
This frontend is configured for CoinGecko data because Binance rejects PythonAnywhere with HTTP 451.
