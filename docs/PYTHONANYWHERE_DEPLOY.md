# Deploy to PythonAnywhere Free

Use this path if Render asks for a card and you do not have one. PythonAnywhere Free can run a small Python web app without a credit card, but it has tight limits.

## Important limits

- Free accounts have limited CPU and disk.
- New free accounts support one web app.
- Outbound internet is restricted to allowlisted sites, so broker/data APIs may fail unless the domain is allowlisted.
- This is good for demo endpoints, not live autonomous trading.

## Steps

1. Create a free account at PythonAnywhere.
2. Open **Consoles** then start a Bash console.
3. Clone the repo:

```bash
git clone https://github.com/gr0wth6in-alt/predict-pake-agent-mas.git
cd predict-pake-agent-mas
```

4. Install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

5. Go to **Web** then **Add a new web app**.
6. Choose **Manual configuration** and Python 3.12.
7. Set the virtualenv path to:

```text
/home/YOUR_USERNAME/predict-pake-agent-mas/.venv
```

8. Edit the WSGI configuration file and replace its content with:

```python
import os
import sys

project_home = "/home/YOUR_USERNAME/predict-pake-agent-mas"
if project_home not in sys.path:
    sys.path.insert(0, project_home)

src_path = os.path.join(project_home, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

os.chdir(project_home)
os.environ.setdefault("PYTHONPATH", "src")
os.environ.setdefault("TRADING_MODE", "paper")
os.environ.setdefault("SYMBOL", "BTCUSD")
os.environ.setdefault("DEFAULT_MODEL_PATH", "models/btcusd_demo_nb.json")
os.environ.setdefault("DEFAULT_CSV_PATH", "examples/mixed_training_prices.csv")
os.environ.setdefault("ALLOWED_ORIGINS", "*")

from trading_agent.wsgi_app import app as application
```

Replace `YOUR_USERNAME` with your PythonAnywhere username.

9. Reload the web app.
10. Test:

```text
https://YOUR_USERNAME.pythonanywhere.com/health
https://YOUR_USERNAME.pythonanywhere.com/model/status
```

## API endpoints

```text
GET  /health
GET  /model/status
GET  /market/coins
GET  /market/ohlc
POST /predict
POST /paper/run-once
POST /backtest
```

Example body for `/predict`:

```json
{
  "symbol": "BTCUSD",
  "coin_id": "bitcoin",
  "vs_currency": "usd",
  "days": 30,
  "data_source": "coingecko"
}
```
