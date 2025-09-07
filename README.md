# Cross-Sectional Momentum
## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp configs/settings.example.yaml configs/settings.yaml
```

Fill `ALPHAVANTAGE_API_KEY` in `.env`. Alpaca keys are optional.

## Run

```bash
python scripts/run_backtest.py
```

Notes:
- Alpha Vantage free keys are rate-limited; caching should be enabled in code.
- Equity returns should use adjusted close.
