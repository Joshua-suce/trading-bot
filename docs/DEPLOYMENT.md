# Deployment

## Environment

The bot has one exchange execution mode:

```bash
python -m src.main --mode trade
```

Select Binance Demo Trading:

```dotenv
BINANCE_API_KEY=demo_key
BINANCE_API_SECRET=demo_secret
BINANCE_API_URL=https://demo-fapi.binance.com
ALLOW_MAINNET_TRADING=false
```

Select Binance mainnet:

```dotenv
BINANCE_API_KEY=mainnet_key
BINANCE_API_SECRET=mainnet_secret
BINANCE_API_URL=https://fapi.binance.com
ALLOW_MAINNET_TRADING=true
```

Use API keys with Futures and read permissions, IP restrictions, and no
withdrawal permission.

## Docker

```bash
docker build -t trading-bot:enterprise .
docker compose up -d
docker compose ps
docker compose logs -f trading-bot
```

Stop:

```bash
docker compose down
```

The `data` volume stores audit, logs, reports, and model artifacts.

## Bare Metal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --mode health
python -m src.main --mode trade
```

## Mainnet Checklist

- `BINANCE_API_URL=https://fapi.binance.com`
- `ALLOW_MAINNET_TRADING=true`
- Valid mainnet API credentials
- API key IP restrictions enabled
- No withdrawal permission
- Strategy approval valid when required
- Telegram or Discord alerts verified
- Audit database mounted on durable storage
- Health status checked
- Exchange positions and open orders reconciled
- Risk and exposure limits reviewed

Run on Binance Demo Trading first. Demo uses the same production execution path.
