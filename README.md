# Binance Futures Trading Bot

Enterprise-oriented Binance Futures trading bot with one unified exchange
execution path, multi-timeframe technical/ML signals, risk controls, auditing,
health checks, alerts, and Docker support.

## Modes

| Mode | Purpose |
|---|---|
| `trade` | Execute orders on the Binance environment selected by `BINANCE_API_URL` |
| `train` | Train the machine-learning ensemble from exchange OHLCV data |
| `dashboard` | Open the Streamlit monitoring dashboard |
| `admin` | Inspect or change operational controls |
| `health` | Report audit, control, event, and open-position health |
| `soak` | Run a bounded offline reliability simulation through fake adapters |

## One Trading Setup

Demo and mainnet use the same command and execution code:

```powershell
.\.venv\Scripts\python.exe -m src.main --mode trade
```

For Binance Demo Trading:

```dotenv
BINANCE_API_KEY=your_demo_key
BINANCE_API_SECRET=your_demo_secret
BINANCE_API_URL=https://demo-fapi.binance.com
```

For Binance mainnet:

```dotenv
BINANCE_API_KEY=your_mainnet_key
BINANCE_API_SECRET=your_mainnet_secret
BINANCE_API_URL=https://fapi.binance.com
ALLOW_MAINNET_TRADING=true
```

`ALLOW_MAINNET_TRADING` is a safety authorization, not a second execution mode.
Mainnet may also require a valid strategy approval artifact. Demo and mainnet
otherwise use the same order, protection, reconciliation, audit, and alert path.

Only the two official HTTPS hosts above are accepted. This prevents credentials
from being sent to an unknown endpoint.

## Runtime Flow

1. Validate endpoint, credentials, controls, and mainnet governance.
2. Connect to Binance Futures.
3. Restore and reconcile audited exchange positions.
4. Scan configured symbols and timeframes sequentially.
5. Validate closed OHLCV candles.
6. Fuse technical and trained ML signals.
7. Apply risk, loss, drawdown, position-count, and exposure controls.
8. Submit a market entry.
9. Submit reduce-only stop-loss and take-profit protection.
10. Audit and alert every lifecycle event.

There is no locally simulated paper-trading branch. Demo trading uses Binance
Demo Trading orders.

## Commands

```powershell
# Unified trading
.\scripts\run_bot.cmd --mode trade

# Train models
.\scripts\run_bot.cmd --mode train --symbol BTCUSDT --timeframe 1h --limit 1000

# Health and monitoring
.\scripts\run_bot.cmd --mode health
.\scripts\run_bot.cmd --mode dashboard

# Offline operational soak
.\scripts\run_bot.cmd --mode soak --soak-iterations 5

# Controls
.\scripts\run_bot.cmd --mode admin --admin-action status
.\scripts\run_bot.cmd --mode admin --admin-action pause --reason "maintenance"
.\scripts\run_bot.cmd --mode admin --admin-action resume
.\scripts\run_bot.cmd --mode admin --admin-action emergency-stop --reason "risk event"
```

The launcher always uses the project virtual environment. When invoking Python
directly, use `.\.venv\Scripts\python.exe`, not the system `python`.

## Quality Gate

```powershell
.\scripts\smoke_check.ps1
```

The gate runs formatting, imports, lint, secret scanning, mypy, pytest with a
75% coverage floor, an isolated health check, and an offline soak.

## Docker

```powershell
docker build -t trading-bot:enterprise .
docker run --rm --env-file .env -v ${PWD}\data:/app/data trading-bot:enterprise
```

The existing image must be rebuilt whenever source or environment contracts
change.

Trading involves financial risk. Validate with Binance Demo Trading before
authorizing mainnet.
