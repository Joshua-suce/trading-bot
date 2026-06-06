# Binance Futures Trading Bot

Enterprise-oriented Binance Futures trading bot with paper and live execution,
multi-timeframe technical/ML signals, risk controls, auditing, health checks,
alerts, and Docker support.

## Supported Modes

| Mode | Purpose |
|---|---|
| `paper` | Evaluate live market data and simulate execution without exchange orders |
| `live` | Submit protected Binance Futures orders after all safety checks pass |
| `train` | Train the machine-learning ensemble from exchange OHLCV data |
| `dashboard` | Open the Streamlit monitoring dashboard |
| `admin` | Inspect or change operational controls |
| `health` | Report audit, control, event, and open-position health |
| `soak` | Run a bounded offline reliability simulation |

The historical `backtest` command mode has been removed. Historical-analysis
modules may remain as internal validation utilities, but they are not exposed as
a bot operating mode.

## Quick Start

```powershell
cd C:\trading-bot
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m src.main --mode health
.\.venv\Scripts\python.exe -m src.main --mode paper
```

Paper mode is the default:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

Paper and live modes require Binance API credentials. Keep
`BINANCE_TESTNET=true` during validation.

## Common Commands

```powershell
# Paper trading
.\.venv\Scripts\python.exe -m src.main --mode paper

# Model training
.\.venv\Scripts\python.exe -m src.main --mode train --symbol BTCUSDT --timeframe 1h --limit 1000

# Dashboard
.\.venv\Scripts\python.exe -m src.main --mode dashboard

# Health
.\.venv\Scripts\python.exe -m src.main --mode health

# Reliability soak
.\.venv\Scripts\python.exe -m src.main --mode soak --soak-iterations 5

# Administrative status
.\.venv\Scripts\python.exe -m src.main --mode admin --admin-action status
```

## Safety Controls

```powershell
.\.venv\Scripts\python.exe -m src.main --mode admin --admin-action pause --reason "maintenance"
.\.venv\Scripts\python.exe -m src.main --mode admin --admin-action resume --reason "checks passed"
.\.venv\Scripts\python.exe -m src.main --mode admin --admin-action emergency-stop --reason "risk event"
.\.venv\Scripts\python.exe -m src.main --mode admin --admin-action clear-emergency --reason "exchange reconciled"
.\.venv\Scripts\python.exe -m src.main --mode admin --admin-action revoke-strategy --reason "approval withdrawn"
```

Live mode remains blocked when strategy approval is required but no valid
approval artifact is deployed.

## Trading Logic

1. Scan configured symbols and timeframes sequentially.
2. Validate OHLCV freshness, continuity, volume, and closed-candle state.
3. Calculate technical indicators.
4. Fuse technical and trained ML signals.
5. Reject weak or neutral decisions.
6. Apply pause, emergency, drawdown, loss, position-count, and exposure limits.
7. Size the position from account equity and stop distance.
8. Submit the entry and protective stop/target orders.
9. Audit and alert every lifecycle event.
10. Reconcile live exchange positions and protective orders periodically.

## Configuration

Important `.env` settings include:

```dotenv
BINANCE_TESTNET=true
ALLOW_LIVE_TRADING=false
ALLOW_MAINNET_PAPER=false
TRADING_ENABLED=true

SYMBOLS=BTCUSDT,ETHUSDT
TIMEFRAMES=5m,15m,30m
POSITION_SCOPE=symbol

RISK_PER_TRADE=0.02
MAX_LEVERAGE=3
MAX_POSITION_SIZE=0.02
MAX_OPEN_POSITIONS=3
MAX_TOTAL_OPEN_NOTIONAL_PCT=0.20
MAX_SYMBOL_OPEN_NOTIONAL_PCT=0.10
DAILY_LOSS_LIMIT=0.05
MAX_DRAWDOWN=0.15
```

Use the `*_FILE` secret settings in managed environments when possible.
Never commit `.env`.

## Quality Gate

```powershell
.\scripts\smoke_check.ps1
```

The gate runs formatting, import checks, linting, secret scanning, mypy,
pytest with a 75% coverage floor, health checks, and an offline soak.

## Docker

```powershell
docker build -t trading-bot:enterprise .
docker run --rm --env-file .env -v ${PWD}\data:/app/data trading-bot:enterprise
```

Or:

```powershell
docker compose up -d
docker compose logs -f trading-bot
docker compose down
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Production runbook](docs/PRODUCTION_RUNBOOK.md)
- [Strategy](docs/STRATEGY.md)

Trading involves financial risk. Paper-test, monitor, and approve the strategy
before enabling live mainnet execution.
