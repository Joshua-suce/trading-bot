# AI Trading Bot — Binance Futures

A hybrid **technical analysis** and **machine learning** trading bot for **Binance Futures** (USD-margined perpetual contracts). Supports backtesting, paper trading, and live trading with comprehensive risk management.

## Features

- **30+ Technical Indicators** — Trend (EMA, SMA, MACD, ADX, PSAR, Ichimoku), Momentum (RSI, Stochastic, CCI, Williams %R, MFI, TSI), Volatility (Bollinger Bands, ATR, Keltner, Donchian), Volume (OBV, VWAP, Volume Profile)
- **Machine Learning** — XGBoost classifier (3-class: sell/hold/buy) + optional LSTM neural network, combined via weighted ensemble
- **Signal Fusion** — TA signals (40% weight) blended with ML predictions (60% weight) via `SignalAggregator`
- **Backtesting Engine** — Realistic simulation with configurable slippage (0.05%), commission (0.04%), ATR-based stop-loss/take-profit, trailing stops, and drawdown tracking
- **Risk Management** — Position sizing based on `RISK_PER_TRADE`, max drawdown guard (15%), daily loss limit (5%), max consecutive losses (3)
- **Live Trading** — WebSocket candle streams, automatic order placement, stop-loss/take-profit via reduce-only orders, trailing stop updates
- **Monitoring** — Streamlit dashboard (port 8501), Telegram + Discord alerts, structured logging with loguru
- **Paper Trading** — Simulated mode without real order execution
- **Docker Support** — Reproducible deployment via Docker and docker-compose

## Quick Start

```bash
# Clone and enter directory
cd trading-bot

# Create virtual environment
bash scripts/create_venv.sh
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Binance API keys (or leave empty for offline backtest)

# Run an offline backtest (no API keys needed)
python -m src.main --mode backtest --offline

# Run production readiness checks
python scripts/smoke_check.py --fast

# Scan repository text files for accidental secrets
python scripts/secret_scan.py
```

The smoke check enforces linting, formatting, type checks, secret scanning, health, offline soak, offline backtest, and a minimum test coverage threshold of 70%.

## Installation

### Prerequisites

- Python 3.11 or later (3.13+ recommended)
- Binance Futures account (for live/paper trading)
- Optional: TensorFlow 2.15+ for LSTM models (requires Python < 3.14)

### Virtual Environment

```bash
# Linux/macOS
bash scripts/create_venv.sh
source .venv/bin/activate

# Windows PowerShell
.\scripts\create_venv.ps1
.\.venv\Scripts\Activate.ps1
```

### Dependencies

```bash
pip install -r requirements.txt

# Optional: ML dependencies (LSTM)
pip install -r requirements-ml.txt
```

## Configuration

All configuration is done through environment variables in `.env` (copy from `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `BINANCE_API_KEY` | — | Binance API key |
| `BINANCE_API_SECRET` | — | Binance API secret |
| `BINANCE_API_KEY_FILE` | optional | Optional file path for Binance API key |
| `BINANCE_API_SECRET_FILE` | optional | Optional file path for Binance API secret |
| `BINANCE_TESTNET` | `true` | Use Binance testnet (recommended for testing) |
| `ALLOW_LIVE_TRADING` | `false` | Required opt-in before `live` mode can run against mainnet |
| `ALLOW_MAINNET_PAPER` | `false` | Required opt-in before `paper` mode can touch mainnet account APIs |
| `SYMBOLS` | `BTCUSDT,ETHUSDT` | Comma-separated symbols to trade |
| `TIMEFRAMES` | `5m,15m,30m,1h,4h,1d` | Comma-separated timeframes |
| `POSITION_SCOPE` | `symbol` | `symbol` allows one open trade per symbol; `symbol_timeframe` allows one per symbol/timeframe |
| `SCAN_SLEEP_SECONDS` | `5` | Scheduler idle sleep between due-scan checks |
| `RECONCILIATION_INTERVAL_SECONDS` | `300` | Live position/order reconciliation cadence |
| `MIN_OHLCV_CANDLES` | `50` | Minimum candles required before indicators/signals run |
| `MAX_CANDLE_DELAY_MULTIPLIER` | `3.0` | Reject candles older than timeframe times this multiplier |
| `ALLOW_ZERO_VOLUME_CANDLES` | `false` | Whether zero-volume candles are allowed |
| `MAX_ENTRY_SLIPPAGE_BPS` | `25` | Block entry market orders above this bid/ask slippage |
| `MIN_ORDER_NOTIONAL` | `5` | Minimum notional after exchange precision normalization |
| `MAX_LEVERAGE` | `3` | Maximum leverage (policy-capped at 20x) |
| `MAX_POSITION_SIZE` | `0.02` | Max fraction of equity per position |
| `MAX_OPEN_POSITIONS` | `3` | Maximum simultaneous audited open positions |
| `MAX_TOTAL_OPEN_NOTIONAL_PCT` | `0.20` | Maximum total open notional as fraction of equity |
| `MAX_SYMBOL_OPEN_NOTIONAL_PCT` | `0.10` | Maximum per-symbol open notional as fraction of equity |
| `DAILY_LOSS_LIMIT` | `0.05` | Max daily loss as fraction of equity |
| `MAX_DRAWDOWN` | `0.15` | Max drawdown before stopping |
| `RISK_PER_TRADE` | `0.02` | Fraction of equity risked per trade |
| `MODEL_UPDATE_INTERVAL_HOURS` | `24` | How often to retrain ML models |
| `FEATURE_LOOKBACK` | `100` | Lookback periods for feature engineering |
| `PREDICTION_HORIZON` | `6` | Prediction horizon in periods |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `LOG_DIR` | `data/logs` | Log file directory |
| `JSON_LOGS` | `true` | Write structured JSONL logs alongside text logs |
| `AUDIT_DB_PATH` | `data/audit/trading_audit.db` | SQLite audit, controls, and open-trade state database |
| `TRADING_ENABLED` | `true` | Environment-level trading enable/disable switch |
| `REQUIRE_STRATEGY_APPROVAL` | `true` | Require an unexpired strategy approval artifact before live mode |
| `STRATEGY_APPROVAL_PATH` | `data/governance/strategy_approval.json` | Strategy approval artifact path |
| `STRATEGY_APPROVAL_MAX_AGE_HOURS` | `168` | Approval expiry window |
| `MIN_APPROVAL_TRADES` | `20` | Minimum approved backtest trades |
| `MIN_APPROVAL_PROFIT_FACTOR` | `1.10` | Minimum approved profit factor |
| `MAX_APPROVAL_DRAWDOWN_PCT` | `20.0` | Maximum approved drawdown percentage |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token for alerts (optional) |
| `TELEGRAM_CHAT_ID` | — | Telegram chat ID (optional) |
| `DISCORD_WEBHOOK_URL` | — | Discord webhook URL (optional) |
| `MODEL_DIR` | `data/models` | Where trained models are stored |

## Usage

For deployment and incident procedures, see [Production Runbook](docs/PRODUCTION_RUNBOOK.md).

### Modes

The bot supports seven operating modes via `--mode`:

| Mode | Description |
|---|---|
| `backtest` | Run historical simulation with configurable parameters |
| `paper` | Simulated trading without real order execution |
| `live` | Real trading on Binance Futures (requires valid API keys) |
| `train` | Train ML models on historical data |
| `dashboard` | Launch Streamlit monitoring UI |
| `admin` | Inspect and update trading controls |
| `health` | Emit JSON health status for containers and automation |

### Backtesting

```bash
# Fetch live data from Binance and backtest
python -m src.main --mode backtest --symbol BTCUSDT --timeframe 1h --limit 500 --initial-capital 10000

# Offline backtest with generated mock data (no API keys needed)
python -m src.main --mode backtest --offline

# With verbose debug logging
python -m src.main --mode backtest --offline --verbose
```

### Paper Trading

```bash
python -m src.main --mode paper
```

### Live Trading

```bash
python -m src.main --mode live
```

> **Warning**: Live trading places real orders. Ensure your API keys have strict IP whitelisting and no withdrawal permissions.

### Admin Controls

```bash
# Show durable controls and audited open trades
python -m src.main --mode admin --admin-action status

# Create an approval artifact from a backtest result
python -m src.main --mode admin --admin-action approve-strategy --offline --symbol BTCUSDT --timeframe 1h --limit 1000 --reason "demo approval"

# Inspect strategy approval status
python -m src.main --mode admin --admin-action strategy-status

# Revoke strategy approval
python -m src.main --mode admin --admin-action revoke-strategy --reason "strategy review required"

# Pause new entries without clearing existing state
python -m src.main --mode admin --admin-action pause --reason "operator pause"

# Resume after a manual pause
python -m src.main --mode admin --admin-action resume --reason "checks passed"

# Force all new entries to stop until explicitly cleared
python -m src.main --mode admin --admin-action emergency-stop --reason "manual safety stop"

# Clear emergency stop after exchange state and protective orders are verified
python -m src.main --mode admin --admin-action clear-emergency --reason "reconciled"
```

Live mode restores audited open trades on startup, reconciles them with Binance positions, verifies protective stop/target orders, and activates the emergency stop if unmanaged exposure or missing protection is detected.

### Training ML Models

```bash
python -m src.main --mode train --symbol BTCUSDT --timeframe 1h --limit 1000
```

### Dashboard

```bash
python -m src.main --mode dashboard
# Open http://localhost:8501
```

### Health Check

```bash
python -m src.main --mode health
```

The health check returns JSON and exits non-zero only for critical service failures such as an unavailable audit database. Emergency stop and manual pause report as degraded trading state.

### Offline Soak Test

```bash
python -m src.main --mode soak --soak-iterations 2 --soak-report data/reports/soak_report.json
```

The soak mode runs a bounded offline paper-trading path with fake exchange data and writes a JSON report. It verifies startup alert flow, trade open/close audit events, health-compatible audit state, and clean shutdown with no open paper trades.

### CLI Arguments

| Argument | Default | Choices |
|---|---|---|
| `--mode` | `backtest` | backtest, paper, live, train, dashboard, admin, health, soak |
| `--symbol` | `BTCUSDT` | Any Binance Futures symbol |
| `--timeframe` | `1h` | 1m, 5m, 15m, 1h, 4h, 1d, etc. |
| `--limit` | `500` | Number of candles to fetch |
| `--initial-capital` | `10000` | Starting capital for backtest |
| `--offline` | (flag) | Use generated mock data |
| `--verbose` | (flag) | Enable debug logging |
| `--admin-action` | `status` | status, pause, resume, emergency-stop, clear-emergency |
| `--reason` | empty | Reason recorded for admin control changes |
| `--soak-iterations` | `2` | Number of offline soak scan iterations |
| `--soak-report` | `data/reports/soak_report.json` | Soak report JSON output path |

## Trading Strategy

### Signal Generation

The bot uses a **dual-signal fusion** approach:

1. **Technical Analysis (40% weight)** — Rule-based signals from:
   - EMA 9/21 crossovers
   - RSI oversold/overbought (thresholds at 30/70)
   - MACD histogram zero-line crossovers
   - Bollinger Band touches (%B)
   - ADX trend strength filter (weak trend attenuates signals)
   - Volume ratio confirmation

2. **Machine Learning (60% weight)** — Ensemble prediction:
   - XGBoost 3-class classifier (-1 sell, 0 hold, +1 buy)
   - Optional LSTM sequence model
   - Weighted ensemble (XGBoost 60%, LSTM 40%)

### Risk Management

- **Position Sizing** — Risk-based: `position_size = (equity * risk_per_trade) / stop_loss_distance`
- **Stop Loss** — ATR-based at 1.5x ATR from entry
- **Take Profit** — 2:1 risk-reward ratio by default
- **Trailing Stop** — Activates after price moves 1x ATR in profit
- **Portfolio Guards** — Max drawdown (15%), daily loss limit (5%), max consecutive losses (3)

### Supported Indicators

| Category | Indicators |
|---|---|
| **Trend** | EMA (9, 21, 50, 200), SMA (20, 50), MACD (12/26/9), ADX (14) |
| **Momentum** | RSI (14), Stochastic (14/3/3), CCI (20), Williams %R (14), MFI (14), TSI |
| **Volatility** | Bollinger Bands (20/2), ATR (14), Keltner Channels, Historical Volatility (21), Donchian Channels (20) |
| **Volume** | OBV, VWAP, Volume SMA (20), Volume Ratio |

## Architecture

```
src/
├── main.py                  # CLI entry point, argument parsing, mode routing
├── config.py                # Pydantic settings from .env
├── backtest/                # Backtesting engine, metrics, trade types
│   ├── engine.py            # Core backtest loop with slippage & commission
│   ├── metrics.py           # Sharpe, Sortino, Calmar, drawdown, etc.
│   └── types.py             # BacktestTrade dataclass
├── exchange/                # Binance Futures connectivity
│   ├── client.py            # ccxt async REST + WebSocket client
│   ├── account.py           # Account info, position queries
│   └── stream.py            # Real-time candle stream via WebSocket
├── execution/               # Order and position management
│   ├── order_manager.py     # Market, limit, stop-loss, take-profit orders
│   └── position_manager.py  # Enter/exit long/short positions with SL/TP
├── indicators/              # Technical analysis (30+ indicators)
│   ├── compute.py           # Aggregator: computes all indicators at once
│   ├── trend.py             # EMA, SMA, MACD, ADX, PSAR, Ichimoku
│   ├── momentum.py          # RSI, Stochastic, CCI, Williams %R, MFI, TSI
│   ├── volatility.py        # Bollinger Bands, ATR, Keltner, HV, Donchian
│   └── volume.py            # OBV, VWAP, Volume Profile, EOM, Volume SMA
├── live/                    # Live/paper trading loop
│   └── loop.py              # Main trading orchestration
├── models/                  # Machine learning
│   ├── feature_engineer.py  # Feature engineering from indicators
│   ├── classifier.py        # XGBoost classifier
│   ├── lstm_model.py        # LSTM neural network (optional)
│   ├── ensemble.py          # Weighted ensemble
│   └── trainer.py           # Training pipeline
├── monitoring/              # Logging, dashboard, alerts
│   ├── logger.py            # Loguru configuration (console + file)
│   ├── dashboard.py         # Streamlit dashboard
│   └── alerter.py           # Telegram & Discord alerts
├── risk/                    # Risk management
│   ├── portfolio.py         # Portfolio tracking, daily PnL, drawdown
│   ├── position_sizer.py    # Position sizing based on risk
│   └── stop_loss.py         # SL, TP, trailing stop calculation
├── signals/                 # Signal generation & fusion
│   ├── ta_signal.py         # TA rule-based signals
│   └── aggregator.py        # Fuses TA + ML into FinalSignal
└── utils/                   # Utilities
    └── helpers.py           # Rounding, time conversions, mock OHLCV
```

### Data Flow

```
main.py ──► config.py (.env)
  │
  ├── backtest ──► exchange/client.py (fetch OHLCV)
  │                    │
  │                    ▼
  │              indicators/compute.py (30+ indicators)
  │                    │
  │                    ▼
  │              signals/ta_signal.py (TA rules)
  │                    │
  │                    ▼
  │              backtest/engine.py (simulate trades)
  │                    │
  │                    ▼
  │              backtest/metrics.py (performance)
  │
  ├── train ───► exchange/client.py (fetch OHLCV)
  │                    │
  │                    ▼
  │              models/feature_engineer.py
  │              models/classifier.py (XGBoost)
  │              models/lstm_model.py (LSTM)
  │              models/ensemble.py
  │
  ├── paper/live ──► live/loop.py
  │                    ├── exchange/stream.py (WebSocket)
  │                    ├── signals/aggregator.py (TA + ML)
  │                    ├── risk/position_sizer.py
  │                    ├── risk/stop_loss.py
  │                    ├── risk/portfolio.py
  │                    ├── execution/order_manager.py
  │                    └── execution/position_manager.py
  │
  └── dashboard ──► monitoring/dashboard.py (Streamlit)
                    monitoring/alerter.py (Telegram/Discord)
```

## Docker

```bash
# Build
docker build -t trading-bot:latest .

# Run backtest
docker run --env-file .env -v $(pwd)/data:/app/data trading-bot:latest \
  python -m src.main --mode backtest --offline

# Run with docker-compose (dashboard at http://localhost:8501)
docker-compose up --build
```

## Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src tests/

# Specific test file
python -m pytest tests/test_backtest.py -v
```

### Test Files

| File | Tests |
|---|---|
| `tests/test_main.py` | CLI argument parsing |
| `tests/test_utils.py` | Mock OHLCV generation |
| `tests/test_indicators.py` | All indicator computations |
| `tests/test_signals.py` | TA signal generation & aggregation |
| `tests/test_risk.py` | Stop-loss and trailing stop logic |
| `tests/test_models.py` | Feature engineering |
| `tests/test_backtest.py` | Backtest engine integration |

## Development

```bash
# Format code
python -m black .
python -m isort src tests

# Lint
python -m ruff check src tests

# Type check
python -m mypy src
```

## Project Structure

```
config/              # Configuration files (reserved)
data/
├── logs/            # Trading and error logs (loguru)
└── models/          # Saved ML models (joblib/h5)
docs/                # Documentation
notebooks/           # Jupyter notebooks (reserved)
scripts/
├── create_venv.sh   # Linux/macOS venv setup
└── create_venv.ps1  # Windows venv setup
src/                 # Application package
tests/               # pytest test suite
```

## Security

- API keys are read from `.env` only — never committed to version control
- Use Binance testnet for development (`BINANCE_TESTNET=true`)
- Restrict API keys to specific IP addresses
- Use API keys with **No Withdrawal** permission only
- The `.gitignore` excludes `.env`, `data/`, and credential files

## License

MIT
