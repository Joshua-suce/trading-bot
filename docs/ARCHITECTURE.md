# Architecture

## Overview

The trading bot follows a **modular pipeline architecture**. Each component is independently testable and swappable. The pipeline processes market data through stages: fetch → indicators → signals → risk → execution.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py (CLI Entry Point)                   │
│              argparse ──► mode dispatch ──► config.py              │
└──────┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
  backtest/    train/    paper/    live/    dashboard/
    engine    trainer    loop      loop     streamlit
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   indicators/compute.py                      │
  │   trend.py │ momentum.py │ volatility.py │ volume.py         │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                     signals/                                  │
  │   ta_signal.py (rule-based) │ aggregator.py (TA + ML fuse)   │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                     risk/                                     │
  │   portfolio.py │ position_sizer.py │ stop_loss.py            │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                     execution/                                │
  │   order_manager.py │ position_manager.py                     │
  └──────────────────────────────────────────────────────────────┘
```

## Core Modules

### `src/config.py` — Configuration
- Uses `pydantic-settings` to parse `.env` into a typed `Settings` object
- Singleton `settings` instance imported by all modules
- Properties (`symbols_list`, `timeframes_list`, `exchange_config`) provide derived values

### `src/exchange/client.py` — Exchange Client
- Async REST client via `ccxt.async_support.binanceusdm`
- WebSocket client via `ccxt.pro.binanceusdm`
- Testnet support via URL override when `BINANCE_TESTNET=true`
- Methods: `fetch_ohlcv()`, `fetch_balance()`, `create_order()`, `cancel_order()`, `set_leverage()`, `watch_ohlcv()`
- Context manager (`async with ExchangeClient()`) for automatic connection lifecycle

### `src/indicators/` — Technical Analysis
- Four indicator categories, each in its own file with a class
- `compute.py` provides `compute_all_indicators(df)` — the single entry point that runs all indicators on a DataFrame
- Indicator modules are stateless singletons (instantiated at module level)

### `src/signals/` — Signal Generation
- **`ta_signal.py`**: `TechnicalSignal.generate(df)` applies rule-based logic:
  - EMA crossovers, RSI levels, MACD cross, Bollinger Band touches
  - ADX trend strength filter attenuates signals in weak trends
  - Volume ratio boosts confidence
  - Weighted voting across all triggered sub-signals
- **`aggregator.py`**: `SignalAggregator.generate(df)` computes both TA and ML signals, then fuses them:
  - TA weight: 40%, ML weight: 60% (configurable)
  - Falls back to TA only if ML unavailable

### `src/models/` — Machine Learning
- **`feature_engineer.py`**: Creates 50+ features from raw indicators (lags, rolling stats, crossovers, targets)
- **`classifier.py`**: XGBoost 3-class classifier (-1, 0, +1) with `predict_with_confidence()`
- **`lstm_model.py`**: Optional LSTM sequence model (requires TensorFlow); uses 60-period sequences
- **`ensemble.py`**: `ModelEnsemble` combines XGBoost (60%) + LSTM (40%) weighted voting
- **`trainer.py`**: Training pipeline — indicator calc → feature engineering → train XGBoost → build ensemble

### `src/backtest/` — Backtesting
- **`engine.py`**: `BacktestEngine.run(df, signal_fn)` iterates candles:
  - For each candle, calls `signal_fn` on available data (no lookahead bias)
  - Tracks open positions with ATR-based SL/TP
  - Detects SL/TP hits on high/low of next candle
  - Trailing stop updates on each candle
  - Closes positions at end of data
  - Realistic: 0.04% commission, 0.05% slippage
- **`metrics.py`**: Calculates Total Return, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor, Avg Hold Time

### `src/live/loop.py` — Live Trading Loop
- Connects to exchange, sets leverage per symbol
- Starts WebSocket candle streams for all symbol/timeframe pairs
- On each candle: checks for existing position → generates fused signal → enters if signal confident
- Paper mode logs trades without execution; live mode places real orders
- Periodic account updates (every 60s) for portfolio tracking

### `src/risk/` — Risk Management
- **`portfolio.py`**: Tracks account info, daily PnL, drawdown, consecutive losses; enforces trade blocks
- **`position_sizer.py`**: Calculates quantity from `(equity * risk_per_trade) / stop_loss_distance`, constrained by max position size and leverage
- **`stop_loss.py`**: Calculates SL (1.5x ATR), TP (2:1 RR), trailing stop activation (1x ATR profit)

### `src/execution/` — Order Execution
- **`order_manager.py`**: Wraps exchange order creation; handles market, limit, stop-market, stop-limit, take-profit orders with reduce-only flag
- **`position_manager.py`**: Full position lifecycle — checks portfolio limits, sizes position, places entry + SL + TP orders, manages trailing stops, exits and cleans up

### `src/monitoring/` — Monitoring
- **`logger.py`**: Loguru configuration — colored console, daily rotating files (30 day retention), separate error log (90 day retention)
- **`dashboard.py`**: Streamlit app with equity curve, drawdown chart, positions table, trade history (currently placeholder data)
- **`alerter.py`**: Async Telegram (via bot API) and Discord (via webhook) alerts for trades, errors, daily summaries

## Data Flow Details

### Backtest Flow
```
1. main.py parses CLI args
2. main.py: fetch OHLCV from exchange (or generate mock)
3. main.py: create TechnicalSignal instance
4. main.py: wrap signal generation in signal_fn
5. BacktestEngine.run():
   a. compute_all_indicators(df) — adds all TA columns
   b. for each candle i (starting at index 100):
      - if not in position: call signal_fn(current[:i]) → get direction + confidence
      - if signal direction != 0 and confidence > 0.3: enter position
      - if in position: check next candle's high/low for SL/TP hits
      - if no SL/TP hit: check for signal reversal
      - update trailing stop
   c. close any open position at end
   d. calculate metrics
```

### Live Trade Flow
```
1. LiveTradingLoop.start():
   a. ExchangeClient.connect()
   b. get_account_info() → update PortfolioManager
   c. for each symbol: set_leverage()
   d. for each (symbol, timeframe): start CandleStream
2. On each candle (via _on_callback):
   a. if mode == "paper": log candle (no execution)
   b. if mode == "live":
      - check if already in position → if yes, manage trailing stop
      - fetch recent OHLCV → compute indicators
      - SignalAggregator.generate(df) → FinalSignal
      - if direction != 0 and confidence >= 0.4:
        - PositionManager.enter_long/enter_short()
          - PortfolioManager.can_trade() → checks drawdown, daily loss, consecutive losses
          - StopLossManager.calculate() → SL/TP levels
          - PositionSizer.calculate() → quantity
          - OrderManager.market_order() → entry
          - OrderManager.stop_loss_order() → SL
          - OrderManager.limit_order() → TP
```

## Key Design Decisions

1. **src/ layout** — Package under `src/` prevents import confusion and enforces explicit relative imports
2. **Singleton settings** — `config.py` instantiates `Settings()` at module level, imported everywhere
3. **Stateless indicators** — Indicator classes hold no state; all data is passed as DataFrames
4. **Async throughout** — Exchange calls, streams, and alerts are async; backtest is synchronous
5. **Position-based backtesting** — Tracks individual trades rather than per-candle rebalancing, matching live behavior
6. **ccxt pro for WebSocket** — Uses `ccxt.pro` for real-time streaming with automatic reconnection
