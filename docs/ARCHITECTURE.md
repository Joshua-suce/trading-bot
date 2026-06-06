# Architecture

## Runtime Flow

```text
CLI
  -> preflight and governance
  -> exchange/account connection
  -> sequential symbol/timeframe scheduler
  -> OHLCV quality validation
  -> indicator computation
  -> TA and ML signal aggregation
  -> portfolio and exposure controls
  -> position sizing and ATR protection
  -> paper or live execution
  -> audit, logs, alerts, health, reconciliation
```

## Modules

- `src/main.py`: CLI and mode dispatch.
- `src/config.py`: validated environment configuration and secret-file loading.
- `src/preflight.py`: credential, environment, and live-governance checks.
- `src/live/`: runtime scheduler, closed-candle processing, and data quality.
- `src/signals/`: technical and ML signal fusion.
- `src/indicators/`: indicator calculations.
- `src/models/`: feature engineering, training, and model ensemble.
- `src/risk/`: portfolio limits, sizing, stop loss, and take profit.
- `src/execution/`: order guards and position lifecycle.
- `src/exchange/`: Binance REST and stream adapters.
- `src/audit/`: durable SQLite controls, events, and trades.
- `src/governance/`: live strategy approval validation.
- `src/monitoring/`: alerts, structured logs, dashboard, and health.
- `src/soak/`: bounded offline operational reliability simulation.

## Operating Modes

- `paper`: live data with simulated entries and exits.
- `live`: exchange orders with protection and reconciliation.
- `train`: model training.
- `dashboard`: monitoring UI.
- `admin`: operational controls.
- `health`: machine-readable service health.
- `soak`: offline reliability exercise.

There is no user-facing backtest mode.

## Safety Boundaries

- Closed candles only.
- Duplicate-candle suppression.
- Data-quality rejection before signal generation.
- Manual pause and emergency stop.
- Daily loss, drawdown, consecutive-loss, position-count, and exposure limits.
- Precision, minimum-notional, and slippage order guards.
- Protective-order verification.
- Emergency flatten after an unprotected entry.
- Startup and periodic live reconciliation.
- Secret redaction in logs and audit payloads.

## Persistence

SQLite stores:

- audit events,
- open and closed trades,
- operational controls,
- correlation identifiers,
- protective order identifiers.

The Docker deployment mounts `/app/data` for durable runtime state.
