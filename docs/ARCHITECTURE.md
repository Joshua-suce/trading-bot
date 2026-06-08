# Architecture

## Unified Trading Path

```text
trade mode
  -> endpoint and credential preflight
  -> Binance Futures client
  -> account snapshot and reconciliation
  -> sequential closed-candle scanning
  -> data-quality validation
  -> TA/ML signal aggregation
  -> portfolio and exposure controls
  -> precision/slippage/minimum-notional guards
  -> controlled same-direction market entry leg
  -> leg-owned stop-loss and take-profit orders
  -> audit, alerts, monitoring, and reconciliation
```

`BINANCE_API_URL` selects Binance Demo Trading or mainnet. There is no separate
paper execution implementation and no local simulated order path in production.

## Environment Selection

- `https://demo-fapi.binance.com`: Binance Futures Demo Trading
- `https://fapi.binance.com`: Binance Futures mainnet

The endpoint validator rejects HTTP, paths, query strings, and unknown hosts.
Mainnet additionally requires `ALLOW_MAINNET_TRADING=true` and, when configured,
a valid strategy approval.

## Modules

- `src/main.py`: CLI dispatch.
- `src/config.py`: endpoint, secrets, and validated runtime policy.
- `src/preflight.py`: environment and governance checks.
- `src/live/`: scheduler, candle processing, and data quality.
- `src/exchange/`: shared Binance REST/WebSocket implementation.
- `src/signals/` and `src/models/`: technical/ML decision generation.
- `src/risk/`: account, loss, sizing, and exposure controls.
- `src/execution/`: shared order and position lifecycle.
- `src/audit/`: durable events, controls, and trade state.
- `src/monitoring/`: alerts, logs, dashboard, and health.
- `src/soak/`: offline adapters that exercise the trading lifecycle safely.

## Failure Safety

- Missing protective orders trigger emergency flattening.
- Failed add-on protection rolls back only the new leg and preserves existing
  symbol protection.
- Reconciliation compares Binance's net symbol quantity with summed audit legs.
- Completed leg protection is finalized before aggregate quantity validation.
- Missing conditional history is recovered from private trade executions, so a
  partial net-position reduction can be attributed to the correct audited leg.
- Fatal shutdown submits one aggregate reduce-only exit per Binance symbol,
  then closes the corresponding internal legs from the confirmed fill.
- Failed emergency flattening activates emergency stop.
- Unmanaged exchange positions block reconciliation.
- Missing audited protection blocks reconciliation.
- Repeated account failures stop the trading loop.
- Secrets and URL signatures are redacted from logs.
