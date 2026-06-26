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
  -> strategy contract (scope, confidence, cost, risk, horizon)
  -> portfolio and exposure controls
  -> precision/slippage/minimum-notional guards
  -> controlled same-direction market entry leg
  -> leg-owned stop-loss and take-profit orders
  -> audit, alerts, monitoring, and reconciliation
```

`BINANCE_API_URL` selects Binance Demo Trading or mainnet. There is no separate
paper execution implementation and no local simulated order path in production.

## Strategy Boundary

`src/strategies/policy.py` is the shared contract between signal evaluation,
position sizing, stop/target construction, live trade management, and
walk-forward governance. Strategy-specific values are resolved through
`StrategyRegistry`; downstream modules do not infer behavior from timeframe or
reuse scalp settings for swing trades.

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
- `src/backtest/`: deterministic historical execution, after-cost metrics, and
  symbol/timeframe/strategy baseline evidence. Historical results are not a
  live-strategy approval until replay uses the complete live decision path.
- `src/live/`: scheduler, candle processing, and data quality.
- `src/exchange/`: shared Binance REST/WebSocket implementation.
- `src/signals/` and `src/models/`: technical/ML decision generation.
- `src/risk/`: account, loss, sizing, and exposure controls.
- `src/execution/`: shared order and position lifecycle.
- `src/audit/`: durable events, controls, trade state, and signal observations.
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
- Ambiguous market exits are confirmed from private executions by exact order
  ID before the bot declares the exit failed.
- Disappeared protection receives bounded settlement retries so Binance's
  position, order, and execution views can converge before an emergency is
  declared.
- Fatal shutdown verifies the real exchange quantity before submitting a
  reduce-only aggregate close. Quantity mismatches are flattened but left
  reconciliation-pending in the audit store rather than falsely finalized.
- Timeframe scans are aligned to UTC exchange candle boundaries with a short
  close grace period, while duplicate-candle suppression preserves idempotency.
- ML health is tracked per aggregator with 3-consecutive-failure degradation
  and instant recovery alerts via Telegram. `FORCE_TA_ONLY` skips ML entirely.
- Strategy entries pass a volatility, trend, momentum, volume, extension, and
  higher-timeframe quality gate before execution. Intraday entries require the
  cached 1h trend regime to agree with the requested direction.
- Technical votes are capped by indicator family to prevent correlated signal
  stacking. Regime routing applies price-action confirmation: trend pullbacks
  around EMA 50/Fibonacci, rejection candles for range/reversal entries, and
  ATR-buffered strong-close Donchian breakouts. TA/ML fusion weights vary by
  regime while active opposing ML still blocks non-trend setups. Mature regime
  entries require at least two independent technical signal families.
- The 1-minute and 3-minute scopes use an isolated scalp route. It does not
  reuse slower swing fallbacks, carries a smaller risk budget and tighter
  ATR-based exits, and checks live bid/ask spread plus fee-adjusted target edge
  before order submission. WebSocket close events feed a bounded rolling cache;
  stale streams automatically restore REST scanning. Scalp signals have a
  close-to-decision latency budget, and open positions receive independent
  break-even, favorable-peak trailing, partial-profit, and maximum-hold
  management. Repeated stream failures open a per-scope reconnect circuit while
  prioritized REST scanning continues with its own bounded latency allowance.
  Missing Demo ticker bid/ask values fall back to public order-book depth before
  the spread gate decides. Demo defaults to prioritized REST because its
  futures WebSocket may be unreachable; mainnet retains the stream fast lane,
  with an explicit Demo override available after operator verification. This
  remains closed-candle execution, not HFT.
- Every generated closed-candle signal is stored separately from trade history,
  including threshold skips and quality rejections. A later candle resolves
  the observation with its actual horizon and directional return, allowing
  strategy quality, risk blocking, and execution success to be diagnosed
  independently.
- Rolling performance governance groups accepted signal outcomes and realized
  trades by symbol, timeframe, and strategy. Wilson confidence bounds and
  minimum sample sizes produce advisory promote, watch, or disable candidates.
  Recommendations never alter live scope configuration automatically.
- Entry and direct-exit execution attempts have a separate durable ledger. It
  records order latency, fill-resolution latency, fill source, actual slippage,
  order recovery, protection-placement latency, and terminal status. Dashboard
  health ratings use rolling failure, slippage, and protection-latency policy.
  Protective SL/TP exits continue through reconciliation because Binance owns
  their trigger timing; their final price and PnL remain in trade history.
- Newly trained ML models use volatility-aware multi-candle labels and record
  their label schema in model metadata. Legacy models remain loadable but emit
  a retraining warning.
- Automatic in-process retraining is disabled by default so model fitting cannot
  delay latency-sensitive market-data and execution work. Train saved models
  with `run_bot.cmd --mode train`; when explicitly enabled, the background worker
  trains only on closed candles, processes scopes sequentially, records audit
  events, and retains the prior model after a failure.
- Historical replay uses the same strategy confidence policy and quality-gate
  construction as live trading, with shifted higher-timeframe context to prevent
  look-ahead. Walk-forward reports contain disjoint train/test folds and are
  required for mainnet strategy approval.
- Portfolio entry checks include same-direction correlated exposure, unfavorable
  funding, visible order-book depth, and estimated fill impact. Non-urgent
  strategies use bounded post-only entry attempts with partial-fill cleanup and
  market fallback; urgent scalp entries remain market-routed.
- ML candidates are saved separately from active models, measured for validation
  quality and feature-distribution drift, and promoted with a retained rollback
  artifact. Live trading never trains or promotes a model inline by default.
- The external supervisor monitors an atomic process heartbeat and performs
  bounded restart with backoff. Rollout governance advances only through tests,
  replay, walk-forward validation, soak, Demo canary, and limited mainnet, where
  separate canary sizing remains active by default.
- Dashboard trade actions are persisted to an atomic SQLite request queue.
  The dashboard never connects to Binance directly: the trading loop claims
  requests, applies normal risk and protection logic to entries, and resolves
  exact audited correlation IDs for reduce-only exits.
- Advanced dashboard entries can require the normal multi-timeframe quality
  gate and use policy-bounded sizing leverage. Pending requests are cancellable,
  duplicate active commands are rejected, and interrupted processing commands
  fail closed for operator review instead of being retried after restart.
- TCPConnector uses configurable DNS cache TTL, connection pooling limits, and
  TCP keepalive. WebSocket reconnect uses random jitter to avoid thundering herd.
- Periodic background clock sync runs at a configurable interval (default 1h) to
  proactively prevent `InvalidNonce` errors at runtime.
- Fatal shutdown submits one aggregate reduce-only exit per Binance symbol,
  then closes the corresponding internal legs from the confirmed fill.
- Failed emergency flattening activates emergency stop.
- Unmanaged exchange positions block reconciliation.
- Missing audited protection blocks reconciliation.
- Repeated account failures stop the trading loop.
- Secrets and URL signatures are redacted from logs.
