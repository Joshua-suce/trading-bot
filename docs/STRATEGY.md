# Trading Strategy

## Overview

The bot uses a **hybrid signal fusion** strategy combining rule-based technical analysis with machine learning predictions. Signals are generated independently and then fused into a single decision.

### Independent Strategy Contracts

Every strategy is governed by a dedicated `StrategyPolicy`. The policy owns its
supported timeframes, confidence floor, risk fraction, stop geometry,
reward/risk target, maximum holding bars, after-cost edge requirement, spread
limit, and walk-forward promotion criteria. Signal generation cannot bypass
this contract.

ML directional labels are cost-adjusted. The minimum labelled move is the
greater of the configured return floor and estimated round-trip fees plus the
required net edge. Model artifacts must match that label schema, horizon, ATR
multiplier, and cost floor before live inference will load them.

Trend entries below one hour, including trend scalps, require an explicitly
aligned closed 1-hour regime. A neutral higher-timeframe regime is not treated
as directional confirmation. Range scalps and confirmed reversal strategies
retain their separate entry rules.

Holding periods are expressed in bars rather than one global wall-clock value.
For example, the trend policy allows 12 bars: one hour on 5-minute data, three
hours on 15-minute data, and twelve hours on 1-hour data. Range and
countertrend positions receive shorter horizons, while breakout positions have
more time to follow through. Scalp horizons remain separately configurable for
1-minute and 3-minute scopes.

Before any strategy submits an entry, its target must cover estimated
round-trip fees, the current bid/ask spread, and the configured minimum net
edge. This prevents trades whose nominal profit target cannot pay their likely
execution costs.

Consecutive-loss cooldowns are tracked per strategy. Portfolio-wide daily-loss
and drawdown limits remain global, but a trend loss streak does not disable an
otherwise healthy scalp or range strategy.

### Market-Regime Routing

Signals are classified before entry qualification:

- **Trend**: EMA 50/200 continuation or confirmed EMA 50/Fibonacci pullback,
  with strict TA/ML agreement.
- **Transition**: emerging direction from a neutral EMA structure, still requiring
  strict TA/ML agreement and directional quality.
- **Range**: low-ADX Bollinger/RSI mean reversion with a wick-rejection candle.
  An inactive ML model does not block the setup, but an active opposing ML
  prediction does.
- **Breakout**: Donchian boundary break with ADX and volume confirmation. A
  neutral 1-hour regime is allowed; an opposing 1-hour regime is not. The close
  must clear the prior boundary by an ATR-normalized buffer, and the breakout
  candle must have a large body and close near its directional extreme. Entries
  that are more than the configured ATR distance from EMA 50 are rejected.
- **Reversal**: counter-trend entry only after multiple reversal votes plus RSI
  exhaustion, momentum turn, directional movement, volume confirmation, and a
  directional rejection candle.
- **Scalp**: an isolated 1-minute/3-minute path using VWAP and EMA 50 location,
  stochastic crossover, MACD momentum, RSI, volume, and candle confirmation.
  Trend scalps and range-edge rejection scalps are supported. A 1-minute or
  3-minute candle that does not satisfy the scalp rules remains neutral and
  cannot fall through to the slower swing rules.
- **Unconfirmed countertrend**: always rejected.

This expands market coverage without treating every price fluctuation as a
trade. Portfolio exposure, loss limits, stop protection, cooldowns, and order
validation apply identically to every strategy.

Correlated technical signals are capped by indicator family before voting. For
example, multiple RSI conditions can contribute only their strongest vote, so
one oscillator cannot create artificial conviction by appearing several times.
Trend, range, breakout, and reversal entries then require support from at least
two independent technical families. Transition signals may use one technical
family because they still require strict ML confirmation. Countertrend signals
also require ML confirmation and must pass the stricter reversal quality gate.

Fusion weights also follow the market regime:

| Regime | Effective TA/ML emphasis |
|---|---|
| Trend / transition | 40% / 60% |
| Range | 70% / 30% |
| Breakout | approximately 60% / 40% |
| Reversal | approximately 53% / 47% |
| Scalp | approximately 80% / 20% |

The ML model remains more influential in established trends, while direct
market-structure evidence receives more weight for range and breakout setups.

Each regime has its own configurable final confidence floor. Trend signals use
the base threshold, while transition, range, breakout, reversal, and
countertrend signals use progressively stricter defaults. Executed trades keep
their strategy attribution through restarts and closure so the dashboard and
history reports can compare win rate and realized PnL by strategy.

### Scalping Controls

Scalping is enabled by the default `1m` and `3m` timeframes, but uses the same
closed-candle decision model as the rest of the bot. A dedicated WebSocket fast
lane detects those closes, while a health-checked REST path takes over whenever
the stream is stale. It is not tick-level or high-frequency execution.

Before a scalp order is submitted, the bot:

- rejects bid/ask spreads above `SCALP_MAX_SPREAD_BPS`;
- requires the take-profit distance to cover estimated round-trip fees, current
  spread, and `SCALP_MIN_NET_EDGE_BPS`;
- uses `SCALP_RISK_PER_TRADE`, which is lower than the normal trade budget;
- applies scalp-specific ATR stop, minimum/maximum stop, and reward/risk values;
- applies a separate short re-entry cooldown.

The runtime also:

- keeps a bounded rolling candle cache and avoids refetching history for healthy
  streamed scopes;
- processes independent market-data scopes concurrently under a fixed worker
  limit;
- schedules REST scalp scans after a separate candle-publication grace period,
  with bounded freshness retries retained for unusually delayed candles;
- rejects scalp signals that arrive beyond the configured close-to-decision
  latency budget; WebSocket events use the strict budget while REST fallback
  receives a separate bounded allowance for network and scheduling overhead;
- moves protection to fee-adjusted break-even after sufficient favorable move;
- trails from the favorable peak after a stronger R-multiple trigger;
- optionally realizes one partial profit and immediately resizes persisted
  protection for the remaining quantity;
- exits positions that exceed the configured maximum holding time.

After repeated WebSocket connection failures, a per-scope circuit breaker
temporarily keeps that scope on REST instead of continuously reconnecting.
Scalp REST work is prioritized ahead of slower timeframes. If Binance Demo
omits ticker bid/ask fields, the cost gate reads the top of the public order
book; it still refuses the entry when no trustworthy two-sided quote exists.

Binance Demo uses the prioritized REST fast lane by default because its futures
WebSocket endpoint may be unavailable from some networks. Mainnet retains
WebSocket streaming. Demo streaming can be enabled explicitly with
`SCALP_DEMO_WEBSOCKET_ENABLED=true` for environments where it has been verified.

These controls reduce avoidable cost and overtrading. They do not guarantee
profitability. Each symbol and timeframe still needs enough demo observations
and closed trades before its scalp performance can be judged.

### Signal Evidence

Every evaluated closed-candle signal is persisted with its strategy, direction,
confidence threshold, quality result, and execution status. The next observed
candle for the same symbol and timeframe resolves the observation with:

- raw market return in basis points;
- direction-adjusted return in basis points;
- direction correctness outside a 5 basis-point neutral zone;
- the actual elapsed evaluation horizon.

This evidence includes skipped and rejected setups. It measures whether the
signal direction was useful, not whether a real trade made money. Trade PnL
still includes sizing, entry fills, fees, slippage, stops, targets, and exit
execution. The dashboard **Signals** tab keeps these two questions separate.

### Performance Governance

Accepted signal outcomes and closed trades are reviewed over a configurable
rolling window for each symbol, timeframe, and strategy. A scope remains
`insufficient` until the minimum signal sample is reached. Mature scopes are
classified as:

- `promote_candidate`: positive directional edge with statistical support and
  no contradictory realized trade record;
- `watch`: mixed or inconclusive evidence;
- `disable_candidate`: persistently weak signal outcomes or sufficiently poor
  realized trade results.

The calculation uses a 95% Wilson interval so small or noisy samples do not
receive strong ratings. These are operator recommendations only. Changing
`DISABLED_STRATEGY_SCOPES` still requires a reviewed configuration update and
restart.

### Execution Quality

Strategy quality and execution quality are evaluated separately. Each attempted
entry records:

- signal price versus validated fill price and resulting slippage;
- order submission and fill-resolution latency;
- whether an ambiguous order was recovered by client order ID;
- whether fill price came directly from the order or exchange recovery;
- stop-loss/take-profit placement latency;
- terminal success or failure stage.

Direct dashboard or shutdown exits record order and fill-resolution timing.
Protective exchange-triggered exits are finalized by reconciliation and remain
visible through trade PnL and audit events. The dashboard **Execution** tab uses
a rolling sample to classify each symbol and phase as `healthy`, `degraded`,
`critical`, or `insufficient`.

## Signal Weighting

| Source | Weight | Description |
|---|---|---|
| Technical Analysis | 40% | Rule-based from 30+ indicators |
| ML Ensemble | 60% | XGBoost + optional LSTM |

## 1. Technical Analysis Signals

### EMA 50/200 Trend Framework
- **Golden cross**: EMA 50 crosses above EMA 200 (bullish, strength 0.75)
- **Death cross**: EMA 50 crosses below EMA 200 (bearish, strength 0.75)
- **Bullish continuation**: price above EMA 50 above EMA 200, positive EMA 50
  slope, rising MACD histogram, and RSI confirmation
- **Bearish continuation**: price below EMA 50 below EMA 200, negative EMA 50
  slope, falling MACD histogram, and RSI confirmation

EMA 50/200 defines the primary trend regime. Short-term indicators cannot
override the regime without enough opposing signal strength.

### Fibonacci Confluence

Fibonacci levels are calculated from the rolling 100-candle swing high and low:

- 23.6%
- 38.2%
- 50.0%
- 61.8%
- 78.6%

The strategy uses the nearest core retracement level (38.2%, 50%, or 61.8%):

- In an EMA 50/200 bullish regime, a reclaim or support hold adds a bullish vote.
- In an EMA 50/200 bearish regime, a rejection or resistance hold adds a bearish
  vote.
- ATR supplies a volatility-aware tolerance around each level.
- Fibonacci does not trigger a trade independently; it blends with EMA, MACD,
  RSI, Bollinger, Donchian, ADX, and volume votes.

### RSI (Strength: 0.3–0.5)
- **Oversold** (RSI < 30): Bullish signal (strength 0.5)
- **Overbought** (RSI > 70): Bearish signal (strength 0.5)
- **Bullish bias** (RSI < 40): Mild bullish (strength 0.3)
- **Bearish bias** (RSI > 60): Mild bearish (strength 0.3)

### MACD (Strength: 0.55)
- **Bullish cross**: MACD histogram crosses above zero
- **Bearish cross**: MACD histogram crosses below zero
- Momentum shift detection

### Bollinger Bands (Strength: 0.25–0.45)
- **Lower band bounce** (close ≤ lower band): Bullish (strength 0.45)
- **Upper band reject** (close ≥ upper band): Bearish (strength 0.45)
- **%B < 0.2**: Oversold (strength 0.25)
- **%B > 0.8**: Overbought (strength 0.25)

### ADX Trend Filter
- ADX < 30 (weak trend): All signal strengths **attenuated by 50%**
- ADX ≥ 30: Full signal strength
- Range: 0–1 multiplier based on ADX/50

### Volume Confirmation
- Volume ratio (current / SMA 20) boosts confidence by up to 30%
- Applied as multiplier: `(0.7 + 0.3 * min(vol_ratio / 2, 1))`

### Signal Voting
All triggered sub-signals vote with their direction (+1/-1) weighted by strength:
```
weighted_dir = Σ(strength_i * direction_i) / Σ(strength_i)
direction = sign(weighted_dir) if |weighted_dir| > 0.15 else 0
```

## 2. Machine Learning Signals

### Feature Engineering (50+ features)

| Category | Features |
|---|---|
| **Lagged Returns** | Return lags at [1, 2, 3, 5, 10, 21] periods |
| **Lagged Log Returns** | Log return lags at same intervals |
| **Rolling Stats** | Mean, std, skew, kurtosis over [5, 10, 21] windows |
| **RSI Derivatives** | RSI moving average, divergence, threshold breaches |
| **MACD Derivatives** | Signal crossovers, histogram % of price |
| **BB Derivatives** | Position within bands, breakout flags |
| **Volatility** | ATR %, HV rank, volatility ratio |
| **Volume** | OBV signal, VWAP distance, volume spike flag |
| **Price vs MA** | Distance from EMA 50/200 |
| **Crossover** | EMA 50/200 cross direction |
| **Fibonacci** | Distance from 23.6/38.2/50/61.8/78.6 retracement levels |

### Target Definition
- **+1** (Buy): Next-period price increases
- **0** (Hold): Next-period price unchanged
- **-1** (Sell): Next-period price decreases

Because EMA 9/21 features were replaced with EMA 50/200 and Fibonacci features,
existing saved ML models must be retrained before their predictions are used.

### XGBoost Classifier
- 3-class classification: -1, 0, +1
- Hyperparameters: 200 estimators, max depth 6, learning rate 0.01, subsample 0.8
- Output: class prediction + confidence (max softmax probability)
- Direction labels are encoded to contiguous XGBoost class IDs during training
  and decoded back to -1/0/+1 for live signals.
- Training writes a scoped artifact such as
  `MODEL_DIR/xgb_BTCUSDT_1h.json`. Trade mode loads it only for that exact
  symbol/timeframe. Other scopes remain TA-only until separately trained.
- Models use XGBoost's native JSON format plus JSON class metadata; pickle-based
  artifacts are rejected.

### LSTM (Optional)
- Architecture: 2-layer LSTM (64 → 32 units) + Dense(16) + Dense(1)
- Sequence length: 60 periods
- Output: regression prediction → sign gives direction, magnitude gives confidence
- Requires TensorFlow (Python < 3.14)

### Ensemble
- XGBoost weight: 60%
- LSTM weight: 40%
- Weighted voting by `weight * direction * confidence`
- Minimum threshold: `|weighted_sum / total_weight| < 0.1` → hold

## 3. Signal Aggregation

```
ta_signal = TechnicalSignal.generate(df_ind)       # direction, strength, source
ml_signal = ModelEnsemble.predict(X_3d)             # direction, confidence, strength

If both available:
  combined = ta.direction * 0.4 * ta.strength + ml.direction * 0.6 * ml.strength
  confidence = 0.4 * ta.strength + 0.6 * ml.strength
Else if ML only:
  combined = ml.direction * ml.strength
  confidence = ml.strength
Else:
  combined = ta.direction * ta.strength
  confidence = ta.strength

direction = sign(combined / confidence) if |combined / confidence| > 0.15 else 0
```

## 4. Risk Management

### Position Sizing
```
risk_amount        = equity * RISK_PER_TRADE (default 1%)
sl_distance        = ATR * 1.0
base_quantity      = risk_amount / sl_distance
max_quantity       = MAX_POSITION_SIZE * equity / entry_price
quantity           = min(base_quantity, max_quantity)
```

### Stop Loss & Take Profit
```
stop_loss   = entry_price ± (ATR × 1.0)
take_profit = entry_price ± (ATR × 1.0 × 2.0)  # 2:1 risk-reward
```

### Portfolio Guards
| Guard | Threshold | Action |
|---|---|---|
| Trade legs per symbol | 1 | Block additional entries |
| Global open trade legs | 6 | Block additional entries |
| Per-symbol notional | 10% of equity | Block additional entries |
| Total open notional | 20% of equity | Block additional entries |
| Max Drawdown | 15% | Block new trades |
| Daily Loss | 5% of equity | Block new trades |
| Consecutive Losses | 3 | Block new trades for 30 minutes |

The consecutive-loss circuit breaker is restored from the audit database after
a restart. It resets immediately after a profitable close or automatically
after `CONSECUTIVE_LOSS_COOLDOWN_SECONDS`. Repeated blocked signals are
deduplicated before alert delivery.

## 5. Entry Criteria

A trade is entered when all conditions are met:
1. Final signal direction is non-zero
2. Signal confidence meets the configured threshold for its strategy
3. No existing leg on the same symbol/timeframe
4. Portfolio guards pass (drawdown, daily loss, consecutive losses)
5. No conflicting exchange-side exposure exists for the symbol
6. Symbol and portfolio exposure limits allow entry

Binance reports the same-direction legs as one net position. The bot reconciles
the exchange quantity against the sum of its audited legs and tracks each
leg's protective orders independently.

## 6. Exit Criteria

A position is exited when any condition is met:
1. **Stop loss hit** — Price reaches SL level
2. **Take profit hit** — Price reaches TP level
3. **Fatal shutdown** - Open exchange positions are closed and audited
