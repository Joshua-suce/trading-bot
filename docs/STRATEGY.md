# Trading Strategy

## Overview

The bot uses a **hybrid signal fusion** strategy combining rule-based technical analysis with machine learning predictions. Signals are generated independently and then fused into a single decision.

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
risk_amount        = equity * RISK_PER_TRADE (default 2%)
sl_distance        = ATR * 1.5
base_quantity      = risk_amount / sl_distance
max_quantity       = MAX_POSITION_SIZE * equity / entry_price
quantity           = min(base_quantity, max_quantity)
```

### Stop Loss & Take Profit
```
stop_loss   = entry_price ± (ATR × 1.5)
take_profit = entry_price ± (ATR × 1.5 × 2.0)  # 2:1 risk-reward
```

### Trailing Stop
- Activates when price moves 1× ATR in profit direction
- Updates stop on each candle to lock in profit

### Portfolio Guards
| Guard | Threshold | Action |
|---|---|---|
| Max Drawdown | 15% | Block new trades |
| Daily Loss | 5% of equity | Block new trades |
| Consecutive Losses | 3 | Block new trades |

## 5. Entry Criteria

A trade is entered when all conditions are met:
1. Final signal direction is non-zero
2. Signal confidence ≥ 0.4
3. No existing position on the symbol
4. Portfolio guards pass (drawdown, daily loss, consecutive losses)

## 6. Exit Criteria

A position is exited when any condition is met:
1. **Stop loss hit** — Price reaches SL level
2. **Take profit hit** — Price reaches TP level
3. **Signal reversal** — Opposite signal with confidence > 0.4
4. **Controlled shutdown** - Open exchange positions are closed and audited
