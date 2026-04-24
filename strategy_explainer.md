# Trading Strategies — Plain English Guide

> This document explains the 5 strategies in the **Council of Agents** trading system in simple language.  
> No maths degree needed. We'll use analogies and examples throughout.

---

## The Big Picture — How the System Works

Think of the system as a **committee of experts** who all look at the same stock (Reliance Industries) and vote on whether to buy, sell, or wait. Each expert has a specialty:

```
📊 Charts & Patterns   → ML Predictor
📰 News Sentiment      → Sentiment Analyst  
🌍 Market Environment  → Regime Classifier
⚖️ Final Vote          → Orchestrator
```

The system tests **5 strategies** to see which approach works best. Let's look at each one.

---

## Strategy 1 — Buy and Hold  
*The "Lazy but Smart" Baseline*

### What It Does
The simplest possible strategy: **buy Reliance on day one and never sell**.  
Just hold it for the entire period (up to 5 years) and collect the gains.

### Why We Need This
Every other strategy has to **beat** Buy & Hold to prove it's adding any value. If an automated strategy can't beat just sitting on the shares, it's probably not worth the effort and transaction costs.

### How It Makes Decisions
1. Day 1: Buy as many Reliance shares as possible with the starting capital (₹1,00,000)
2. Every other day: Do absolutely nothing. Hold.
3. Last day: Sell everything and count the profit.

### Real Example
If Reliance was ₹2,400 in January 2020 and ₹2,950 in January 2025, you made roughly **+22.9%** by doing nothing.

### Strengths
- Zero transaction costs (only 2 trades: one buy, one sell)
- Never misses a bull run
- Simple and transparent

### Weaknesses
- Takes the full hit in crashes (lost ~50% in March 2020)
- No downside protection whatsoever
- Can't extract profits during sideways markets

---

## Strategy 2 — Mean Reversion
*The "Rubber Band" Strategy*

### The Core Idea
Imagine a rubber band being stretched. The further you pull it from the centre, the stronger the force pulling it back. Mean reversion assumes **stock prices work the same way** — they tend to snap back toward their average when pushed too far in either direction.

### What It Watches
Three signals combined:

**1. RSI (Relative Strength Index)**  
Think of RSI as a "temperature gauge" for a stock. It ranges 0–100:
- Above 70 = the stock is "running a fever" (overbought, too many buyers)
- Below 30 = the stock has "gone cold" (oversold, too many sellers)
- The strategy looks for **oversold conditions (RSI < 35)** as BUY signals.

**2. Bollinger Bands**  
Bollinger Bands are like a "price corridor" drawn around the stock price — typically ±2 standard deviations from the 20-day average. When price falls *below the lower band*, it's statistically unusual — a potential bounce point.

**3. CMF (Chaikin Money Flow)**  
This measures whether money is *flowing into* or *flowing out of* the stock. A CMF above -0.15 means money isn't aggressively leaving — so buying into an oversold dip is safer.

### How It Makes Decisions

```
🟢 BUY Signal when ALL THREE are true:
   ✓ RSI < 35    (stock is heavily oversold)  
   ✓ Price < lower Bollinger Band  (price is statistically depressed)
   ✓ CMF > -0.15  (not in full distribution/selling mode)

🔴 SELL Signal when BOTH are true:
   ✓ RSI > 65    (stock has recovered, getting overbought)
   ✓ Price > upper Bollinger Band  (price has stretched too far up)
```

### Real Example
Imagine Reliance has fallen 12% in 2 weeks due to crude oil fears. RSI is 31, price is below the lower BB, and CMF is -0.05. The strategy says BUY — "the rubber band is stretched, it will snap back." Once RSI climbs back to 66 and price touches the upper BB, it sells for a profit.

### Strengths
- Great at capturing short-term bounces
- Performs well in sideways, choppy markets
- Clear entry and exit rules — no judgement calls

### Weaknesses
- **Can catch a falling knife** — sometimes stocks are cheap for a reason (bad earnings, fraud). RSI can stay below 30 for weeks during a real crash.
- Works poorly in strong trending markets (the stock keeps breaking new highs, so a "BUY the dip" signal never arrives at the right moment)

---

## Strategy 3 — Pure Momentum  
*The "Trend is Your Friend" Strategy*

### The Core Idea
Completely opposite to Mean Reversion. Momentum says: **what's going up tends to keep going up** (at least for a while). Don't fight the trend — follow it.

Think of it like surfing: you don't create the wave, you just spot it and ride it.

### What It Watches

**1. The Three Moving Averages (MA20, MA50, MA200)**  
A moving average is just the average price over the last N days, plotted as a line. It smooths out daily noise.

- **MA20** = 20-day average (short-term trend)
- **MA50** = 50-day average (medium-term trend)  
- **MA200** = 200-day average (long-term trend)

When all three are perfectly aligned — **MA20 > MA50 > MA200** — the stock is trending up across *all timeframes at once*. This is the most powerful bullish signal in technical analysis.

**2. ADX (Average Directional Index)**  
ADX measures **trend strength**, not direction. It ranges from 0 to 100:
- ADX < 20 = weak, directionless market (dangerous for trend-following)
- ADX > 25 = strong, directional trend (ideal for momentum)

The strategy only trades when ADX > 22 — it only rides *strong* trends.

**3. Price Rising Day-Over-Day**  
As a final confirmation, the strategy checks that today's price is higher than yesterday's — the trend is actually continuing *right now*, not stalling.

### How It Makes Decisions

```
🟢 BUY Signal when ALL FOUR are true:
   ✓ MA20 > MA50 > MA200  (perfect bullish alignment)
   ✓ ADX > 22             (trend is strong, not just drifting)
   ✓ Price > yesterday's price  (still moving up today)

🔴 SELL Signal when EITHER is true:
   ✓ MA20 < MA50  (short-term trend broken)
   ✓ MA50 < MA200 (medium-term trend broken)
   AND ADX > 20   (there's enough trend for the breakdown to matter)
```

### Real Example
Reliance is in a strong bull run. MA20 = ₹2,840, MA50 = ₹2,610, MA200 = ₹2,390. All aligned. ADX = 38 (strong trend). The strategy buys. Three months later, MA20 drops below MA50 — a sign the trend is exhausted. The strategy sells.

### Strengths
- Captures the big multi-month trends that produce the largest returns
- Low trading frequency (fewer transaction costs)
- Well-suited to strongly trending markets

### Weaknesses
- **Entry is always "late"** — by the time all three MAs align, the stock has already moved 10-15% off the bottom
- **Exit is always late too** — the trend has to actually break before you sell
- Works poorly in sideways/choppy markets (whipsaws — lots of entry/exit signals that cancel out quickly)

---

## Strategy 4 — ML-Only (Machine Learning Ensemble)
*The "Pattern Recognition Computer" Strategy*

### The Core Idea
Instead of using 2-3 hand-picked rules, train a **machine learning model** on 41 different indicators and let it discover which combinations of features predict when Reliance will rise over the next 3 days.

The prediction target is: **"Will the closing price 3 days from now be higher than today's closing price?"** — a yes/no question the model tries to answer for every trading day.

### What It Analyzes — The 41 Features

| Category | Features | What They Capture |
|---|---|---|
| Momentum | RSI, MACD, Stochastic, Williams %R, CCI, TSI | Overbought/oversold conditions |
| Rate of Change | ROC(1d, 5d, 10d, 20d) | Short-term speed of price movement |
| Trend | MA50, ADX | Direction and strength of trend |
| Volatility | ATR, Ann. Volatility, BB Squeeze, Vol Ratio | How much the stock is moving |
| Bands | BB Position, Keltner Position | Where price is within its range |
| Volume & Money | OBV, CMF, Volume Ratio | Whether big money is buying or selling |
| Price Structure | Gap, Intraday Range, Consecutive Days, Z-score, VWAP Dev | Price behaviour patterns |
| Autocorrelation | 5-day return autocorr | Momentum vs mean-reversion regime |
| Lagged | RSI lag5, ADX lag5, Vol lag5, MACD lag5, Stoch lag5 | How indicators looked 5 days ago |
| Regime Flags | Is Uptrend, Is Bull Regime | Market context |

### The Ensemble Model

Two models work together to make predictions:

**Random Forest** — Trains 300 decision trees, each looking at a random subset of features and asking a series of yes/no questions. The 300 trees vote, and the majority wins.

**HistGradientBoosting** — Trains sequentially: each new tree focuses on the mistakes of all previous trees. Excellent at learning complex, non-linear patterns.

Both models output a **probability** (0.0 to 1.0) of the stock rising. The probabilities are averaged with weights (40% RF, 60% HGB).

### The Confidence Gate — Why We Have a "No-Trade Zone"

```
P(rise) > 0.55 → BUY   (confident the stock will rise)
P(rise) < 0.45 → SELL  (confident the stock will fall)
0.45 ≤ P ≤ 0.55 → HOLD (not sure enough — do nothing)
```

This is critical: **markets are mostly random noise**. On ~30-40% of days the model genuinely doesn't know what will happen. Rather than forcing a BUY or SELL (and paying transaction costs to lose money), the model says "I don't know — sit this one out."

### How Overfitting is Prevented

**The Train/Test Split:**  
The model is trained on the first 80% of historical data (e.g., 4 years) and tested on the last 20% (1 year) it has **never seen**. This simulates what would happen in real deployment.

**Walk-Forward Cross-Validation:**  
The model is re-trained 5 times at different cutoff points — fold 1 might use 3 years of training and test on year 4, fold 2 uses 3.4 years and tests on the next block, etc. This produces 5 different accuracy scores, telling us **whether the model is consistently good or just lucky once**.

### Strengths
- Processes 41 indicators simultaneously — no hand-picking
- Self-discovers non-obvious feature combinations
- Confidence gating avoids noise trades
- Walk-forward validation provides honest performance estimates

### Weaknesses
- **Black box** — hard to explain exactly *why* it bought today
- Can overfit to the training period if not carefully validated
- Requires significant historical data (uses 5 years)
- Slower to run than rule-based strategies

---

## Strategy 5 — Agent Council (Full System)
*The "Committee of Experts" Strategy*

### The Core Idea
This is the most sophisticated strategy. It takes the ML signal and **modifies it** using two additional pieces of intelligence: news sentiment and market regime. Three agents work together as a council.

### The Three Agents

**🤖 Agent 1 — ML Predictor**  
Provides the "base" buy/sell signal (exactly as in Strategy 4).

**📰 Agent 2 — Sentiment Analyst**  
Reads 10 recent news headlines about Reliance and scores each one from -1 (very negative) to +1 (very positive) using VADER — a natural language processing algorithm tuned for financial text.

The average score determines the "news mood":
- **STRONG_BULL** (avg > +0.20): Good news environment. Allow all ML buys.
- **NEUTRAL** (avg between -0.20 and +0.20): Mixed news. No change.
- **STRONG_BEAR** (avg < -0.20): Bad news. Block all new buy signals. Also make it easier to trigger a sell (the SELL threshold widens from P<0.45 to P<0.48).

Additionally, the sentiment score **continuously adjusts position size** — more bullish news → slightly larger position; more bearish news → smaller position.

**🌍 Agent 3 — Regime Classifier**  
Looks at two things to classify the overall market environment:

- **Trend:** Is the current price above the 200-day moving average? (Simple: if yes, bull market; if no, bear market)
- **Volatility:** Is the 20-day annualised volatility above 20%? (High volatility = risky, choppy market)

Combined into 4 regimes with different position size multipliers:

| Regime | Condition | Position Size |
|---|---|---|
| 🟢 BULL_LOW_VOL | Price > MA200, Vol < 20% | 1.0× (full position) |
| 🟡 BULL_HIGH_VOL | Price > MA200, Vol ≥ 20% | 0.7× (reduce slightly) |
| 🟠 BEAR_LOW_VOL | Price < MA200, Vol < 20% | 0.5× (defensive) |
| 🔴 BEAR_HIGH_VOL | Price < MA200, Vol ≥ 20% | 0.3× (minimal exposure) |

**⚖️ The Orchestrator — Combining Everything**

```
🟢 FINAL BUY requires ALL FOUR:
   ✓ ML says BUY (P > 0.55)
   ✓ Sentiment is NOT STRONG_BEAR
   ✓ RSI < 75 (not wildly overbought)
   ✓ ADX > 15 (some trend present — not dead flat)

🔴 FINAL SELL requires:
   ✓ ML says SELL (P < 0.45, or P < 0.48 if STRONG_BEAR sentiment)
   ✓ RSI > 20 (not already at extreme oversold)

⚪ HOLD: If neither BUY nor SELL conditions are fully met.

Position size = 10% base × regime_multiplier × sentiment_multiplier
               (Range: ~1.5% to 12% of portfolio per trade)
```

### Real Example Walk-Through

Say today is a Wednesday:
1. ML model predicts P(rise in 3 days) = 0.61 → ML says BUY
2. News: "RIL profits hit record" scores +0.72 → STRONG_BULL sentiment (+1.15× size)
3. Regime: Price > MA200, vol = 16% → BULL_LOW_VOL (1.0× size)
4. RSI = 62 (< 75 ✓), ADX = 29 (> 15 ✓), Sentiment ≠ BEAR ✓
5. → **FINAL: BUY at size 10% × 1.0 × 1.15 = 11.5% of portfolio**

### Strengths
- More conservative than ML alone — requires multi-factor confirmation
- Adapts position size to market conditions automatically
- Sentiment provides a real-world market context layer

### Weaknesses
- More filters = fewer trades. Can miss some opportunities the ML alone would have caught
- Static news headlines (should ideally be refreshed daily in production)
- Each added filter adds its own source of error

---

## Feature Glossary — 41 Indicators Explained Simply

| Feature | Simple Explanation |
|---|---|
| **RSI** | "Is the stock on fire or frozen?" 0-100 temperature gauge. >70=hot, <30=cold |
| **MACD** | "Is momentum accelerating or slowing?" Difference between two EMA lines |
| **MACD Signal** | The 9-day smoothed MACD — crossovers signal trend changes |
| **MACD Histogram** | How fast MACD is moving. Bar above zero = accelerating up |
| **Stochastic %K** | RSI-like, but based on high-low range rather than closes. 0-100 |
| **Stochastic %D** | 3-day smooth of %K. Crossovers are buy/sell signals |
| **Williams %R** | Mirror of Stochastic. -100 to 0. Near 0 = overbought, near -100 = oversold |
| **CCI** | "How far from average?" >+100 = unusually high, <-100 = unusually low |
| **TSI** | Double-smoothed momentum. Crosses above zero = turning bullish |
| **ROC(1d/5d/10d/20d)** | "How much did the price change in last N days?" (raw %) |
| **MA50** | Average price over 50 days. Acts as a support/resistance level |
| **ADX** | "How strong is the current trend?" 0-100. >25 = valid trend |
| **ATR** | "How much does the price typically move each day?" (range) |
| **Volatility** | 20-day annualised price volatility |
| **Vol Ratio** | Short-term vol / long-term vol. >1 = volatility expanding |
| **BB Squeeze** | Are Bollinger Bands tighter than usual? Precedes big moves |
| **BB Position** | Where is price within the Bollinger Band? 0=bottom, 1=top |
| **Keltner Position** | Like BB Position but uses ATR-based bands (less noisy) |
| **Volume MA** | 20-day average daily volume |
| **Volume Ratio** | Today's volume / 20-day average. >1.5 = unusually busy |
| **OBV** | "Accumulation detector" — rising OBV means volume is buying-heavy |
| **CMF** | Fraction of volume that's positive (above midpoint). +ve = accumulation |
| **Price Change** | Today's % return (close-to-close) |
| **Price Momentum** | 5-day return — short-term trend direction |
| **Return Autocorrelation** | Does a run of up days tend to continue? (+ve=momentum, -ve=mean reversion) |
| **Gap** | How much did the price gap up/down at open vs yesterday's close |
| **Intraday Range** | (High - Low) / Close — daily volatility proxy |
| **Consecutive Days** | How many days in a row has the stock been going up or down? |
| **Price Z-Score** | How far is price from its 50-day average in standard deviations? |
| **VWAP Deviation** | Is price above or below volume-weighted average price? |
| **RSI lag5** | What was RSI 5 trading days ago? |
| **ADX lag5** | What was ADX 5 trading days ago? |
| **Vol lag5** | What was volatility 5 trading days ago? |
| **MACD lag5** | What was MACD 5 trading days ago? |
| **Stoch lag5** | What was Stochastic 5 trading days ago? |
| **Is Uptrend** | Is price > 50-day MA? (1=yes, 0=no) |
| **Is Bull Regime** | Is price > 200-day MA? (1=yes, 0=no) |
| **Volume** | Raw daily volume in shares |

---

## Why 5 Strategies Instead of Just One?

Running all 5 simultaneously lets us **answer important questions**:

| Question | How We Answer It |
|---|---|
| Is this market predictable? | Compare B&H baseline return to ML accuracy |
| Does technical analysis actually work? | Compare Mean Reversion and Momentum to B&H |
| Does ML add alpha beyond rules? | Compare ML-Only to rule-based strategies |
| Does sentiment/regime help? | Compare Agent Council to ML-Only |
| Is the ML genuinely learning or just overfitting? | Walk-forward cross-validation across 5 time windows |

If the Agent Council beats all other strategies **consistently** across the walk-forward folds, that's meaningful evidence of a real, exploitable edge.

---

## Important Caveats

> ⚠️ **This is a research system, not a trading recommendation.**

- All results are **backtests** — past performance does not guarantee future results
- Transaction costs (0.1%), slippage (0.05%), and stop-losses (5%) are modelled but may differ from reality
- Short selling is not modelled — only long (buy) positions
- RELIANCE.NS is one of India's most liquid stocks; strategies may not work as well on smaller, less liquid stocks
- The model was trained and tested on specific historical periods; structural market changes could degrade performance

---

*Last updated: April 2026 · System: Council of Agents v4 · Data: Yahoo Finance 5Y RELIANCE.NS*
