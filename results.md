# Strategy Results & Performance Manifesto (v5)

**Asset:** `RELIANCE.NS` (NSE)  
**Period:** 01 Apr 2021 – 17 Apr 2026 (1,246 days total | 996 Train | 250 Test OOS)  
**Initial Capital:** ₹1,00,000  
**Friction Model:** 0.10% transaction costs | 0.05% slippage | 5% stop-loss | 20% drawdown circuit breaker

---

## 📈 Performance Summary (Default Mode — 10% Allocation)

The table below showcases the final performance of the strategies after resolving the **Sharpe/Sortino cash-drag calculation bias** (assuming cash earns the risk-free rate) and implementing **structural regime-aware entry gates** for the Agent Council.

| Strategy | Final Value | Return % | Return on Alloc Cap % | Capital Alloc % | Sharpe | Sortino | Calmar | Max DD % | Trades | Win Rate % | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Buy & Hold** | ₹99,424 | -0.58% | -5.76% | 10.00% | -0.61 | -0.62 | -0.21 | -0.56% | 1 | 0.00% | 0.00× |
| **Mean Reversion** | ₹100,021 | +0.02% | +0.21% | 10.00% | -0.01 | -0.01 | +0.00 | -0.87% | 1 | 100.00% | 0.00× |
| **Momentum** | ₹99,610 | -0.39% | -3.90% | 10.00% | -0.22 | -0.29 | -0.07 | -1.05% | 2 | 50.00% | 0.10× |
| **ML-Only (OOS)** | ₹101,900 | +1.90% | +19.00% | 10.00% | +0.57 | +1.05 | +0.44 | -0.87% | 16 | 68.75% | 3.09× |
| **Agent Council** | ₹100,786 | +0.79% | +7.86% | 10.00% | +0.46 | +0.90 | +0.41 | -0.39% | 3 | 100.00% | 0.00× |

---

## 🧠 System Manifesto & Key Insights

### 1. Structural Regime Gates vs. Cosmetic Resizing
In previous versions (v4), the Agent Council adjusted position sizes during volatile or bearish regimes but still entered all trades recommended by the ML model. This resulted in the same trade count (16) and a net drag on performance. In v5, **structural regime gating** is enforced:
*   `BEAR_HIGH_VOL`: Blocks entries completely.
*   `BEAR_LOW_VOL`: Elevates the confidence threshold to `ml_prob > 0.65`.
*   **Result:** The Agent Council is highly selective, taking only 3 trades with a **100% win rate** and a significantly lower maximum drawdown (`-0.39%` vs. `-0.87%` for ML-Only).

### 2. Correcting the Risk-Adjusted Ratio Calculations (Cash-Drag Fix)
Historically, the backtester calculated Sharpe and Sortino ratios on raw portfolio values without accounting for cash interest. For strategies holding 90% cash, this introduced a severe math error where standard deviation was compressed but the benchmark subtraction was not, resulting in artificially negative Sharpe ratios (e.g. `-5.90` to `-21.70`).
*   **Fix:** The formulas now correctly model cash earning the risk-free rate (4% p.a.). This scales the excess returns properly, making the metrics financially valid and exposure-invariant.

### 3. Strategy Conviction & Capital Routing
The system allows switching to **Kelly Sizing** or **User's Choice** modes:
*   **Kelly Sizing:** Dynamically channels capital to strategies with the highest observed edge (e.g., allocating ~23.2% to ML-Only) while keeping underperforming strategies at 0%.
*   **User's Choice:** Allows manual allocation via dashboard controls to express trade conviction directly.

---

*Verified: 17 June 2026 | System Version: v5 | Engine: Custom Modular Backtester*
