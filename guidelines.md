## Page 1

# Automated Multi-Strategy Trading via Machine Learning and Agentic AI
**V4 Implementation Framework**

**Aniketh Arunkumar**
MIT CCE-B | Reg No: 240953526 | Roll: 39

**Dhruv Sharma**
MIT CCE-B | Reg No: 240953504 | Roll: 38

**Muhir Kapoor**
MIT CCE-B | Reg No: 240953576 | Roll: 43

**Aashvi Budhia**
MIT CCE-B | Reg No: 240953578 | Roll: 44

**Rishi Malaya**
MIT CCE-B | Reg No: 240953688 | Roll: 59

**Abstract**—Retail investors in the Indian stock market (NSE) struggle to balance technical noise with fundamental sentiment. This project implements an autonomous "Council of Agents" trading system that synthesizes 41 technical indicators, VADER-based news sentiment, and market regime classification. We evaluate five distinct strategies—Buy & Hold, Mean Reversion, Momentum, ML-Ensemble, and Agent Council—validated via 5-fold Walk-Forward Cross-Validation on 5 years of historical RELIANCE.NS data.

**Index Terms**—Artificial Intelligence, Machine Learning, FinTech, Agentic AI, Indian Stock Market, Ensemble Learning.

## I. PROBLEM STATEMENT

Retail investors in the Indian stock market often struggle to balance technical indicators with fundamental health. Manual analysis for Swing and Momentum trading is time-consuming and prone to emotional bias. Furthermore, traditional backtesting often ignores slippage and transaction costs, leading to unrealistic performance expectations. There is a critical need for integrated systems that process quantitative ML signals alongside qualitative sentiment to provide holistic, risk-managed decisions.

## II. OBJECTIVE

The primary objective is to implement a modular trading framework, the “Council of Agents,” capable of:
*   **Technical Intelligence:** Training a soft-voting ensemble (Random Forest + HistGradientBoosting) to predict 3-day forward returns based on 41 curated features.
*   **Sentiment Intelligence:** Utilizing VADER NLP to parse financial news headlines and modulate position sizing based on qualitative mood.
*   **Regime Awareness:** Classifying market states into 4 quadrants (Bull/Bear × Low/High Vol) to adapt risk exposure.
*   **Robust Validation:** Implementing a custom backtesting engine with 0.1% costs, 0.05% slippage, and 5-fold Walk-Forward CV to ensure OOS (Out-of-Sample) stability.

---

## Page 2

## III. METHODOLOGY & STRATEGIES

The system implements five specific strategies to benchmark performance:

### A. Implemented Strategies
1.  **Buy & Hold (Baseline):** Passive investment in RELIANCE.NS over the 5-year period.
2.  **Mean Reversion:** Rule-based capture of oversold bounces using **RSI < 35**, price below **Lower Bollinger Band**, and **CMF > -0.15** (money flow confirmation).
3.  **Pure Momentum:** Trend-following using perfect MA alignment (**MA20 > MA50 > MA200**) gated by **ADX > 22** (trend strength).
4.  **ML-Only Ensemble:** A soft-voting classifier (RF + HistGBT) trained on 41 features. Decisions are gated by a confidence threshold (**P > 0.55 for BUY**, **P < 0.45 for SELL**).
5.  **Agent Council (Fusion):** The flagship strategy combining ML signals with Sentiment overrides and Regime-based position scaling.

### B. The Technical Agent (Ensemble ML)
This module generates predictive signals using 41 indicators:
*   **Momentum/Volume:** RSI, MACD, OBV, CMF, Williams %R, CCI, TSI.
*   **Volatility/Structure:** ATR, BB Squeeze, Keltner Position, VWAP Deviation.
*   **ML Model:** A hybrid ensemble of **Random Forest** (for robustness) and **HistGradientBoosting** (for non-linear patterns) predicting a **3-day forward horizon**.

### C. The Fundamental Agent (Sentiment NLP)
Utilizes the **VADER** lexicon to analyze news headlines. It identifies "Strong Bear" news environments to block BUY signals and "Strong Bull" environments to boost position confidence. It continuously calculates a **Sentiment Weight** (-1 to +1) used for dynamic sizing.

### D. The Regime Analyst
Classifies the market into four quadrants:
*   **BULL / BEAR:** Price relative to the 200-day Moving Average.
*   **LOW / HIGH VOL:** 20-day annualized volatility relative to a 20% threshold.
*   Multipliers (0.3x to 1.0x) are applied based on the risk profile of each regime.

### E. Backtesting & Validation Framework
The system avoids "Backtest Overfitting" via:
*   **Walk-Forward CV:** 5-fold expanding window validation to check OOS accuracy distribution.
*   **Horizon-Aligned Exits:** Suppression of SELL signals for the first 3 days after entry to allow the 3-day prediction target to manifest.
*   **Realistic Friction:** Integration of 10bps transaction costs and 5bps slippage per trade.

## IV. OUTCOMES

The system generates a high-fidelity Streamlit dashboard providing:
1.  **Equity Curve Comparison:** Real-time benchmarking of the 5 strategies.
2.  **Trade Intelligence:** Full decision logs explaining the "why" behind every Agent Council move.
3.  **Statistical Validity:** Sharpe, Sortino, Calmar, and Kelly Criterion metrics for risk-adjusted evaluation.

**REFERENCES**
*   M. Lopez de Prado, “Advances in Financial Machine Learning,” Wiley, 2018.
*   Yang et al., “FinGPT: Democratizing Financial Large Language Models,” Proc. IJCAI, 2023.
*   VADER Sentiment Analysis, “A Rule-Based Model for Sentiment Analysis of Social Media Text,” 2014.
