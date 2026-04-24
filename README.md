# Multi-Strategy-Trading
# 📊 Agent-Based Multi-Strategy Trading System

An experimental multi-agent trading system that combines machine learning, technical indicators, and sentiment analysis to evaluate trading strategies under realistic conditions.

---

## 🚀 Overview

This project simulates a trading system using three independent decision-making components:

* **Technical Agent (ML Model):** Uses a Random Forest classifier trained on indicators like RSI, MACD, MA50, and Volume
* **Sentiment Agent:** Uses VADER sentiment analysis on financial headlines
* **Orchestrator Agent:** Combines signals from both agents with rule-based logic and memory

The system evaluates performance through backtesting on historical stock data.

---

## 🧠 Key Insight

> After removing data leakage and enforcing proper train-test separation, ML-based strategies underperform the baseline Buy & Hold strategy.

This highlights:

* The difficulty of outperforming the market
* The importance of realistic evaluation
* The risks of overfitting in financial ML systems

---

## 📈 Strategies Compared

| Strategy   | Description                                    |
| ---------- | ---------------------------------------------- |
| Buy & Hold | Baseline strategy (always invested)            |
| ML-Only    | Trades based only on Random Forest predictions |
| Combined   | ML + Sentiment + Rule-based Orchestrator       |

---

## 📊 Performance Metrics

The system evaluates each strategy using:

* Final Portfolio Value
* Total Return (%)
* Sharpe Ratio
* Maximum Drawdown (%)
* Number of Trades

---

## ⚙️ Features

* Multi-agent architecture
* Machine learning (Random Forest)
* Technical indicators (RSI, MACD, Moving Average)
* Sentiment analysis (VADER)
* Backtesting engine
* Risk metrics (Sharpe, Drawdown)
* Interactive Streamlit dashboard

---

## 🖥️ Dashboard

The Streamlit interface provides:

* Strategy comparison table
* Portfolio value visualization
* Price chart with trading signals
* Indicator plots (RSI, MACD)
* Sentiment analysis breakdown

---

## 🧪 Methodology

* Historical data fetched using `yfinance`
* Dataset split into train (80%) and test (20%)
* Model trained only on training data
* Predictions generated on unseen data (out-of-sample)
* Backtesting performed strictly on test period

---

## ⚠️ Limitations

* Uses basic features (no advanced feature engineering)
* Sentiment data is simulated (not live news feed)
* No hyperparameter tuning
* Transaction costs are simplified
* Not intended for real trading

---

## 🔮 Future Improvements

* Real-time news sentiment integration
* Advanced models (LSTM, XGBoost)
* Feature engineering (volatility, momentum factors)
* Portfolio optimization
* Live trading integration

---

## ▶️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit app

```bash
streamlit run trading_system.py
```

---

## 📁 Project Structure
```
trading_system.py (main orchestration)
├── data_module.py (data fetching)
├── features_module.py (19 technical features)
├── model_module.py (RF training + eval)
├── backtest_module.py (BacktestEngine)
├── agents_module.py (sentiment + signals)
└── insights_module.py (auto recommendations)

Streamlit Dashboard
├── Tab 1: Strategy Comparison
├── Tab 2: ML Analysis
├── Tab 3: Regime Analysis
├── Tab 4: Insights
└── Tab 5: Detailed Charts
```
---

## Disclaimer

This project is for educational and research purposes only.
It does not constitute financial advice.


