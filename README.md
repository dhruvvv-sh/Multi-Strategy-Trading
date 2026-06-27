# Agent-Based Multi-Strategy Trading System (v5)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ensemble-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An experimental, regime-aware multi-agent trading system that combines ensemble machine learning models, technical indicators, and NLP sentiment analysis to simulate, backtest, and optimize trading strategies on historical stock data under realistic market friction conditions.

---

## Key Features

*   **Ensemble ML Predictor**: Integrates a voting classifier combining `RandomForest` and `HistGradientBoosting` models, trained on a comprehensive set of 41 technical features to target a 3-day forward return.
*   **Regime-Aware Agentic Council**: Orchestrates trading signals by gating entries and dynamically scaling position sizes based on a 4-way market regime classifier (Price Trend × Adaptive Volatility) and news sentiment weight.
*   **NLP Sentiment Agent**: Analyzes news headlines using the VADER sentiment analyzer, generating compound scores that modulate buy/sell thresholds and exposure size.
*   **Custom Modular Backtester**: Features a production-grade simulation engine that accounts for 0.10% transaction costs, 0.05% slippage, 5% stop-losses, and a 20% trailing drawdown circuit breaker.
*   **Textbook-Correct Risk Ratios**: Computes exposure-adjusted Sharpe, Sortino, and Calmar ratios by assuming cash balances earn the risk-free rate, eliminating cash-drag bias.
*   **Dynamic Capital Routing**: Implements the Kelly Criterion (half-Kelly sizing) to automatically allocate optimal capital percentages across strategies.
*   **Interactive Terminal-Style UI**: A premium, high-density Streamlit dashboard built with dark-mode styling for real-time visualization of equity curves, feature importances, and decision logs.

---

## System Architecture

```
trading_system.py (Main Orchestration & Dashboard UI)
├── data_module.py (Deterministic Data Fetching & Caching)
├── features_module.py (41 Technical Features & Indicators)
├── model_module.py (Walk-Forward Ensemble Model Training)
├── agents_module.py (Sentiment NLP & 4-Way Regime Gates)
├── backtest_module.py (Backtest Engine & Performance Metrics)
└── insights_module.py (Kelly Allocations & System Recommendations)
```

---

## Strategies Compared

| Strategy | Description |
| :--- | :--- |
| **Buy & Hold** | Always invested baseline strategy. |
| **Mean Reversion** | Rules-based strategy using RSI oversold levels, Bollinger Bands, and Chaikin Money Flow. |
| **Momentum** | Rules-based strategy based on moving average crossovers and ADX trend strength. |
| **ML-Only (OOS)** | Trades strictly based on the Ensemble Classifier's predictions. |
| **Agent Council** | Orchestrates signals by passing ML-Only signals through VADER NLP and regime-aware entry gates. |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run CLI Backtester
```bash
python3 trading_system.py
```

### 3. Launch Streamlit UI Dashboard
```bash
bash run_dashboard.sh
```

---

## Sample Performance (NSE: RELIANCE.NS)

Tested over **250 days out-of-sample** (OOS) from **2021-04-01 to 2026-04-17**:

*   **Buy & Hold**: `-0.58%` Return | Sharpe `-0.61` | Max Drawdown `-0.56%`
*   **ML-Only (OOS)**: `+1.90%` Return (`+19.00%` on allocated capital) | Sharpe `+0.57` | Max Drawdown `-0.87%`
*   **Agent Council**: `+0.79%` Return (`+7.86%` on allocated capital) | Sharpe `+0.46` | Max Drawdown `-0.39%` | **100% Win Rate**

---

## Disclaimer

This repository is a research prototype developed for educational and experimental purposes. It does not constitute financial advice. Past performance is not indicative of future results.
