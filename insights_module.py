"""
Insights generation module — derive actionable conclusions from backtest results.
Extended with: Kelly Criterion sizing, Expected Value per trade, walk-forward insights.
"""
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion: optimal fraction of capital to risk per trade.
    Kelly% = W - (1-W)/R  where W = win_rate, R = avg_win / avg_loss.
    Capped at 25% (half-Kelly) for safety.

    BUG-10 FIX: When avg_loss == 0 (all wins), return max half-Kelly (0.25)
    rather than 0.0 (was failing the guard `avg_loss <= 0`).
    """
    if win_rate <= 0:
        return 0.0
    if avg_loss <= 0:
        # Perfect win rate — Kelly says bet everything; cap at half-Kelly max
        return 0.25
    R = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / R
    return float(np.clip(kelly * 0.5, 0.0, 0.25))   # half-Kelly, max 25%



def expected_value_per_trade(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Expected P&L per trade in % terms.
    EV = (win_rate × avg_win) - ((1 - win_rate) × avg_loss)
    """
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INSIGHTS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(
    backtest_results: dict,
    ml_metrics: dict,
    feature_importance: pd.DataFrame,
    regime_analysis: dict,
    wf_results: dict = None,
) -> list:
    """
    Generate actionable insights from backtesting results.

    Args:
        backtest_results : dict with keys "buy_hold", "ml_only", "combined"
        ml_metrics       : dict from evaluate_ml_model()
        feature_importance: DataFrame from get_feature_importance()
        regime_analysis  : dict from analyze_strategy_by_regime()
        wf_results       : optional dict from walk_forward_validate()

    Returns list of insight strings (markdown-compatible).
    """
    insights = []

    bh_result       = backtest_results.get("buy_hold", {})
    ml_result       = backtest_results.get("ml_only",  {})
    combined_result = backtest_results.get("combined", {})

    bh_return       = bh_result.get("total_return", 0)
    ml_return       = ml_result.get("total_return", 0)
    combined_return = combined_result.get("total_return", 0)

    # ── 1. Market environment ─────────────────────────────────────────────────
    if bh_return < -10:
        insights.append(
            "🔴 **Bearish Market**: Buy & Hold lost {:.1f}%. "
            "Challenging environment for passive strategies.".format(abs(bh_return)))
    elif bh_return < 0:
        insights.append(
            "🟡 **Sideways/Weak Market**: Buy & Hold returned {:.1f}%. "
            "Flat to slightly negative conditions.".format(bh_return))
    elif bh_return < 15:
        insights.append(
            "🟢 **Moderate Uptrend**: Buy & Hold returned +{:.1f}%.".format(bh_return))
    else:
        insights.append(
            "🚀 **Strong Bull Market**: Buy & Hold returned +{:.1f}% — strong market tailwind.".format(bh_return))

    # ── 2. ML accuracy ───────────────────────────────────────────────────────
    test_acc = ml_metrics.get("test_accuracy", 0)
    if test_acc < 0.52:
        insights.append(
            "⚠️ **Weak ML Signal**: OOS accuracy ({:.1%}) barely above random (50%). "
            "Model has limited directional predictive power.".format(test_acc))
    elif test_acc < 0.55:
        insights.append(
            "⚡ **Marginal Predictability**: OOS accuracy ({:.1%}) weak but non-random. "
            "Use with caution — small edge requires tight risk management.".format(test_acc))
    else:
        insights.append(
            "✅ **Solid ML Signal**: OOS accuracy ({:.1%}). "
            "Ensemble shows reasonable directional power.".format(test_acc))

    # ── 3. High-confidence signal rate ───────────────────────────────────────
    high_conf = ml_metrics.get("high_conf_pct", 0)
    low_conf  = ml_metrics.get("low_conf_pct",  0)
    insights.append(
        "🎯 **Signal Conviction**: {:.1%} of test days had high-confidence BUY signals (P>0.60); "
        "{:.1%} had high-confidence SELL signals (P<0.40). "
        "Remaining days = HOLD (no edge).".format(high_conf, low_conf))

    # ── 4. Strategy comparison ────────────────────────────────────────────────
    if ml_return > bh_return and ml_return > 0:
        insights.append(
            "✅ **ML Outperforms B&H**: ML-only +{:.1f}% vs B&H +{:.1f}%.".format(
                ml_return, bh_return))
    elif ml_return > bh_return:
        insights.append(
            "📈 **ML Loses Less**: ML-only {:.1f}% vs B&H {:.1f}%. "
            "Active management preserved capital.".format(ml_return, bh_return))
    else:
        insights.append(
            "❌ **ML Underperforms B&H**: B&H {:.1f}% > ML {:.1f}%. "
            "Market may be trending too cleanly for ML to add value.".format(
                bh_return, ml_return))

    # ── 5. Combined vs ML ────────────────────────────────────────────────────
    delta = combined_return - ml_return
    if delta > 0:
        insights.append(
            "✅ **Sentiment + Regime Adds +{:.1f}%**: Combined ({:.1f}%) > ML-only ({:.1f}%). "
            "Agent council improves over pure ML.".format(delta, combined_return, ml_return))
    else:
        insights.append(
            "❌ **Sentiment/Regime Costs {:.1f}%**: Combined ({:.1f}%) < ML-only ({:.1f}%). "
            "Sentiment/regime filters are being too conservative in this period.".format(
                abs(delta), combined_return, ml_return))

    # ── 6. Expected Value + Kelly ─────────────────────────────────────────────
    wr      = ml_result.get("win_rate",  0) / 100
    avg_win = ml_result.get("avg_win",   0)
    avg_los = ml_result.get("avg_loss",  0)

    ev = expected_value_per_trade(wr, avg_win, avg_los)
    kc = kelly_criterion(wr, avg_win, avg_los)

    ev_sign = "✅" if ev > 0 else "❌"
    insights.append(
        "{} **Expected Value per Trade**: {:.2f}% "
        "(Win rate: {:.1f}%, Avg Win: +{:.2f}%, Avg Loss: -{:.2f}%).".format(
            ev_sign, ev, wr * 100, avg_win, avg_los))

    if kc > 0:
        insights.append(
            "📐 **Kelly Optimal Size**: {:.1f}% of capital per trade (half-Kelly). "
            "Current system uses 10% — {}.".format(
                kc * 100,
                "Kelly suggests MORE aggression" if kc > 0.10 else "conservative vs Kelly"))
    else:
        insights.append(
            "⚠️ **Kelly = 0**: Negative or zero edge — Kelly advises NO position. "
            "System should not be deployed as-is in live markets.")

    # ── 7. Trade frequency ────────────────────────────────────────────────────
    n_trades = ml_result.get("num_trades", 0)
    if n_trades > 20:
        insights.append(
            "⚠️ **High Trade Frequency**: {} trades. "
            "At 0.1% cost + 0.05% slippage, friction alone could cost "
            "~{:.1f}% over the period.".format(n_trades, n_trades * 0.15))
    elif n_trades >= 5:
        insights.append(
            "📊 **Moderate Activity**: {} trades — reasonable signal selectivity.".format(n_trades))
    else:
        insights.append(
            "🔇 **Very Few Trades**: Only {} trades. "
            "Confidence threshold may be too strict.".format(n_trades))

    # ── 8. Regime performance ─────────────────────────────────────────────────
    if regime_analysis:
        best_regime = max(regime_analysis, key=lambda r: regime_analysis[r].get("win_rate", 0))
        best_wr     = regime_analysis[best_regime].get("win_rate", 0)
        insights.append(
            "📈 **Regime Edge**: Strongest performance in **{}** "
            "({:.1f}% win rate). Consider disabling trading in other regimes.".format(
                best_regime, best_wr))

    # ── 9. Walk-forward stability ─────────────────────────────────────────────
    if wf_results and wf_results.get("fold_accuracies"):
        mean_wf = wf_results["mean_accuracy"]
        std_wf  = wf_results["std_accuracy"]
        min_wf  = min(wf_results["fold_accuracies"])
        max_wf  = max(wf_results["fold_accuracies"])

        stability = "✅ Stable" if std_wf < 0.04 else "⚠️ Unstable"
        insights.append(
            "🔁 **Walk-Forward Stability**: {stability} — "
            "mean OOS accuracy {mean:.1%} ± {std:.1%} across {n} folds "
            "(range: {lo:.1%}–{hi:.1%}). "
            "{note}".format(
                stability=stability,
                mean=mean_wf, std=std_wf,
                n=len(wf_results["fold_accuracies"]),
                lo=min_wf, hi=max_wf,
                note="Model generalises consistently." if std_wf < 0.04
                     else "High variance — model may be overfitting to recent data."))

    # ── 10. Risk-adjusted ────────────────────────────────────────────────────
    sharpe = ml_result.get("sharpe_ratio", 0)
    if sharpe > 1.0:
        insights.append("✅ **Strong Risk-Adjusted Return**: Sharpe {:.2f} (>1 is excellent).".format(sharpe))
    elif sharpe > 0:
        insights.append("🟡 **Positive but Modest Sharpe**: {:.2f}. More risk than reward.".format(sharpe))
    else:
        insights.append("🔴 **Negative Sharpe**: {:.2f}. Strategy destroys risk-adjusted value.".format(sharpe))

    # ── 11. Top features ──────────────────────────────────────────────────────
    top_features = feature_importance.head(3)["Feature"].tolist()
    insights.append(
        "🔧 **Top Predictive Features**: {} — "
        "these features drive the most ML decisions. "
        "Monitor their data quality closely.".format(", ".join(top_features)))

    # ── 12. Final recommendation ──────────────────────────────────────────────
    deploy_score = sum([
        ml_return > bh_return,
        ev > 0,
        kc > 0,
        sharpe > 0.5,
        test_acc > 0.52,
    ])

    if deploy_score >= 4:
        insights.append(
            "✅ **RECOMMENDATION — DEPLOY WITH CONFIDENCE**: "
            "{}/5 criteria passed. Positive edge, reasonable risk-adjustment, "
            "and ML outperforms baseline.".format(deploy_score))
    elif deploy_score >= 2:
        insights.append(
            "⚠️ **RECOMMENDATION — PAPER TRADE FIRST**: "
            "{}/5 criteria passed. Mixed evidence — validate on more data "
            "before committing real capital.".format(deploy_score))
    else:
        insights.append(
            "❌ **RECOMMENDATION — DO NOT DEPLOY**: "
            "Only {}/5 criteria passed. Refine signal generation, "
            "try different features or asset class.".format(deploy_score))

    return insights
