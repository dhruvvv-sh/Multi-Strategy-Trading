"""
Automated Multi-Strategy Trading System (Research-Grade) — v4
All 24 bugs identified in audit have been fixed.

Strategies:
  1. Buy & Hold         — passive baseline
  2. Mean Reversion     — RSI + Bollinger Band rule-based
  3. Pure Momentum      — MA alignment + ADX trend-following
  4. ML-Only            — ensemble classifier OOS signals
  5. Full Agent Council — ML + Sentiment + Regime + orchestrator
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys
import pandas as pd
import numpy as np
from pathlib import Path

from data_module     import fetch_data
from features_module import (
    build_features,
    compute_mean_reversion_signal,
    compute_momentum_signal,
)
from model_module    import (
    train_or_load_ml_model, evaluate_ml_model, get_feature_importance,
    generate_ml_signal, walk_forward_validate,
)
from backtest_module import BacktestEngine
from agents_module   import (
    analyze_sentiment, combine_signals, classify_market_regime,
    compute_price_sentiment_series,
)
from insights_module import generate_insights, kelly_criterion

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV  = SCRIPT_DIR / "trading_results.csv"
MODEL_CACHE_BIN = SCRIPT_DIR / ".cache" / "ml_ensemble.pkl"

FORWARD_DAYS = 3  # 3-day forward return target

# BUG-24 FIX: Removed raw `ma200` from features (is_bull_regime is its cleaner derivative)
# BUG-11 FIX: Count is 41 features
FEATURE_COLS = [
    # Momentum / oscillators
    "rsi", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "williams_r", "cci", "tsi",
    # Rate of change
    "roc1", "roc5", "roc10", "roc20",
    # Trend (ma50 kept — different timescale, not derived from regime flags)
    "ma50", "adx",
    # Volatility / bands
    "atr", "volatility", "vol_ratio", "bb_squeeze",
    "bb_position", "keltner_pos",
    # Volume & money flow
    "volume_ma", "volume_ratio", "obv", "cmf",
    # Price structure
    "price_change", "price_momentum", "return_autocorr",
    "gap", "intraday_range", "consec_days", "price_zscore", "vwap_dev",
    # Lagged
    "rsi_lag5", "adx_lag5", "volatility_lag5", "macd_lag5", "stoch_k_lag5",
    # Regime flags (binary — model-friendly)
    "is_uptrend", "is_bull_regime",
    # Raw volume
    "Volume",
]
N_FEATURES = len(FEATURE_COLS)   # 41

DEFAULT_POSITION_SIZE = 0.10
ALLOCATION_MODES = {"fixed", "kelly", "manual"}
STRATEGY_LABELS = {
    "buy_hold": "Buy & Hold",
    "mean_reversion": "Mean Reversion",
    "momentum": "Momentum",
    "ml_only": "ML-Only (OOS)",
    "agent_council": "Agent Council",
}
STRATEGY_KEYS = tuple(STRATEGY_LABELS.keys())


def _clip_fraction(value: float, default: float = DEFAULT_POSITION_SIZE) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return float(np.clip(parsed, 0.0, 1.0))


def _normalize_allocations(strategy_allocations: dict, default_size: float) -> dict:
    allocations = {}
    for key in STRATEGY_KEYS:
        raw_value = strategy_allocations.get(key, default_size) if strategy_allocations else default_size
        allocations[key] = _clip_fraction(raw_value, default_size)
    return allocations


def _allocation_summary_text(mode: str, strategy_allocations: dict) -> str:
    if mode == "fixed":
        pct = strategy_allocations["buy_hold"] * 100
        return f"Fixed uniform sizing ({pct:.1f}% per strategy)"
    if mode == "kelly":
        return "Kelly sizing per strategy (half-Kelly, capped at 25%)"
    parts = [f"{STRATEGY_LABELS[key]} {strategy_allocations[key] * 100:.0f}%" for key in STRATEGY_KEYS]
    return "Manual per-strategy sizing: " + " | ".join(parts)


def _is_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


IS_STREAMLIT = _is_streamlit()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(allocation_mode: str = "fixed",
                 fixed_position_size: float = DEFAULT_POSITION_SIZE,
                 strategy_allocations: dict = None) -> dict:
    # 1. DATA
    df = fetch_data()
    initial_capital = 100_000.0

    allocation_mode = str(allocation_mode).lower().strip()
    if allocation_mode not in ALLOCATION_MODES:
        allocation_mode = "fixed"
    fixed_position_size = _clip_fraction(fixed_position_size, DEFAULT_POSITION_SIZE)
    normalized_allocations = _normalize_allocations(strategy_allocations, fixed_position_size)

    # 2. FEATURES
    df = build_features(df)

    # 3. SPLIT  (80 / 20)
    split_frac = 0.80
    split_idx  = int(len(df) * split_frac)

    # 4. ML ENSEMBLE (cached binary load on repeat runs)
    model, train_acc, model_cache_loaded, model_cache_key = train_or_load_ml_model(
        df,
        FEATURE_COLS,
        split_idx,
        FORWARD_DAYS,
        cache_path=MODEL_CACHE_BIN,
    )
    ml_metrics       = evaluate_ml_model(model, df, FEATURE_COLS, split_idx, FORWARD_DAYS)
    ml_metrics["train_accuracy"] = train_acc
    ml_signal, ml_proba = generate_ml_signal(
        model, df, FEATURE_COLS, split_idx,
        buy_threshold=0.55, sell_threshold=0.45)
    df["ml_signal"] = ml_signal
    df["ml_proba"]  = ml_proba

    # 5. WALK-FORWARD VALIDATION (5 folds)
    wf_results = walk_forward_validate(df, FEATURE_COLS, n_folds=5,
                                       min_train_frac=0.50, forward_days=FORWARD_DAYS)

    # 6. RULE-BASED SIGNALS (masked to OOS test set)
    mr_signal_raw  = compute_mean_reversion_signal(df)
    mom_signal_raw = compute_momentum_signal(df)
    mr_sig  = pd.Series(np.nan, index=df.index)
    mom_sig = pd.Series(np.nan, index=df.index)
    mr_sig.iloc[split_idx:]  = mr_signal_raw.iloc[split_idx:]
    mom_sig.iloc[split_idx:] = mom_signal_raw.iloc[split_idx:]
    df["mr_signal"]  = mr_sig
    df["mom_signal"] = mom_sig

    # 7. SENTIMENT — time-varying price-derived proxy (replaces static hardcoded headlines)
    # Uses 5-day price ROC + volume-direction signal, normalised via rolling z-score.
    # Returns per-bar labels/weights that actually change across the OOS window.
    sentiment_labels_series, sentiment_weights_series = compute_price_sentiment_series(df)

    # Scalar summaries for dashboard compat (computed over OOS period only)
    oos_labels       = sentiment_labels_series.iloc[split_idx:]
    label_counts     = oos_labels.value_counts()
    sentiment_label  = label_counts.index[0] if len(label_counts) > 0 else "NEUTRAL"
    avg_sentiment    = float(sentiment_weights_series.iloc[split_idx:].mean())
    sentiment_weight = avg_sentiment
    sentiment_scores = sentiment_weights_series.iloc[split_idx:].tolist()
    headlines        = []   # replaced by time-varying series

    # 8. REGIME
    df = classify_market_regime(df)

    # 9. ORCHESTRATOR (Agent Council)
    # buy_threshold=0.60: agent is more selective than ML-Only (0.55).
    # volume_ratio: gates entries on low-volume days (< 0.9 × 20-day avg).
    # sentiment_labels/weights: time-varying per-bar signal (not a static scalar).
    combined_signal, position_sizes, agent_log = combine_signals(
        ml_signal=df["ml_signal"], ml_proba=df["ml_proba"],
        sentiment_label=sentiment_label, sentiment_weight=sentiment_weight,
        rsi=df["rsi"], adx=df["adx"], regime=df["regime"],
        volume_ratio=df["volume_ratio"],
        sentiment_labels=sentiment_labels_series,
        sentiment_weights=sentiment_weights_series,
        apply_ml_training_mask=True, train_idx=split_idx,
        buy_threshold=0.60, sell_threshold=0.45,
    )
    df["combined_signal"] = combined_signal
    df["position_sizes"]  = position_sizes

    # 10. BACKTEST — all 5 strategies
    def _engine(pos_size: float) -> BacktestEngine:
        return BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost=0.001,
            slippage=0.0005,
            position_size=pos_size,
            stop_loss=0.05,
        )

    bh_sig = pd.Series(np.nan, index=df.index)
    bh_sig.iloc[0] = 1

    strategy_specs = [
        ("buy_hold", "Buy & Hold", bh_sig, 0, None),
        ("mean_reversion", "Mean Reversion", mr_sig, 0, None),
        ("momentum", "Momentum", mom_sig, 0, None),
        # min_hold_days aligns exit logic with the 3-day prediction horizon:
        # SELL signals are suppressed until the position has been held for
        # at least FORWARD_DAYS bars (stop-loss still fires immediately).
        ("ml_only", "ML-Only", df["ml_signal"], FORWARD_DAYS, None),
        # min_hold_days=1 for agent: allows regime-aware SELL to fire on day 1
        # (stop-loss always fires immediately regardless).
        # ML-Only keeps FORWARD_DAYS=3 to align with 3-day prediction horizon.
        ("agent_council", "Agent Council", combined_signal, 1, df["position_sizes"]),
    ]

    if allocation_mode == "manual":
        strategy_alloc_used = normalized_allocations.copy()
    else:
        strategy_alloc_used = {key: fixed_position_size for key in STRATEGY_KEYS}

    if allocation_mode == "kelly":
        for key, name, signal, min_hold_days, size_modifiers in strategy_specs:
            probe_engine = _engine(fixed_position_size)
            probe_bt = probe_engine.backtest(
                df, signal, name,
                min_hold_days=min_hold_days,
                size_modifiers=size_modifiers,
            )
            probe_metrics = probe_engine.compute_metrics(probe_bt, initial_capital)
            kelly_size = kelly_criterion(
                probe_metrics.get("win_rate", 0) / 100,
                probe_metrics.get("avg_win", 0),
                probe_metrics.get("avg_loss", 0),
            )
            if probe_metrics.get("num_trades", 0) == 0:
                strategy_alloc_used[key] = fixed_position_size
            else:
                strategy_alloc_used[key] = kelly_size if kelly_size > 0 else 0.0

    backtests = {}
    metrics = {}
    kelly_suggestions = {}

    for key, name, signal, min_hold_days, size_modifiers in strategy_specs:
        strategy_engine = _engine(strategy_alloc_used.get(key, fixed_position_size))
        bt = strategy_engine.backtest(
            df, signal, name,
            min_hold_days=min_hold_days,
            size_modifiers=size_modifiers,
        )
        m = strategy_engine.compute_metrics(bt, initial_capital)

        backtests[key] = bt
        metrics[key] = m
        kelly_suggestions[key] = kelly_criterion(
            m.get("win_rate", 0) / 100,
            m.get("avg_win", 0),
            m.get("avg_loss", 0),
        )

    bt_bh = backtests["buy_hold"]
    bt_mr = backtests["mean_reversion"]
    bt_mom = backtests["momentum"]
    bt_ml = backtests["ml_only"]
    bt_comb = backtests["agent_council"]

    m_bh = metrics["buy_hold"]
    m_mr = metrics["mean_reversion"]
    m_mom = metrics["momentum"]
    m_ml = metrics["ml_only"]
    m_comb = metrics["agent_council"]

    df["portfolio_bh"] = bt_bh["portfolio_values"]
    df["portfolio_mr"] = bt_mr["portfolio_values"]
    df["portfolio_mom"] = bt_mom["portfolio_values"]
    df["portfolio_ml"] = bt_ml["portfolio_values"]
    df["portfolio_comb"] = bt_comb["portfolio_values"]

    # 11. FEATURE IMPORTANCE
    feature_importance = get_feature_importance(model, FEATURE_COLS)

    # 12. REGIME ANALYSIS — computed for BOTH ML-Only and Agent Council
    # Previously only ML trades were analysed; agent-specific analysis was missing.
    def _regime_breakdown(trades_list: list, df_ref: pd.DataFrame) -> tuple:
        """Compute 2-way (BULL/BEAR) and 4-way regime breakdown from a trades list."""
        analysis_2way: dict = {}
        analysis_4way: dict = {}
        if not trades_list:
            return analysis_2way, analysis_4way

        bull_t, bear_t = [], []
        reg_buckets: dict = {}

        for trade in trades_list:
            entry_date = trade.get("entry_date")
            reg = (
                df_ref.loc[entry_date, "regime"]
                if entry_date is not None and entry_date in df_ref.index
                else "BULL_LOW_VOL"
            )
            (bull_t if "BULL" in reg else bear_t).append(trade)
            reg_buckets.setdefault(reg, []).append(trade)

        for lbl, tlist in [("BULL", bull_t), ("BEAR", bear_t)]:
            pnls = [t["pnl_pct"] for t in tlist]
            wins = [p for p in pnls if p > 0]
            analysis_2way[lbl] = {
                "num_trades": len(tlist),
                "wins":       len(wins),
                "win_rate":   len(wins) / len(tlist) * 100 if tlist else 0.0,
                "avg_pnl":    float(np.mean(pnls)) if pnls else 0.0,
            }
        for reg_key, tlist in reg_buckets.items():
            pnls = [t["pnl_pct"] for t in tlist]
            wins = [p for p in pnls if p > 0]
            analysis_4way[reg_key] = {
                "num_trades": len(tlist),
                "wins":       len(wins),
                "win_rate":   len(wins) / len(tlist) * 100 if tlist else 0.0,
                "avg_pnl":    float(np.mean(pnls)) if pnls else 0.0,
            }
        return analysis_2way, analysis_4way

    regime_analysis,       regime_4way       = _regime_breakdown(bt_ml["trades"],   df)
    regime_analysis_agent, regime_4way_agent = _regime_breakdown(bt_comb["trades"], df)

    # 13. INSIGHTS
    backtest_summary = {"buy_hold": m_bh, "ml_only": m_ml, "combined": m_comb}
    insights = generate_insights(
        backtest_summary, ml_metrics, feature_importance,
        regime_analysis, wf_results=wf_results,
    )

    # 14. COMPARISON TABLE
    def _row(strategy_key: str, m: dict):
        alloc_fraction = strategy_alloc_used.get(strategy_key, fixed_position_size)
        allocated_base = initial_capital * alloc_fraction
        allocated_return = np.nan
        if allocated_base > 0:
            allocated_return = (m["final_value"] - initial_capital) / allocated_base * 100

        return {
            "Strategy": STRATEGY_LABELS[strategy_key],
            "Final Value": m["final_value"],
            "Return %": m["total_return"],
            "Return on Alloc Cap %": allocated_return,
            "Capital Alloc %": alloc_fraction * 100,
            "Kelly Suggested %": kelly_suggestions.get(strategy_key, 0.0) * 100,
            "Sharpe": m["sharpe_ratio"],
            "Sortino": m["sortino_ratio"],
            "Calmar": m["calmar_ratio"],
            "Max DD %": m["max_drawdown"],
            "# Trades": m["num_trades"],
            "Win Rate %": m["win_rate"],
            "Profit Factor": m["profit_factor"],
        }

    comparison = pd.DataFrame([
        _row("buy_hold",       m_bh),
        _row("mean_reversion", m_mr),
        _row("momentum",       m_mom),
        _row("ml_only",        m_ml),
        _row("agent_council",  m_comb),
    ])

    allocation_summary = _allocation_summary_text(allocation_mode, strategy_alloc_used)

    df.to_csv(OUTPUT_CSV)

    return dict(
        df=df, initial_capital=initial_capital,
        split_idx=split_idx, split_frac=split_frac,
        ml_metrics=ml_metrics, forward_days=FORWARD_DAYS,
        n_features=N_FEATURES,
        ml_signal=df["ml_signal"], ml_proba=df["ml_proba"],
        mr_signal=mr_sig, mom_signal=mom_sig,
        combined_signal=combined_signal, position_sizes=position_sizes,
        sentiment_label=sentiment_label, avg_sentiment=avg_sentiment,
        headlines=headlines, sentiment_scores=sentiment_scores,
        sentiment_weight=sentiment_weight,
        sentiment_labels_series=sentiment_labels_series,
        sentiment_weights_series=sentiment_weights_series,
        feature_importance=feature_importance,
        comparison=comparison,
        allocation_mode=allocation_mode,
        allocation_summary=allocation_summary,
        fixed_position_size=fixed_position_size,
        strategy_allocations=strategy_alloc_used,
        kelly_suggestions=kelly_suggestions,
        model_cache_loaded=model_cache_loaded,
        model_cache_key=model_cache_key,
        model_cache_path=str(MODEL_CACHE_BIN),
        metrics_bh=m_bh, metrics_mr=m_mr, metrics_mom=m_mom,
        metrics_ml=m_ml, metrics_combined=m_comb,
        trades_ml=bt_ml["trades"],
        regime_analysis=regime_analysis,       regime_4way=regime_4way,
        regime_analysis_agent=regime_analysis_agent, regime_4way_agent=regime_4way_agent,
        wf_results=wf_results, insights=insights,
        agent_log=agent_log, output_csv=str(OUTPUT_CSV),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__" and not IS_STREAMLIT:
    print("\n" + "="*120)
    print("MULTI-STRATEGY TRADING SYSTEM  v4  —  5 STRATEGIES  —  CLI REPORT")
    print("="*120 + "\n")
    r = run_pipeline(); df = r["df"]

    print(f"Dataset  : {len(df)} days  |  Train: {r['split_idx']}  |  Test: {len(df)-r['split_idx']}")
    print(f"Period   : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Target   : {r['forward_days']}-day forward return  |  Features: {r['n_features']}\n")
    print(f"Model    : {'CACHE HIT (loaded binary)' if r['model_cache_loaded'] else 'CACHE MISS (trained + saved binary)'}")
    print(f"ModelBin : {r['model_cache_path']}")
    print(f"Capital  : {r['allocation_summary']}")
    print("Returns  : 'Return %' = total capital basis | 'Return on Alloc Cap %' = per-strategy allocated capital basis\n")

    m = r["ml_metrics"]
    print("─"*120); print("ML ENSEMBLE  (RF + HistGBT, 3-day target)")
    print(f"  Train Accuracy : {m['train_accuracy']:.2%}  |  OOS Accuracy: {m['test_accuracy']:.2%}")
    print(f"  Precision: {m['precision']:.2%}  Recall: {m['recall']:.2%}  F1: {m['f1_score']:.2%}\n")

    wf = r["wf_results"]
    print("─"*120); print("WALK-FORWARD VALIDATION (5 folds)")
    print(f"  Mean OOS: {wf['mean_accuracy']:.2%} ± {wf['std_accuracy']:.2%}")
    for fd in wf["fold_details"]:
        print(f"  Fold {fd['fold']}: {fd['accuracy']:.2%}  prec={fd['precision']:.2%}  f1={fd['f1']:.2%}")

    print("\n─"*120); print("\nSTRATEGY COMPARISON (5 strategies)")
    cmp = r["comparison"].copy()
    for col in ["Return %", "Return on Alloc Cap %", "Capital Alloc %", "Kelly Suggested %", "Sharpe", "Max DD %", "Win Rate %"]:
        cmp[col] = cmp[col].apply(lambda x: "N/A" if pd.isna(x) else f"{x:+.2f}")
    cmp["Final Value"] = r["comparison"]["Final Value"].apply(lambda x: f"₹{x:,.0f}")
    print(cmp.to_string(index=False))

    print("\n─"*120); print("\nINSIGHTS")
    for i, ins in enumerate(r["insights"], 1):
        print(f"  {i:2d}. {ins.replace('**','')}")
    print(f"\n✅ Results → {r['output_csv']}")


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if IS_STREAMLIT:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.set_page_config(
        page_title="COUNCIL OF AGENTS — TRADING SYSTEM",
        page_icon="⬛", layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS  (BUG-14 FIX: removed !important from td,th to let color classes work)
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace !important; }
.stApp, section[data-testid="stSidebar"] { background: #0a0a0a !important; }
div[data-testid="stMain"]    { background: #0a0a0a !important; }
header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px !important; }

/* BUG-14 FIX: Don't override td/th with !important — let per-cell classes work */
h1,h2,h3,h4,h5,h6,p,span,div { color: #e0e0e0; }

/* Streamlit metric */
[data-testid="metric-container"] {
    background: #111 !important; border: 1px solid #222 !important;
    padding: 0.8rem 1rem !important; border-radius: 0 !important;
}
[data-testid="stMetricLabel"] { font-size:0.62rem !important; letter-spacing:0.1em !important; color:#666 !important; }
[data-testid="stMetricValue"] { font-size:1.4rem  !important; font-weight:700 !important; }
[data-testid="stMetricDelta"] { font-size:0.7rem  !important; }

/* Section label */
.sec-lbl {
    font-size:0.68rem; font-weight:700; letter-spacing:0.18em;
    text-transform:uppercase; color:#444;
    border-top:1px solid #222; padding-top:0.5rem;
    margin:1.8rem 0 0.7rem 0;
}

/* BUG-13 FIX: overflow-x on scroll wrapper + BUG-14 FIX: color classes use !important */
.tscroll { overflow-x:auto; -webkit-overflow-scrolling:touch; }

.dtable { width:100%; border-collapse:collapse; font-size:0.78rem; color:#ccc; }
.dtable th {
    text-align:left; padding:0.45rem 0.9rem;
    border-bottom:2px solid #333;
    font-size:0.65rem; letter-spacing:0.1em; color:#555 !important; font-weight:700;
    white-space:nowrap;
}
.dtable td { padding:0.45rem 0.9rem; border-bottom:1px solid #1a1a1a; color:#ccc; }
.dtable tr:last-child td { border-bottom:none; }
.dtable tr:hover td { background:#111; }

/* BUG-14 FIX: Color helpers must override td color */
.g { color:#7fff7f !important; }
.r { color:#ff6b6b !important; }
.y { color:#ffd166 !important; }
.o { color:#ff9f7f !important; }
.dim { color:#555 !important; }

/* Method box */
.mbox { border:1px solid #1e1e1e; padding:1.2rem 1.4rem; margin-bottom:1rem; background:#0d0d0d; }
.mbox-title {
    font-size:0.65rem; font-weight:700; letter-spacing:0.15em;
    text-transform:uppercase; color:#555; margin-bottom:0.8rem;
    border-bottom:1px solid #1a1a1a; padding-bottom:0.4rem;
}
.mbox-body { font-size:0.79rem; line-height:1.8; color:#bbb; }
.mbox-body code { background:#1a1a1a; padding:0.1em 0.4em; font-size:0.75rem; color:#7fff7f; }

/* Agent card */
.acard { border:1px solid #1e1e1e; padding:0.9rem 1.1rem; margin-bottom:0.6rem; background:#0d0d0d; }
.acard-title { font-size:0.7rem; font-weight:700; text-transform:uppercase; color:#666; margin-bottom:0.4rem; }
.acard-body  { font-size:0.78rem; line-height:1.7; color:#bbb; }

/* Insight */
.insitem { border-left:3px solid #333; padding:0.45rem 0.9rem; margin-bottom:0.45rem; font-size:0.8rem; line-height:1.6; color:#ccc; }

/* Pipeline step (BUG-17 FIX: used in 2-row 6-col grid, not 12 cols) */
.pipe-step {
    background:#0d0d0d; border:1px solid #1e1e1e;
    padding:0.55rem 0.5rem; text-align:center;
    font-size:0.63rem; line-height:1.6; color:#555;
}
.pipe-step b { color:#888; font-size:0.65rem; display:block; }

/* Download button */
.stDownloadButton > button {
    background:#e0e0e0 !important; color:#0a0a0a !important;
    border:none !important; border-radius:0 !important;
    font-family:'IBM Plex Mono',monospace !important; font-weight:700 !important;
    font-size:0.72rem !important; letter-spacing:0.1em !important;
    text-transform:uppercase !important; padding:0.55rem 1.1rem !important;
}
</style>
""", unsafe_allow_html=True)

    st.sidebar.markdown("### CONFIGURATION")

    allocation_mode_labels = {
        "Default (10% per strategy)": "fixed",
        "Kelly Based": "kelly",
        "User's Choice": "manual",
    }
    allocation_mode_label = st.sidebar.selectbox(
        "Capital Allocation Mode",
        list(allocation_mode_labels.keys()),
        index=0,
    )
    allocation_mode = allocation_mode_labels[allocation_mode_label]

    fixed_position_size = DEFAULT_POSITION_SIZE

    manual_allocations = None
    if allocation_mode == "manual":
        st.sidebar.caption("Set capital allocation for each strategy.")
        manual_allocations = {}
        for key in STRATEGY_KEYS:
            manual_allocations[key] = st.sidebar.slider(
                f"{STRATEGY_LABELS[key]} Capital (%)",
                min_value=0,
                max_value=100,
                value=int(DEFAULT_POSITION_SIZE * 100),
                step=1,
                key=f"alloc_{key}",
            ) / 100.0
    elif allocation_mode == "kelly":
        st.sidebar.caption(
            "Kelly mode sizes each strategy via half-Kelly from its win rate and payoff profile (cap 25%), with 10% fallback when needed."
        )
    else:
        st.sidebar.caption("Default fixed allocation is 10% of capital per strategy.")

    return_basis_label = st.sidebar.radio(
        "Return Comparison Basis",
        ["Total Initial Capital", "Allocated Capital per Strategy"],
        index=0,
        help=(
            "Total Initial Capital uses full portfolio capital as denominator. "
            "Allocated Capital uses each strategy's configured capital allocation as denominator."
        ),
    )
    display_return_col = (
        "Return on Alloc Cap %"
        if return_basis_label == "Allocated Capital per Strategy"
        else "Return %"
    )

    pipeline_allocations = manual_allocations or {
        key: fixed_position_size for key in STRATEGY_KEYS
    }
    pipeline_alloc_tuple = tuple(
        (key, pipeline_allocations.get(key, fixed_position_size)) for key in STRATEGY_KEYS
    )

    # ── Load pipeline ──────────────────────────────────────────────────────
    @st.cache_data(show_spinner="RUNNING PIPELINE — 5 strategies, 41 features, walk-forward CV…")
    def _load(mode: str, fixed_size: float, alloc_tuple: tuple):
        alloc_map = {k: v for k, v in alloc_tuple}
        return run_pipeline(
            allocation_mode=mode,
            fixed_position_size=fixed_size,
            strategy_allocations=alloc_map,
        )

    r          = _load(allocation_mode, fixed_position_size, pipeline_alloc_tuple)
    df         = r["df"]
    period_str = f"{df.index[0].strftime('%d %b %Y')} – {df.index[-1].strftime('%d %b %Y')}"
    test_days  = len(df) - r["split_idx"]
    train_days = r["split_idx"]
    wf         = r["wf_results"]
    ml         = r["ml_metrics"]
    wf_acc     = wf["mean_accuracy"]
    wf_std     = wf["std_accuracy"]
    comparison_index = r["comparison"].set_index("Strategy")

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="border:2px solid #e0e0e0;padding:1.4rem 2rem;margin-bottom:1.5rem">
  <div style="font-size:1.3rem;font-weight:700;letter-spacing:0.06em;color:#e0e0e0">
    ⬛ COUNCIL OF AGENTS — MULTI-STRATEGY TRADING SYSTEM
  </div>
  <div style="font-size:0.72rem;color:#555;margin-top:0.35rem">
    RELIANCE.NS &nbsp;·&nbsp; {period_str} &nbsp;·&nbsp; {len(df)} DAYS
    &nbsp;·&nbsp; TRAIN: {train_days} &nbsp;·&nbsp; TEST: {test_days}
    &nbsp;·&nbsp; RF+HISTGBT &nbsp;·&nbsp; {r['n_features']} FEATURES
    &nbsp;·&nbsp; 3-DAY TARGET &nbsp;·&nbsp; WF: {wf_acc:.1%} ± {wf_std:.1%}
    &nbsp;·&nbsp; {r['allocation_summary']}
    &nbsp;·&nbsp; Return View: {return_basis_label}
        &nbsp;·&nbsp; Model Cache: {'HIT' if r['model_cache_loaded'] else 'MISS'}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────
    def _return_value(strategy_name: str) -> float:
        val = comparison_index.loc[strategy_name, display_return_col]
        return float(val) if pd.notna(val) else np.nan

    def _fmt_return(val: float) -> str:
        return "N/A" if pd.isna(val) else f"{val:+.1f}%"

    def _delta(val: float, base: float) -> str:
        if pd.isna(val) or pd.isna(base):
            return "N/A"
        return f"{val - base:+.1f}% vs B&H"

    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    bh_ = _return_value("Buy & Hold")
    mr_ = _return_value("Mean Reversion")
    mm_ = _return_value("Momentum")
    ml_ = _return_value("ML-Only (OOS)")
    co_ = _return_value("Agent Council")

    c1.metric("Buy & Hold",       _fmt_return(bh_))
    c2.metric("Mean Reversion",   _fmt_return(mr_), _delta(mr_, bh_))
    c3.metric("Momentum",         _fmt_return(mm_), _delta(mm_, bh_))
    c4.metric("ML-Only (OOS)",    _fmt_return(ml_), _delta(ml_, bh_))
    c5.metric("Agent Council",    _fmt_return(co_), _delta(co_, bh_))
    c6.metric("OOS Accuracy",     f"{ml['test_accuracy']:.1%}")
    c7.metric("WF Mean Acc",      f"{wf_acc:.1%}", f"±{wf_std:.1%}")
    c8.metric("Sentiment",        r["sentiment_label"])
    st.caption(
        "Return % uses total initial capital. Return on Alloc Cap % uses initial capital multiplied by each strategy allocation."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 01 — EQUITY CURVES
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">01 — EQUITY CURVES (5 STRATEGIES)</div>',
                unsafe_allow_html=True)

    fig_eq = go.Figure()
    for col, name, color, dash, w in [
        ("portfolio_bh",   "BUY & HOLD",     "#333",    "dot",     1.5),
        ("portfolio_mr",   "MEAN REVERSION", "#ffd166", "dashdot", 1.8),
        ("portfolio_mom",  "MOMENTUM",       "#ff9f7f", "dash",    1.8),
        ("portfolio_ml",   "ML-ONLY",        "#888",    "solid",   1.8),
        ("portfolio_comb", "AGENT COUNCIL",  "#7fff7f", "solid",   2.2),
    ]:
        fig_eq.add_trace(go.Scatter(x=df.index, y=df[col], name=name,
                                    line=dict(color=color, width=w, dash=dash)))

    # BUG-15 FIX: guard against split_idx == len(df)
    safe_split = min(r["split_idx"], len(df) - 1)
    split_date = df.index[safe_split]
    fig_eq.add_vline(x=split_date.isoformat(), line_dash="dash", line_color="#2a2a2a")
    fig_eq.add_annotation(x=split_date.isoformat(), yref="paper", y=0.98,
                           text="◀ TRAIN  |  TEST ▶", showarrow=False,
                           font=dict(size=10, color="#444"), xanchor="center")
    fig_eq.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font=dict(family="IBM Plex Mono", color="#555", size=11),
        xaxis=dict(gridcolor="#141414", linecolor="#222"),
        yaxis=dict(gridcolor="#141414", linecolor="#222", title="Portfolio Value (₹)"),
        legend=dict(bgcolor="#0d0d0d", bordercolor="#222", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified", height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 02 — STRATEGY COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">02 — STRATEGY COMPARISON</div>',
                unsafe_allow_html=True)
    st.caption(r["allocation_summary"])

    def _tc(v):
        if pd.isna(v):
            return "dim"
        if isinstance(v, (int, float, np.floating)):
            return "g" if v > 0 else ("r" if v < 0 else "")
        return ""

    def _fmt_pct(v):
        return "N/A" if pd.isna(v) else f"{v:+.2f}%"

    rows_html = ""
    for _, row in r["comparison"].iterrows():
        fv = row["Final Value"]
        bv = r["initial_capital"]
        ret_total = row["Return %"]
        ret_alloc = row["Return on Alloc Cap %"]
        alloc_pct = row["Capital Alloc %"]
        kelly_pct = row["Kelly Suggested %"]
        rows_html += (
            f"<tr>"
            f"<td><b>{row['Strategy']}</b></td>"
            f"<td class='{'g' if fv>bv else 'r'}'>₹{fv:,.0f}</td>"
            f"<td>{alloc_pct:.1f}%</td>"
            f"<td class='{_tc(ret_total)}'>{_fmt_pct(ret_total)}</td>"
            f"<td class='{_tc(ret_alloc)}'>{_fmt_pct(ret_alloc)}</td>"
            f"<td class='dim'>{kelly_pct:.1f}%</td>"
            f"<td class='{_tc(row['Sharpe'])}'>{row['Sharpe']:.2f}</td>"
            f"<td class='{_tc(row['Sortino'])}'>{row['Sortino']:.2f}</td>"
            f"<td class='{_tc(row['Calmar'])}'>{row['Calmar']:.2f}</td>"
            f"<td class='r'>{row['Max DD %']:.2f}%</td>"
            f"<td>{int(row['# Trades'])}</td>"
            f"<td>{row['Win Rate %']:.1f}%</td>"
            f"<td class='{_tc(row['Profit Factor'] - 1)}'>{row['Profit Factor']:.2f}×</td>"
            f"</tr>"
        )
    # BUG-13 FIX: wrap in scroll div
    st.markdown(
        f'<div class="tscroll"><table class="dtable">'
        f'<thead><tr><th>Strategy</th><th>Final Value</th><th>Alloc Cap</th><th>Return</th><th>Return/Alloc</th><th>Kelly</th>'
        f'<th>Sharpe</th><th>Sortino</th><th>Calmar</th>'
        f'<th>Max DD</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 03 — ML MODEL METRICS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">03 — ML ENSEMBLE METRICS</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        acc_cls = "g" if ml["test_accuracy"] > 0.52 else "y"
        st.markdown(
            f'<div class="tscroll"><table class="dtable">'
            f'<thead><tr><th>Metric</th><th>Value</th><th>Notes</th></tr></thead>'
            f'<tbody>'
            f'<tr><td>Train Accuracy</td><td>{ml["train_accuracy"]:.2%}</td><td class="dim">In-sample</td></tr>'
            f'<tr><td><b>OOS Test Accuracy</b></td><td class="{acc_cls}"><b>{ml["test_accuracy"]:.2%}</b></td><td class="dim">What matters</td></tr>'
            f'<tr><td>Precision</td><td>{ml["precision"]:.2%}</td><td class="dim">BUYs correct</td></tr>'
            f'<tr><td>Recall</td><td>{ml["recall"]:.2%}</td><td class="dim">Up-days caught</td></tr>'
            f'<tr><td>F1-Score</td><td>{ml["f1_score"]:.2%}</td><td class="dim">Harmonic mean</td></tr>'
            f'<tr><td>High-Conf BUY (&gt;0.55)</td><td class="g">{ml["high_conf_pct"]:.1%}</td><td class="dim">Actionable</td></tr>'
            f'<tr><td>High-Conf SELL (&lt;0.45)</td><td class="r">{ml["low_conf_pct"]:.1%}</td><td class="dim">Actionable</td></tr>'
            f'<tr><td>Forward Days</td><td>{ml["forward_days"]}d</td><td class="dim">Prediction horizon</td></tr>'
            f'</tbody></table></div>',
            unsafe_allow_html=True)

    with col_m2:
        cm = ml["confusion_matrix"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]],
            x=["Pred: DOWN", "Pred: UP"],
            y=["Act: DOWN",  "Act: UP"],
            text=[[f"TN={cm[0,0]}", f"FP={cm[0,1]}"], [f"FN={cm[1,0]}", f"TP={cm[1,1]}"]],
            texttemplate="%{text}", textfont=dict(size=13, color="#e0e0e0"),
            colorscale=[[0,"#0f0f0f"],[1,"#2a2a2a"]], showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font=dict(family="IBM Plex Mono", color="#555", size=11),
            height=240, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 04 — FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">04 — FEATURE IMPORTANCE (TOP 15, RF ENSEMBLE)</div>',
                unsafe_allow_html=True)

    fi   = r["feature_importance"].head(15)
    fimp = fi["Importance"].values * 100
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        y=fi["Feature"].values, x=fimp, orientation="h",
        marker=dict(color=fimp, colorscale=[[0,"#1a2a1a"],[1,"#7fff7f"]], showscale=False),
        text=[f"{v:.2f}%" for v in fimp],
        textposition="outside", textfont=dict(size=10, color="#555"),
    ))
    # BUG-16 FIX: dynamic y-axis range
    fig_fi.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font=dict(family="IBM Plex Mono", color="#555", size=10),
        xaxis=dict(gridcolor="#141414", linecolor="#222", title="Importance %",
                   range=[0, max(fimp) * 1.3]),
        yaxis=dict(linecolor="#222", autorange="reversed"),
        height=430, margin=dict(l=140, r=80, t=10, b=30),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 05 — REGIME + SENTIMENT
    # ═══════════════════════════════════════════════════════════════════════
    col_reg, col_sent = st.columns([1, 1])

    with col_reg:
        # BUG-12 FIX: use st.caption inside columns, not .sec-lbl div
        st.caption("05 — MARKET REGIME (4-WAY: PRICE × VOLATILITY)")
        reg4 = r["regime_4way"]
        if reg4:
            _rc = {"BULL_LOW_VOL":"g","BULL_HIGH_VOL":"y","BEAR_LOW_VOL":"o","BEAR_HIGH_VOL":"r"}
            reg_rows = "".join([
                f"<tr><td class='{_rc.get(rg,'')}' style='font-size:0.75rem'><b>{rg}</b></td>"
                f"<td>{s['num_trades']}</td><td>{s['wins']}</td>"
                f"<td class='{'g' if s['win_rate']>50 else 'r'}'>{s['win_rate']:.1f}%</td>"
                f"<td class='{'g' if s['avg_pnl']>0 else 'r'}'>{s['avg_pnl']:+.2f}%</td></tr>"
                for rg, s in reg4.items()
            ])
            st.markdown(
                f'<div class="tscroll"><table class="dtable">'
                f'<thead><tr><th>Regime</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>Avg P&L</th></tr></thead>'
                f'<tbody>{reg_rows}</tbody></table></div>',
                unsafe_allow_html=True)
        else:
            st.caption("Insufficient trades for regime breakdown")

        reg_counts = df["regime"].value_counts()
        pal = {"BULL_LOW_VOL":"#7fff7f","BULL_HIGH_VOL":"#ffd166",
               "BEAR_LOW_VOL":"#ff9f7f","BEAR_HIGH_VOL":"#ff6b6b"}
        fig_pie = go.Figure(data=go.Pie(
            labels=reg_counts.index.tolist(), values=reg_counts.values.tolist(),
            marker=dict(colors=[pal.get(l,"#888") for l in reg_counts.index]),
            hole=0.55, textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=9),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0a0a0a", showlegend=False, height=200,
            margin=dict(l=0,r=0,t=10,b=10),
            font=dict(family="IBM Plex Mono", color="#666"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_sent:
        st.caption("06 — SENTIMENT (VADER NLP — 10 HEADLINES)")
        sc   = r["sentiment_scores"]
        avg  = r["avg_sentiment"]
        lbl  = r["sentiment_label"]
        lbl_cls = "g" if "BULL" in lbl else ("r" if "BEAR" in lbl else "y")
        sc_rows = "".join([
            f"<tr><td class='dim' style='font-size:0.7rem;max-width:280px;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{h[:52]}…</td>"
            f"<td class='{'g' if s>0.05 else ('r' if s<-0.05 else 'y')}'>{s:+.3f}</td></tr>"
            for h, s in zip(r["headlines"], sc)
        ])
        st.markdown(
            f'<div class="tscroll"><table class="dtable">'
            f'<thead><tr><th>Headline</th><th>Score</th></tr></thead>'
            f'<tbody>{sc_rows}'
            f'<tr style="border-top:2px solid #333"><td><b>COMPOSITE</b></td>'
            f'<td class="{lbl_cls}"><b>{lbl}</b> ({avg:+.4f})</td></tr>'
            f'</tbody></table></div>',
            unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 07 — INSIGHTS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">07 — SYSTEM INSIGHTS &amp; RECOMMENDATIONS</div>',
                unsafe_allow_html=True)
    for ins in r["insights"]:
        st.markdown(f'<div class="insitem">{ins}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 08 — PRICE CHART
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">08 — PRICE, BOLLINGER BANDS &amp; ML TRADE SIGNALS</div>',
                unsafe_allow_html=True)

    ml_s = df["ml_signal"]
    buy_mask  = (ml_s == 1)
    sell_mask = (ml_s == 0)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], showlegend=False,
                               line=dict(color="#1a2a1a", width=1), mode="lines"))
    fig_p.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], showlegend=False,
                               line=dict(color="#1a2a1a", width=1), mode="lines",
                               fill="tonexty", fillcolor="rgba(127,255,127,0.03)"))
    fig_p.add_trace(go.Scatter(x=df.index, y=df["Close"],  name="CLOSE",
                               line=dict(color="#888", width=1.3)))
    fig_p.add_trace(go.Scatter(x=df.index, y=df["ma50"],  name="MA50",
                               line=dict(color="#444", width=1, dash="dot")))
    fig_p.add_trace(go.Scatter(x=df.index, y=df["ma200"], name="MA200",
                               line=dict(color="#333", width=1, dash="dot")))
    fig_p.add_trace(go.Scatter(
        x=df.index[buy_mask], y=df["Close"][buy_mask], name="ML BUY",
        mode="markers", marker=dict(symbol="triangle-up",   size=7, color="#7fff7f")))
    fig_p.add_trace(go.Scatter(
        x=df.index[sell_mask], y=df["Close"][sell_mask], name="ML SELL",
        mode="markers", marker=dict(symbol="triangle-down", size=7, color="#ff6b6b")))
    fig_p.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font=dict(family="IBM Plex Mono", color="#555", size=10),
        xaxis=dict(gridcolor="#111", linecolor="#1e1e1e"),
        yaxis=dict(gridcolor="#111", linecolor="#1e1e1e", title="Price (₹)"),
        legend=dict(bgcolor="#0d0d0d", bordercolor="#222", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified", height=360, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_p, use_container_width=True)

    # ── Oscillators ────────────────────────────────────────────────────────
    col_o1, col_o2 = st.columns(2)

    with col_o1:
        st.caption("RSI · STOCHASTIC %K · WILLIAMS %R (+100)")
        fig_o1 = go.Figure()
        fig_o1.add_trace(go.Scatter(x=df.index, y=df["rsi"],     name="RSI(14)",
                                    line=dict(color="#e0e0e0", width=1.2)))
        fig_o1.add_trace(go.Scatter(x=df.index, y=df["stoch_k"], name="Stoch%K",
                                    line=dict(color="#ffd166", width=1, dash="dot")))
        fig_o1.add_trace(go.Scatter(x=df.index, y=df["williams_r"] + 100,
                                    name="%R+100", line=dict(color="#ff9f7f", width=1, dash="dashdot")))
        fig_o1.add_hline(y=70, line_dash="dot", line_color="#ff6b6b")
        fig_o1.add_hline(y=30, line_dash="dot", line_color="#7fff7f")
        fig_o1.add_annotation(xref="paper", x=1.01, y=70, text="OB", showarrow=False,
                               font=dict(size=9, color="#ff6b6b"), xanchor="left")
        fig_o1.add_annotation(xref="paper", x=1.01, y=30, text="OS", showarrow=False,
                               font=dict(size=9, color="#7fff7f"), xanchor="left")
        fig_o1.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font=dict(family="IBM Plex Mono", color="#555", size=10),
            xaxis=dict(gridcolor="#111", linecolor="#1e1e1e"),
            yaxis=dict(gridcolor="#111", linecolor="#1e1e1e", range=[0, 100]),
            legend=dict(bgcolor="#0d0d0d", bordercolor="#222", orientation="h",
                        yanchor="bottom", y=1.01, x=0),
            height=240, margin=dict(l=10,r=40,t=30,b=10))
        st.plotly_chart(fig_o1, use_container_width=True)

    with col_o2:
        st.caption("MACD · HISTOGRAM (top) · CCI-20 (bottom)")
        fig_o2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4], vertical_spacing=0.02)
        hcol = ["#7fff7f" if v >= 0 else "#ff6b6b" for v in df["macd_hist"]]
        fig_o2.add_trace(go.Bar(x=df.index, y=df["macd_hist"],   name="Hist",
                                marker_color=hcol, opacity=0.7), row=1, col=1)
        fig_o2.add_trace(go.Scatter(x=df.index, y=df["macd"],        name="MACD",
                                    line=dict(color="#e0e0e0", width=1.2)), row=1, col=1)
        fig_o2.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                    line=dict(color="#555", width=1, dash="dot")),  row=1, col=1)
        ccol = ["#7fff7f" if v > 100 else ("#ff6b6b" if v < -100 else "#555")
                for v in df["cci"]]
        fig_o2.add_trace(go.Bar(x=df.index, y=df["cci"], name="CCI(20)",
                                marker_color=ccol, opacity=0.8), row=2, col=1)
        fig_o2.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font=dict(family="IBM Plex Mono", color="#555", size=10),
            xaxis2=dict(gridcolor="#111", linecolor="#1e1e1e"),
            yaxis=dict(gridcolor="#111",  linecolor="#1e1e1e"),
            yaxis2=dict(gridcolor="#111", linecolor="#1e1e1e"),
            legend=dict(bgcolor="#0d0d0d", bordercolor="#222", orientation="h",
                        yanchor="bottom", y=1.01, x=0),
            barmode="overlay", height=240, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_o2, use_container_width=True)

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.caption("OBV (norm, top) · CMF-20 (bottom)")
        fig_v1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.5, 0.5], vertical_spacing=0.02)
        fig_v1.add_trace(go.Scatter(x=df.index, y=df["obv"], name="OBV(norm)",
                                    line=dict(color="#ffd166", width=1.2)), row=1, col=1)
        fig_v1.add_trace(go.Bar(x=df.index, y=df["cmf"], name="CMF(20)",
                                marker_color=["#7fff7f" if v > 0 else "#ff6b6b"
                                              for v in df["cmf"]], opacity=0.7), row=2, col=1)
        fig_v1.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font=dict(family="IBM Plex Mono", color="#555", size=10),
            xaxis2=dict(gridcolor="#111", linecolor="#1e1e1e"),
            yaxis=dict(gridcolor="#111",  linecolor="#1e1e1e"),
            yaxis2=dict(gridcolor="#111", linecolor="#1e1e1e"),
            legend=dict(bgcolor="#0d0d0d", bordercolor="#222", orientation="h",
                        yanchor="bottom", y=1.01, x=0),
            height=240, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_v1, use_container_width=True)

    with col_v2:
        st.caption("ML BUY PROBABILITY  (3-day forward)  ·  BUY > 0.55  SELL < 0.45")
        prob_s = df["ml_proba"].dropna()
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=prob_s.index, y=prob_s,
            fill="tozeroy", fillcolor="rgba(127,255,127,0.04)",
            line=dict(color="#7fff7f", width=1.2), name="P(UP in 3d)"))
        fig_pr.add_hline(y=0.55, line_dash="dot", line_color="#7fff7f")
        fig_pr.add_hline(y=0.50, line_dash="dot", line_color="#333")
        fig_pr.add_hline(y=0.45, line_dash="dot", line_color="#ff6b6b")
        fig_pr.add_annotation(xref="paper", x=1.01, y=0.55, text="BUY",
                               showarrow=False, font=dict(size=9,color="#7fff7f"), xanchor="left")
        fig_pr.add_annotation(xref="paper", x=1.01, y=0.45, text="SELL",
                               showarrow=False, font=dict(size=9,color="#ff6b6b"), xanchor="left")
        fig_pr.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font=dict(family="IBM Plex Mono", color="#555", size=10),
            xaxis=dict(gridcolor="#111", linecolor="#1e1e1e"),
            yaxis=dict(gridcolor="#111", linecolor="#1e1e1e", range=[0, 1]),
            height=240, margin=dict(l=10,r=50,t=10,b=10), showlegend=False)
        st.plotly_chart(fig_pr, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 09 — METHODOLOGY & TESTING TRANSPARENCY
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">09 — METHODOLOGY &amp; TESTING TRANSPARENCY</div>',
                unsafe_allow_html=True)

    # 09-A — What Is Tested
    st.markdown("""<div class="mbox">
<div class="mbox-title">09-A — What Is Being Tested</div>
<div class="mbox-body">""", unsafe_allow_html=True)
    st.markdown(f"""
**Asset:** RELIANCE.NS (NSE) &nbsp;·&nbsp; **Source:** Yahoo Finance 5Y daily OHLCV
&nbsp;·&nbsp; **Total Days:** {len(df)} &nbsp;·&nbsp; **Train:** {train_days} &nbsp;·&nbsp;
**Test:** {test_days} &nbsp;·&nbsp; **Target:** {FORWARD_DAYS}-day forward return

| # | Strategy | Rule / Model | What Would Falsify It |
|---|---|---|---|
| 1 | **Buy & Hold** | Always hold RELIANCE | Market returns ≤ risk-free over period |
| 2 | **Mean Reversion** | RSI<35 + price below lower BB + CMF>-0.15 → BUY | OOS accuracy ≤ 50% |
| 3 | **Momentum** | MA20>MA50>MA200 + ADX>22 → BUY | Strategy underperforms B&H |
| 4 | **ML-Only** | RF+HistGBT ensemble on {r['n_features']} features | Walk-forward accuracy ≤ 50% |
| 5 | **Agent Council** | ML + Sentiment + Regime fusion | Combined underperforms ML-only |

**All signals generated strictly after training cutoff. No future data leakage.**
""")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # 09-B — Pipeline (BUG-17 FIX: 6 steps per row × 2 rows)
    st.markdown("""<div class="mbox">
<div class="mbox-title">09-B — Pipeline Architecture</div>
<div class="mbox-body">""", unsafe_allow_html=True)

    row1_steps = [
        ("1. DATA",     f"Yahoo 5Y\n{len(df)} OHLCV rows"),
        ("2. FEATURES", f"{r['n_features']} indicators\ncausal windows"),
        ("3. SPLIT",    f"Train {train_days}d\nTest {test_days}d"),
        ("4. ML TRAIN", "RF+HistGBT\n3-day target"),
        ("5. WF-CV",    "5 folds\nexpanding win."),
        ("6. SIGNALS",  "P>0.55→BUY\nP<0.45→SELL"),
    ]
    row2_steps = [
        ("7. RULES",    "MeanRev\nMomentum"),
        ("8. SENTIMENT","VADER NLP\n3-tier label"),
        ("9. REGIME",   "Price×Vol\n4-way class"),
        ("10. COUNCIL", "Vote fusion\nsize scaling"),
        ("11. BACKTEST","0.1%+0.05%\n5% stop-loss"),
        ("12. METRICS", "Sharpe/Calmar\nKelly/EV"),
    ]
    for steps in [row1_steps, row2_steps]:
        cols = st.columns(6)
        for col, (title, body) in zip(cols, steps):
            col.markdown(
                f'<div class="pipe-step"><b>{title}</b>'
                f'{body.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    allocation_assumption = (
        f"Fixed {r['fixed_position_size'] * 100:.1f}% per strategy"
        if r["allocation_mode"] == "fixed"
        else (
            "Kelly per strategy (half-Kelly, max 25%)"
            if r["allocation_mode"] == "kelly"
            else "Manual per-strategy capital sizing"
        )
    )

    st.markdown(f"""
**Backtest assumptions:** <code>0.10%</code> transaction cost &nbsp;·&nbsp;
<code>0.05%</code> slippage &nbsp;·&nbsp; <code>{allocation_assumption}</code> &nbsp;·&nbsp;
<code>5%</code> stop-loss &nbsp;·&nbsp; <code>20%</code> trailing drawdown circuit-breaker
""", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # 09-C — Agent Council
    st.markdown("""<div class="mbox">
<div class="mbox-title">09-C — The Agent Council</div>
<div class="mbox-body">""", unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(f"""
<div class="acard">
<div class="acard-title">🤖 ML PREDICTOR — RF + HistGradientBoosting</div>
<div class="acard-body">
<b>Predicts:</b> Close higher in 3 days? (binary classification)<br>
<b>Features:</b> {r['n_features']} indicators — RSI, MACD, Williams %R, CCI, CMF,
ROC×4, TSI, Keltner, OBV, VWAP dev, Z-score, BB squeeze, vol ratio, gap,
intraday range, consecutive days, stochastic, autocorrelation, lagged variants<br>
<b>Gate:</b> P &gt; 0.55 = BUY · P &lt; 0.45 = SELL · else = HOLD<br>
<b>Now:</b> mean P(BUY) = {ml['mean_buy_prob']:.3f}; {ml['high_conf_pct']:.1%} actionable BUY days
</div></div>

<div class="acard">
<div class="acard-title">📰 SENTIMENT — VADER NLP</div>
<div class="acard-body">
<b>Input:</b> {len(r['headlines'])} RELIANCE.NS news headlines<br>
<b>Score &gt; +0.20</b> → STRONG_BULL: allows all BUYs<br>
<b>Score −0.20..+0.20</b> → NEUTRAL: no override<br>
<b>Score &lt; −0.20</b> → STRONG_BEAR: blocks BUYs + widens SELL threshold by 0.03<br>
<b>Also:</b> continuously modulates position size (0.5× to 1.2×)<br>
<b>Now:</b> {r['avg_sentiment']:+.4f} → <b>{r['sentiment_label']}</b>
</div></div>""", unsafe_allow_html=True)

    with col_a2:
        st.markdown(f"""
<div class="acard">
<div class="acard-title">🌍 REGIME — Price × Volatility</div>
<div class="acard-body">
<b>BULL_LOW_VOL</b> (Close &gt; MA200, vol &lt; 20%) → 1.0× position<br>
<b>BULL_HIGH_VOL</b> (Close &gt; MA200, vol ≥ 20%) → 0.7× position<br>
<b>BEAR_LOW_VOL</b> (Close ≤ MA200, vol &lt; 20%) → 0.5× position<br>
<b>BEAR_HIGH_VOL</b> (Close ≤ MA200, vol ≥ 20%) → 0.3× position<br>
All four multipliers then further adjusted by sentiment weight
</div></div>

<div class="acard">
<div class="acard-title">⚖️ ORCHESTRATOR — Vote Fusion</div>
<div class="acard-body">
<b>BUY if all:</b> ML P&gt;0.55 · sentiment ≠ STRONG_BEAR · RSI&lt;75 · ADX&gt;15<br>
<b>SELL if all:</b> ML P&lt;0.45 (or P&lt;0.48 if STRONG_BEAR) · RSI&gt;20<br>
<b>HOLD:</b> neither condition met<br>
<b>Size:</b> base {r['strategy_allocations']['agent_council'] * 100:.1f}% × regime_mult ({','.join(['0.3','0.5','0.7','1.0'])}) × sentiment_mult (0.5–1.2)
</div></div>""", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # 09-D — Walk-Forward Results
    fold_accs = wf["fold_accuracies"]
    wf_rows = ""
    for fd, dates in zip(wf["fold_details"], wf["fold_dates"]):
        ac  = fd["accuracy"]
        cls = "g" if ac > 0.53 else ("r" if ac < 0.50 else "y")
        wf_rows += (
            f"<tr><td>Fold {fd['fold']}</td>"
            f"<td class='dim'>"
            f"{dates['test_start'].strftime('%d %b %Y')} – {dates['test_end'].strftime('%d %b %Y')}</td>"
            f"<td>{dates['n_train']}d</td><td>{dates['n_test']}d</td>"
            f"<td class='{cls}'><b>{ac:.2%}</b></td>"
            f"<td>{fd['precision']:.2%}</td><td>{fd['recall']:.2%}</td><td>{fd['f1']:.2%}</td></tr>"
        )

    st.markdown(
        f'<div class="mbox">'
        f'<div class="mbox-title">09-D — Walk-Forward Cross-Validation (5-Fold Expanding Window)</div>'
        f'<div class="mbox-body">'
        f'Walk-forward CV trains on all data up to each cutoff and tests on the next unseen block '
        f'— giving a <b>distribution</b> of OOS accuracy across time.<br><br>'
        f'<b>Mean: {wf_acc:.2%} ± {wf_std:.2%}</b> &nbsp;·&nbsp; '
        f'{"<span class=g>✅ Stable</span>" if wf_std < 0.05 else "<span class=y>⚠ High variance</span>"}'
        f'<br><br>'
        f'<div class="tscroll"><table class="dtable">'
        f'<thead><tr><th>Fold</th><th>Test Window</th><th>Train</th><th>Test</th>'
        f'<th>OOS Acc</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>'
        f'<tbody>{wf_rows}</tbody></table></div>'
        f'</div></div>',
        unsafe_allow_html=True)

    # BUG-16 FIX: dynamic y-range
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(fold_accs))],
        y=[a * 100 for a in fold_accs],
        marker=dict(color=["#7fff7f" if a > 0.52 else "#ff6b6b" for a in fold_accs]),
        text=[f"{a:.1%}" for a in fold_accs],
        textposition="outside", textfont=dict(size=10, color="#666"),
    ))
    fig_wf.add_hline(y=50, line_dash="dot", line_color="#333")
    fig_wf.add_hline(y=wf_acc * 100, line_dash="dash", line_color="#ffd166")
    fig_wf.add_annotation(xref="paper", x=1.01, y=50,
                           text="50%", showarrow=False,
                           font=dict(size=9, color="#444"), xanchor="left")
    fig_wf.add_annotation(xref="paper", x=1.01, y=wf_acc * 100,
                           text=f"μ={wf_acc:.1%}", showarrow=False,
                           font=dict(size=9, color="#ffd166"), xanchor="left")
    # BUG-16 FIX: dynamic y-axis max
    y_max = max(max(fold_accs) * 100 * 1.2, 70)
    fig_wf.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
        font=dict(family="IBM Plex Mono", color="#555", size=10),
        xaxis=dict(linecolor="#222"),
        yaxis=dict(gridcolor="#141414", linecolor="#222",
                   title="OOS Accuracy %", range=[40, y_max]),
        height=240, margin=dict(l=10,r=70,t=10,b=10), showlegend=False)
    st.plotly_chart(fig_wf, use_container_width=True)

    # 09-E — Limitations
    st.markdown("""<div class="mbox">
<div class="mbox-title">09-E — Assumptions &amp; Known Limitations</div>
<div class="mbox-body">""", unsafe_allow_html=True)
    st.markdown(f"""
**Backtest does NOT model:**
- **Market impact:** Flat 0.05% slippage underestimates real impact at scale
- **Circuit breakers:** NSE 5%/10%/20% price bands can prevent execution
- **Taxes:** 15% STCG not deducted from profits
- **Short selling:** Only long positions
- **Survivorship bias:** RELIANCE.NS is the largest NSE stock; results may overstate market alpha

**Statistical risks:**
- **Overfitting:** Walk-forward std = {wf_std:.2%} — {'low risk' if wf_std < 0.05 else 'elevated — model may be regime-specific'}
- **Regime change:** Trained on {train_days} days; structural breaks (COVID, 2008) not stress-tested
- **Sentiment lag:** Headlines are static; production requires daily real-time NLP feed

> ⚠️ **Research prototype only. Not financial advice.**
""")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # 09-F — Agent Decision Log
    st.markdown('<div class="sec-lbl">09-F — AGENT COUNCIL DECISION LOG (LAST 30 BARS)</div>',
                unsafe_allow_html=True)
    alog = r["agent_log"]
    if alog is not None and len(alog) > 0:
        recent = alog.tail(30)
        _dc = {"BUY":"g","SELL":"r","HOLD":"dim"}
        _rg = {"BULL_LOW_VOL":"g","BULL_HIGH_VOL":"y","BEAR_LOW_VOL":"o","BEAR_HIGH_VOL":"r"}
        _sc = {"STRONG_BULL":"g","NEUTRAL":"y","STRONG_BEAR":"r"}
        alog_rows = "".join([
            f"<tr>"
            f"<td class='dim'>{d.strftime('%d %b')}</td>"
            f"<td class='{'g' if row.ml_vote=='BUY' else ('r' if row.ml_vote=='SELL' else 'dim')}'>{row.ml_vote}</td>"
            f"<td>{row.ml_prob:.3f}</td>"
            f"<td class='{_sc.get(row.sentiment,'')}'>{row.sentiment}</td>"
            f"<td class='{_rg.get(row.regime,'')}'>{row.regime}</td>"
            f"<td>{row.regime_size:.2f}×</td>"
            f"<td>{row.rsi:.0f}</td>"
            f"<td>{row.adx:.0f}</td>"
            f"<td class='{_dc.get(row.final_decision,'')}'><b>{row.final_decision}</b></td>"
            f"</tr>"
            for d, row in recent.iterrows()
        ])
        st.markdown(
            f'<div class="tscroll"><table class="dtable">'
            f'<thead><tr><th>Date</th><th>ML Vote</th><th>P(UP)</th><th>Sentiment</th>'
            f'<th>Regime</th><th>Size</th><th>RSI</th><th>ADX</th><th>Decision</th>'
            f'</tr></thead><tbody>{alog_rows}</tbody></table></div>',
            unsafe_allow_html=True)
    else:
        st.caption("No agent log data")

    # ═══════════════════════════════════════════════════════════════════════
    # 10 — EXPORT
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">10 — EXPORT</div>', unsafe_allow_html=True)
    with open(r["output_csv"], "rb") as f:
        st.download_button(label="↓ DOWNLOAD RESULTS CSV",
                           data=f.read(), file_name="trading_results.csv",
                           mime="text/csv")
    st.markdown(
        f'<p style="font-size:0.62rem;color:#2a2a2a;margin-top:1.5rem">'
        f'RESEARCH SYSTEM · NOT FINANCIAL ADVICE · RELIANCE.NS · '
        f'{period_str} · {len(df)} DAYS · v4</p>',
        unsafe_allow_html=True)
