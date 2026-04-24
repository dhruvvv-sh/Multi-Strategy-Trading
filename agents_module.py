"""
Trading agents module — multi-agent decision system.

Agents:
  1. SentimentAgent   — Price-derived time-varying proxy (ROC + volume direction)
                        → STRONG_BULL / NEUTRAL / STRONG_BEAR per OOS bar
  2. RegimeAgent      — Price × Volatility (adaptive 80th-pctile threshold)
                        → 4-way regime classification
  3. OrchestratorAgent— Confidence-weighted vote fusion with regime-aware sizing
                        AND regime-aware exit thresholds → final BUY / SELL / HOLD
"""
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sentiment(headlines: list) -> tuple:
    """
    Analyze sentiment of news headlines using VADER compound scores.
    Kept for backward compatibility / one-shot headline scoring.

    Returns:
        avg_sentiment   : float (mean compound score)
        sentiment_label : str  ("STRONG_BULL" | "NEUTRAL" | "STRONG_BEAR")
        sentiment_scores: list of per-headline compound scores
        sentiment_weight: float used by orchestrator (-1 to +1, continuous)
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg = float(np.mean(scores))

    if avg > 0.20:
        label = "STRONG_BULL"
    elif avg < -0.20:
        label = "STRONG_BEAR"
    else:
        label = "NEUTRAL"

    weight = float(np.clip(avg / 0.5, -1.0, 1.0))
    return avg, label, scores, weight


def compute_price_sentiment_series(df: pd.DataFrame) -> tuple:
    """
    Derive a per-bar, time-varying sentiment proxy from price and volume data.

    Components (both normalised via 60-day rolling z-score then clipped to [-1,+1]):
        - 5-day price ROC  — captures short-term momentum (60% weight)
        - Volume-direction — volume excess weighted by price direction:
              positive when large volume occurs on up-days (accumulation signal)
              negative when large volume occurs on down-days (distribution signal)
              (40% weight)

    Combined score ∈ [-1, +1]:
        > +0.20  → STRONG_BULL
        < -0.20  → STRONG_BEAR
        else     → NEUTRAL

    Returns:
        sentiment_labels  : pd.Series[str]   per-bar label
        sentiment_weights : pd.Series[float] per-bar weight ∈ [-1, +1]
    """
    # ── Momentum component ────────────────────────────────────────────────────
    roc5     = df["Close"].pct_change(5).fillna(0)
    roc5_mu  = roc5.rolling(60, min_periods=20).mean()
    roc5_std = roc5.rolling(60, min_periods=20).std().replace(0, np.nan)
    roc5_norm = ((roc5 - roc5_mu) / roc5_std).fillna(0).clip(-2, 2) / 2   # → [-1,+1]

    # ── Volume-direction component ─────────────────────────────────────────────
    price_dir = np.sign(df["Close"].pct_change().fillna(0))
    if "volume_ratio" in df.columns:
        vol_ratio = df["volume_ratio"].fillna(1.0)
    else:
        vol_ma    = df["Volume"].rolling(20, min_periods=5).mean().replace(0, np.nan)
        vol_ratio = (df["Volume"] / vol_ma).fillna(1.0)

    vol_signal_raw = price_dir * (vol_ratio - 1.0)
    vol_mu         = vol_signal_raw.rolling(60, min_periods=20).mean()
    vol_std        = vol_signal_raw.rolling(60, min_periods=20).std().replace(0, np.nan)
    vol_norm       = ((vol_signal_raw - vol_mu) / vol_std).fillna(0).clip(-2, 2) / 2

    # ── Combined score ─────────────────────────────────────────────────────────
    score   = (0.6 * roc5_norm + 0.4 * vol_norm).clip(-1.0, 1.0)

    labels  = pd.Series("NEUTRAL", index=df.index, dtype=object)
    labels[score >  0.20] = "STRONG_BULL"
    labels[score < -0.20] = "STRONG_BEAR"

    return labels, score.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — MARKET REGIME
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_COLORS = {
    "BULL_LOW_VOL":  "#7fff7f",   # green
    "BULL_HIGH_VOL": "#ffd166",   # amber
    "BEAR_LOW_VOL":  "#ff9f7f",   # orange
    "BEAR_HIGH_VOL": "#ff6b6b",   # red
}

REGIME_COLORS = _REGIME_COLORS  # re-exported for dashboard


def classify_market_regime(df: pd.DataFrame, percentile_window: int = 252) -> pd.DataFrame:
    """
    4-way market regime classification combining trend × volatility.

    Regimes:
        BULL_LOW_VOL   — Trending up, calm market (best for momentum / full size)
        BULL_HIGH_VOL  — Trending up, choppy (standard size)
        BEAR_LOW_VOL   — Trending down, quiet (reduced size)
        BEAR_HIGH_VOL  — Trending down, volatile (minimal size)

    Volatility threshold: ADAPTIVE — 80th percentile of trailing
    `percentile_window`-day annualised vol distribution.

    Rationale: A fixed 20% threshold (US-VIX baseline) misclassifies most
    Indian large-cap sessions as HIGH_VOL (RELIANCE.NS typical vol: 25–35%).
    Using the 80th percentile makes HIGH_VOL a *relative* concept anchored
    to the asset's own volatility history.
    """
    df = df.copy()
    if "ma200" not in df.columns:
        df["ma200"] = df["Close"].rolling(200).mean()

    df["_ann_vol"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    # Adaptive 80th-percentile vol threshold
    expanding_p80 = df["_ann_vol"].expanding(60).quantile(0.80)
    rolling_p80   = df["_ann_vol"].rolling(percentile_window, min_periods=60).quantile(0.80)
    df["_vol_p80"] = rolling_p80.fillna(expanding_p80).fillna(0.30)

    is_bull     = df["Close"] > df["ma200"]
    is_high_vol = df["_ann_vol"] > df["_vol_p80"]

    conditions = [
        is_bull  & ~is_high_vol,
        is_bull  &  is_high_vol,
        ~is_bull & ~is_high_vol,
        ~is_bull &  is_high_vol,
    ]
    choices = ["BULL_LOW_VOL", "BULL_HIGH_VOL", "BEAR_LOW_VOL", "BEAR_HIGH_VOL"]
    df["regime"]  = np.select(conditions, choices, default="BULL_LOW_VOL")
    df["is_bull"] = is_bull.astype(int)
    df["ann_vol"] = df["_ann_vol"]
    df.drop(columns=["_ann_vol", "_vol_p80"], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — ORCHESTRATOR  (confidence-weighted signal fusion)
# ─────────────────────────────────────────────────────────────────────────────

def combine_signals(
    ml_signal: pd.Series,
    ml_proba: pd.Series,
    sentiment_label: str,           # scalar — used as fallback if series not supplied
    sentiment_weight: float,        # scalar — used as fallback if series not supplied
    rsi: pd.Series,
    adx: pd.Series,
    regime: pd.Series,
    volume_ratio: pd.Series = None,          # NEW: volume confirmation gate
    sentiment_labels: pd.Series = None,      # NEW: time-varying labels (overrides scalar)
    sentiment_weights: pd.Series = None,     # NEW: time-varying weights (overrides scalar)
    apply_ml_training_mask: bool = True,
    train_idx: int = 0,
    buy_threshold: float = 0.60,   # higher than ML-Only's 0.55 — more selective entries
    sell_threshold: float = 0.45,
) -> tuple:
    """
    Confidence-weighted orchestrator fusing ML probability, time-varying sentiment,
    regime classification, volume, RSI and ADX.

    Design principles vs ML-Only:
        1. More selective entries (buy_threshold=0.60 > ML's 0.55).
        2. Volume confirmation on BUY: volume_ratio >= 0.9 (near-avg or better).
        3. Time-varying sentiment gate: STRONG_BEAR bars suppress BUY entries.
        4. Regime-based position sizing (agent sizes UP on best setups):
               BULL_LOW_VOL  → 1.3× (conviction trade — full allocation + boost)
               BULL_HIGH_VOL → 1.0× (standard size — same as ML baseline)
               BEAR_LOW_VOL  → 0.7× (below-average — reduce exposure)
               BEAR_HIGH_VOL → 0.5× (minimal — very defensive)
        5. Regime-aware SELL thresholds (benefit of the doubt in uptrends):
               BULL_LOW_VOL  → exit only if ml_prob < 0.36 (strong conviction needed)
               BULL_HIGH_VOL → exit if ml_prob < 0.42
               BEAR_LOW_VOL  → exit if ml_prob < 0.45 (standard)
               BEAR_HIGH_VOL → exit quickly if ml_prob < 0.48
        6. Sentiment boosts SELL threshold by 0.03 when STRONG_BEAR.
        7. Sentiment modulates position size continuously via sentiment_weight.
        8. RSI guard on SELL: in BULL regimes, don't exit into oversold (RSI < 30);
           in BEAR regimes, don't exit into extreme oversold (RSI < 25).

    Returns:
        signal         : pd.Series  (1=BUY, 0=SELL, NaN=HOLD)
        position_sizes : pd.Series  (size modifier × engine.position_size)
        agent_log      : pd.DataFrame — per-bar agent decisions (test set only)
    """
    signal         = pd.Series(np.nan, index=ml_signal.index)
    position_sizes = pd.Series(0.0,    index=ml_signal.index)

    _REGIME_SIZE = {
        "BULL_LOW_VOL":  1.3,   # agent sizes UP in the best regime
        "BULL_HIGH_VOL": 1.0,   # same as ML baseline
        "BEAR_LOW_VOL":  0.7,   # defensive
        "BEAR_HIGH_VOL": 0.5,   # very defensive
    }

    _REGIME_SELL_THR = {
        "BULL_LOW_VOL":  0.36,  # need very strong signal to exit uptrend
        "BULL_HIGH_VOL": 0.42,  # moderate conviction to exit
        "BEAR_LOW_VOL":  0.45,  # standard threshold
        "BEAR_HIGH_VOL": 0.48,  # exit quickly in dangerous regime
    }

    log_rows = []

    for i in range(len(signal)):
        if apply_ml_training_mask and i < train_idx:
            continue

        ml_prob = ml_proba.iloc[i] if not pd.isna(ml_proba.iloc[i]) else 0.50
        ml_sig  = ml_signal.iloc[i]
        rsi_val = rsi.iloc[i]
        adx_val = adx.iloc[i]
        reg     = regime.iloc[i] if i < len(regime) else "BULL_LOW_VOL"
        size    = _REGIME_SIZE.get(reg, 1.0)
        sell_thr = _REGIME_SELL_THR.get(reg, sell_threshold)

        # ── Volume confirmation ───────────────────────────────────────────────
        if volume_ratio is not None and i < len(volume_ratio):
            vol_ratio_val = float(volume_ratio.iloc[i])
            if pd.isna(vol_ratio_val):
                vol_ratio_val = 1.0
        else:
            vol_ratio_val = 1.0

        # ── Time-varying sentiment (per-bar series preferred over scalar) ─────
        if sentiment_labels is not None and i < len(sentiment_labels):
            bar_label  = sentiment_labels.iloc[i]
            bar_weight = float(sentiment_weights.iloc[i]) if sentiment_weights is not None else 0.0
            if pd.isna(bar_weight):
                bar_weight = 0.0
        else:
            bar_label  = sentiment_label
            bar_weight = sentiment_weight

        sentiment_blocks_buy  = bar_label == "STRONG_BEAR"
        sentiment_boosts_sell = bar_label == "STRONG_BEAR"
        # Sentiment modulates position size: bearish reduces, bullish boosts
        sentiment_size_mult = float(np.clip(0.8 + 0.4 * bar_weight, 0.5, 1.2))
        effective_size      = size * sentiment_size_mult

        # ── RSI guard for SELL — don't exit into oversold conditions ──────────
        # In uptrend (BULL), an oversold RSI is more likely a dip than a reversal
        rsi_oversold_thr   = 30 if "BULL" in reg else 25
        rsi_allows_sell    = rsi_val > rsi_oversold_thr

        final_signal = np.nan
        reason       = "NO_SIGNAL"

        # ── BUY ──────────────────────────────────────────────────────────────
        if (
            not pd.isna(ml_sig)
            and int(ml_sig) == 1
            and ml_prob > buy_threshold          # more selective than ML (0.60 vs 0.55)
            and not sentiment_blocks_buy         # time-varying STRONG_BEAR = no entry
            and vol_ratio_val >= 0.9             # at least near-average volume
            and rsi_val < 72                     # not heavily overbought
            and adx_val > 18                     # confirmed trend strength
        ):
            final_signal = 1
            reason = (
                f"ML={ml_prob:.2f}|RSI={rsi_val:.0f}|ADX={adx_val:.0f}"
                f"|VR={vol_ratio_val:.2f}|{reg}|sent={bar_weight:+.2f}"
            )

        # ── SELL — regime-aware thresholds ───────────────────────────────────
        elif (
            not pd.isna(ml_sig)
            and int(ml_sig) == 0
            and ml_prob < (sell_thr + (0.03 if sentiment_boosts_sell else 0.0))
            and rsi_allows_sell                  # don't exit into oversold
        ):
            final_signal = 0
            reason = (
                f"ML={ml_prob:.2f}|RSI={rsi_val:.0f}|SELL"
                f"|{reg}|sell_thr={sell_thr:.2f}|bear_sent={sentiment_boosts_sell}"
            )

        signal.iloc[i]         = final_signal
        position_sizes.iloc[i] = effective_size if not pd.isna(final_signal) else 0.0

        if pd.isna(ml_sig):
            ml_vote_str = "HOLD"
        elif int(ml_sig) == 1:
            ml_vote_str = "BUY"
        else:
            ml_vote_str = "SELL"

        log_rows.append({
            "date":           ml_signal.index[i],
            "ml_prob":        round(ml_prob, 3),
            "ml_vote":        ml_vote_str,
            "sentiment":      bar_label,
            "regime":         reg,
            "regime_size":    round(effective_size, 2),
            "rsi":            round(rsi_val, 1),
            "adx":            round(adx_val, 1),
            "vol_ratio":      round(vol_ratio_val, 2),
            "final_decision": "BUY"  if final_signal == 1
                              else "SELL" if final_signal == 0
                              else "HOLD",
            "reason":         reason,
        })

    agent_log = pd.DataFrame(log_rows).set_index("date") if log_rows else pd.DataFrame()

    return signal, position_sizes, agent_log
