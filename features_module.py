"""
Feature engineering module — 38-feature comprehensive set.
New (beyond original 26): Williams %R, CCI, CMF, ROC-4, TSI,
Keltner position, vol-ratio (short/long), gap, intraday range,
consecutive-day streak, price Z-score, close vs VWAP-proxy.
"""
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVE INDICATORS  (original)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    high_diff = high.diff();  low_diff = -low.diff()
    plus_dm  = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0),  low_diff,  0.0), index=df.index)
    tr = pd.Series(np.maximum(high - low, np.maximum(
        np.abs(high - close.shift()), np.abs(low - close.shift()))), index=df.index)
    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = (plus_dm.ewm(span=period, adjust=False).mean()  / atr * 100).fillna(0)
    minus_di = (minus_dm.ewm(span=period, adjust=False).mean() / atr * 100).fillna(0)
    di_sum   = plus_di + minus_di
    dx  = pd.Series(np.where(di_sum > 0, np.abs(plus_di - minus_di) / di_sum * 100, 0.0), index=df.index)
    return dx.ewm(span=period, adjust=False).mean()


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + std * std_dev, sma - std * std_dev


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = np.maximum(df["High"] - df["Low"], np.maximum(
        np.abs(df["High"] - df["Close"].shift()),
        np.abs(df["Low"]  - df["Close"].shift())))
    return pd.Series(tr, index=df.index).rolling(period).mean()


def compute_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["Close"].pct_change().rolling(period).std() * np.sqrt(252)


def compute_volume_features(df: pd.DataFrame, period: int = 20) -> tuple:
    volume_ma    = df["Volume"].rolling(period).mean()
    volume_ratio = df["Volume"] / volume_ma
    return volume_ma, volume_ratio


def compute_price_features(df: pd.DataFrame) -> tuple:
    price_change   = df["Close"].pct_change()
    price_momentum = df["Close"] / df["Close"].shift(5) - 1
    return price_change, price_momentum


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS FROM v2  (OBV, Stochastic, MACD Histogram, BB Squeeze, AutoCorr)
# ─────────────────────────────────────────────────────────────────────────────

def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff().fillna(0))
    obv_raw   = (direction * df["Volume"]).cumsum()
    obv_ma    = obv_raw.rolling(20).mean().replace(0, np.nan)
    return (obv_raw / obv_ma).fillna(1.0)


def compute_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple:
    lo = df["Low"].rolling(k).min();  hi = df["High"].rolling(k).max()
    rng = (hi - lo).replace(0, np.nan)
    pct_k = (df["Close"] - lo) / rng * 100
    return pct_k.fillna(50.0), pct_k.rolling(d).mean().fillna(50.0)


def compute_macd_histogram(df: pd.DataFrame) -> pd.Series:
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    return macd - macd.ewm(span=9, adjust=False).mean()


def compute_bb_squeeze(series: pd.Series, period: int = 20, lookback: int = 50) -> pd.Series:
    _, upper, lower = compute_bollinger_bands(series, period)
    bb_width   = (upper - lower) / series.rolling(period).mean()
    mean_width = bb_width.rolling(lookback).mean()
    return (bb_width < mean_width).astype(float).fillna(0.0)


def compute_return_autocorr(series: pd.Series, lag: int = 5, window: int = 20) -> pd.Series:
    returns = series.pct_change()
    def _ac(x):
        return float(pd.Series(x).autocorr(lag=lag)) if len(x) >= lag + 2 else 0.0
    return returns.rolling(window + lag).apply(_ac, raw=True).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# NEW INDICATORS  (v3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — ranges -100 to 0; above -20 = overbought, below -80 = oversold."""
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    rng = (hi - lo).replace(0, np.nan)
    return ((hi - df["Close"]) / rng * -100).fillna(-50.0)


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index — measures deviation from a 'typical price' SMA."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((tp - sma) / (0.015 * mad.replace(0, np.nan))).fillna(0.0)


def compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — sustained positive = accumulation, negative = distribution."""
    hi, lo, cl, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    rng = (hi - lo).replace(0, np.nan)
    mfm = ((cl - lo) - (hi - cl)) / rng          # money-flow multiplier (-1 to +1)
    mfv = mfm * vol
    return (mfv.rolling(period).sum() / vol.rolling(period).sum()).fillna(0.0)


def compute_roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change over `period` days — % return."""
    return (series / series.shift(period) - 1).fillna(0.0)


def compute_tsi(series: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """
    True Strength Index — double-smoothed momentum oscillator.
    Ranges roughly -100 to +100; crossover with signal line triggers trades.
    """
    m  = series.diff()
    m2 = m.abs()
    ds   = m.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    ds2  = m2.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    return (ds / ds2.replace(0, np.nan) * 100).fillna(0.0)


def compute_keltner_position(df: pd.DataFrame, ema_period: int = 20,
                              atr_mult: float = 2.0) -> pd.Series:
    """
    Position of Close within Keltner Channel (0 = lower band, 1 = upper band).
    Similar idea to BB position but uses ATR-based bands → less noise.
    """
    mid   = df["Close"].ewm(span=ema_period, adjust=False).mean()
    atr   = compute_atr(df, ema_period)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    rng   = (upper - lower).replace(0, np.nan)
    return ((df["Close"] - lower) / rng).clip(0, 1).fillna(0.5)


def compute_vol_ratio(df: pd.DataFrame,
                      short: int = 5, long: int = 21) -> pd.Series:
    """
    Short-term volatility / Long-term volatility ratio.
    >1 = volatility expanding (caution), <1 = volatility compressing (breakout brewing).
    """
    short_vol = df["Close"].pct_change().rolling(short).std()
    long_vol  = df["Close"].pct_change().rolling(long).std()
    return (short_vol / long_vol.replace(0, np.nan)).fillna(1.0)


def compute_gap(df: pd.DataFrame) -> pd.Series:
    """Overnight gap: (Open - prev_Close) / prev_Close. Large gaps = significant events."""
    return ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)).fillna(0.0)


def compute_intraday_range(df: pd.DataFrame) -> pd.Series:
    """(High - Low) / Close — normalised daily range; proxy for intraday volatility."""
    return ((df["High"] - df["Low"]) / df["Close"]).fillna(0.0)


def compute_consecutive_days(series: pd.Series) -> pd.Series:
    """
    Signed streak of consecutive up/down days.
    +3 = up 3 days in a row, -2 = down 2 days in a row.
    Uses a simple iterative approach wrapped in pandas apply-like logic.
    """
    delta   = np.sign(series.diff().fillna(0))
    streak  = np.zeros(len(series))
    for i in range(1, len(streak)):
        if delta.iloc[i] == 0:
            streak[i] = 0
        elif delta.iloc[i] > 0:
            streak[i] = max(0, streak[i - 1]) + 1
        else:
            streak[i] = min(0, streak[i - 1]) - 1
    return pd.Series(streak, index=series.index)


def compute_price_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    """Z-score of closing price over rolling window — identifies mean-reversion setups."""
    mu  = series.rolling(window).mean()
    sig = series.rolling(window).std().replace(0, np.nan)
    return ((series - mu) / sig).fillna(0.0)


def compute_vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Approximate rolling VWAP deviation.
    (Close - VWAP) / VWAP  where VWAP = sum(TP*Vol)/sum(Vol) over rolling window.
    """
    tp   = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (tp * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()
    return ((df["Close"] - vwap) / vwap.replace(0, np.nan)).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY-SPECIFIC SIGNAL HELPERS  (for rule-based strategies)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_reversion_signal(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based Mean Reversion signal.
    BUY  (1): RSI < 35  AND  price below lower BB  AND  CMF > -0.15 (not extreme distribution)
    SELL (0): RSI > 65  AND  price above upper BB
    HOLD(NaN): otherwise
    """
    _, bb_upper, bb_lower = compute_bollinger_bands(df["Close"])
    rsi = compute_rsi(df["Close"])
    cmf = compute_cmf(df)

    buy  = (rsi < 35)  & (df["Close"] < bb_lower) & (cmf > -0.15)
    sell = (rsi > 65)  & (df["Close"] > bb_upper)

    sig = pd.Series(np.nan, index=df.index)
    sig[buy]  = 1
    sig[sell] = 0
    return sig


def compute_momentum_signal(df: pd.DataFrame) -> pd.Series:
    """
    Pure Momentum / Trend-Following signal.
    BUY  (1): MA20 > MA50 AND MA50 > MA200 AND ADX > 22 AND Close > previous Close
    SELL (0): MA20 < MA50 OR  MA50 < MA200 (trend breakdown)
    HOLD(NaN): otherwise
    """
    ma20  = df["Close"].rolling(20).mean()
    ma50  = df["Close"].rolling(50).mean()
    ma200 = df["Close"].rolling(200).mean()
    adx   = compute_adx(df)

    aligned_up   = (ma20 > ma50) & (ma50 > ma200)
    aligned_down = (ma20 < ma50) | (ma50 < ma200)
    price_up = df["Close"] > df["Close"].shift(1)

    sig = pd.Series(np.nan, index=df.index)
    sig[aligned_up   & (adx > 22) & price_up]   = 1
    sig[aligned_down & (adx > 20)]               = 0
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE BUILDER  (38 features)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 38-feature set for ML model training.
    Layout:
        Original core (RSI, MACD, MA, ADX, ATR, Vol, BB, Volume, Price) — 12
        v2 additions (OBV, Stoch, MACD Hist, BB Squeeze, AutoCorr)    —  6
        v3 additions (WR, CCI, CMF, ROC×4, TSI, Keltner, VolRatio,
                      Gap, IntraDayRange, ConsecDays, PriceZScore, VWAP Dev) — 14
        Lagged (rsi, adx, vol, macd, stoch_k) × lag5                  —  5
        Regime flags (is_uptrend, is_bull_regime)                      —  2
        Raw volume                                                       —  1
    Total: 38 (excluding Volume which is also a features col)
    """
    df = df.copy()

    # ── Momentum / oscillators ────────────────────────────────────────────────
    df["rsi"]      = compute_rsi(df["Close"])
    ema12          = df["Close"].ewm(span=12, adjust=False).mean()
    ema26          = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]     = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = compute_macd_histogram(df)
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df)
    df["williams_r"]  = compute_williams_r(df)      # NEW
    df["cci"]         = compute_cci(df)              # NEW
    df["tsi"]         = compute_tsi(df["Close"])     # NEW

    # ── Rate of Change (4 timeframes) ─────────────────────────────────────────
    df["roc1"]  = compute_roc(df["Close"],  1)       # NEW
    df["roc5"]  = compute_roc(df["Close"],  5)       # NEW
    df["roc10"] = compute_roc(df["Close"], 10)       # NEW
    df["roc20"] = compute_roc(df["Close"], 20)       # NEW

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ma50"]  = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()
    df["adx"]   = compute_adx(df)

    # ── Volatility / bands ────────────────────────────────────────────────────
    df["atr"]          = compute_atr(df)
    df["volatility"]   = compute_volatility(df)
    df["vol_ratio"]    = compute_vol_ratio(df)       # NEW (short/long vol ratio)
    df["bb_squeeze"]   = compute_bb_squeeze(df["Close"])

    _, df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["Close"])
    bb_rng = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_position"]  = ((df["Close"] - df["bb_lower"]) / bb_rng).fillna(0.5)
    df["keltner_pos"]  = compute_keltner_position(df)   # NEW

    # ── Volume & money flow ───────────────────────────────────────────────────
    df["volume_ma"], df["volume_ratio"] = compute_volume_features(df)
    df["obv"]  = compute_obv(df)
    df["cmf"]  = compute_cmf(df)                     # NEW

    # ── Price structure ───────────────────────────────────────────────────────
    df["price_change"],   df["price_momentum"] = compute_price_features(df)
    df["return_autocorr"] = compute_return_autocorr(df["Close"])
    df["gap"]            = compute_gap(df)           # NEW
    df["intraday_range"] = compute_intraday_range(df) # NEW
    df["consec_days"]    = compute_consecutive_days(df["Close"])  # NEW
    df["price_zscore"]   = compute_price_zscore(df["Close"])      # NEW
    df["vwap_dev"]       = compute_vwap_deviation(df)             # NEW

    # ── Lagged features (5-day look-back) ─────────────────────────────────────
    df["rsi_lag5"]        = df["rsi"].shift(5)
    df["adx_lag5"]        = df["adx"].shift(5)
    df["volatility_lag5"] = df["volatility"].shift(5)
    df["macd_lag5"]       = df["macd"].shift(5)
    df["stoch_k_lag5"]    = df["stoch_k"].shift(5)   # NEW lag

    # ── Regime flags ──────────────────────────────────────────────────────────
    df["is_uptrend"]    = (df["Close"] > df["ma50"]).astype(int)
    df["is_bull_regime"] = (df["Close"] > df["ma200"]).astype(int)

    return df
