"""
Data acquisition and preprocessing module.
Uses 5-year window for more training signal.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_file_path(ticker_sym: str, start: str, end: str) -> Path:
    safe_ticker = ticker_sym.replace(".", "_")
    return CACHE_DIR / f"ohlcv_{safe_ticker}_{start}_{end}.csv"


def _normalise_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[OHLCV_COLS].copy()
    raw.dropna(inplace=True)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw.astype(float)


def _load_cached_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if not set(OHLCV_COLS).issubset(cached.columns):
            return pd.DataFrame()
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        cached = cached[OHLCV_COLS].copy()
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        return cached.astype(float)
    except Exception:
        return pd.DataFrame()


def _save_cached_ohlcv(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except Exception:
        # Cache write failures should not block strategy execution.
        pass


def fetch_data(ticker_sym: str = "RELIANCE.NS",
               start: str = "2021-04-01",
               end: str = "2026-04-20",
               use_cache: bool = True,
               force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance with deterministic local caching.

    Behavior:
      1) Load cached OHLCV if available (unless force_refresh=True)
      2) Else fetch from Yahoo and cache the snapshot
      3) On fetch failure, fall back to cache if present
      4) Final fallback: deterministic synthetic data
    """
    cache_file = _cache_file_path(ticker_sym, start, end)

    if use_cache and not force_refresh:
        cached = _load_cached_ohlcv(cache_file)
        if not cached.empty:
            return cached

    try:
        tk  = yf.Ticker(ticker_sym)
        raw = tk.history(start=start, end=end, interval="1d")
        if raw.empty:
            raise ValueError("Empty response")
        norm = _normalise_ohlcv(raw)
        if norm.empty:
            raise ValueError("No usable OHLCV rows after normalization")
        if use_cache:
            _save_cached_ohlcv(cache_file, norm)
        return norm
    except Exception:
        if use_cache:
            cached = _load_cached_ohlcv(cache_file)
            if not cached.empty:
                return cached
        fallback = generate_synthetic_data(end_date=end)
        if use_cache:
            _save_cached_ohlcv(cache_file, fallback)
        return fallback


def generate_synthetic_data(n: int = 1260, end_date: str = "2026-04-20") -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for reproducible fallback runs."""
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp(end_date).normalize(), periods=n)
    rets  = np.random.normal(0.0004, 0.013, n)
    close = 2450.0 * np.cumprod(1 + rets)
    rng   = np.abs(np.random.normal(0.008, 0.004, n))
    opens = close * (1 + np.random.uniform(-0.005, 0.005, n))
    return pd.DataFrame({
        "Open":   opens,
        "High":   np.maximum(close, opens) * (1 + rng),
        "Low":    np.minimum(close, opens) * (1 - rng),
        "Close":  close,
        "Volume": np.random.randint(3_000_000, 15_000_000, n).astype(float),
    }, index=dates)
