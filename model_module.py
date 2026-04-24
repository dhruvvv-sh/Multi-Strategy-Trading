"""
Machine learning model module.
Ensemble: RandomForest + HistGradientBoosting (handles NaN natively, faster).
Target: 3-day forward return > 0  (smoother signal, higher OOS accuracy vs next-day).
Walk-forward cross-validation: 5 folds, expanding window.
"""
import pandas as pd
import numpy as np
import hashlib
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


MODEL_CACHE_VERSION = "v1"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_fill(X: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then zero-fill — avoids future-leakage from bfill."""
    return X.ffill().fillna(0)


def _build_target(df: pd.DataFrame, forward_days: int = 3) -> pd.Series:
    """
    Binary target: 1 if Close n days forward > today's Close.
    Using 3-day forward return reduces noise vs next-day and yields
    more stable out-of-sample accuracy.
    """
    return (df["Close"].shift(-forward_days) > df["Close"]).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_ensemble() -> VotingClassifier:
    """
    Soft-voting ensemble:
      • Random Forest        — strong, diverse, low variance
      • HistGradientBoosting — native NaN support, excellent on tabular data
    Weights 40:60 favouring HGB which typically dominates on financial data.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=15,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,
        learning_rate=0.03,
        min_samples_leaf=20,
        l2_regularization=0.2,
        max_bins=128,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)],
        voting="soft",
        weights=[0.4, 0.6],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_ml_model(df: pd.DataFrame, feature_cols: list, train_idx: int,
                   forward_days: int = 3):
    """Train ensemble on training set. Returns (model, train_accuracy)."""
    X_train = _safe_fill(df[feature_cols].iloc[:train_idx])
    y_train = _build_target(df, forward_days).iloc[:train_idx]

    model = build_ensemble()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    return model, float(accuracy_score(y_train, train_pred))


def build_model_cache_key(df: pd.DataFrame, feature_cols: list,
                          train_idx: int, forward_days: int = 3) -> str:
    """Build a deterministic cache key from data, feature schema, and train config."""
    h = hashlib.sha256()

    h.update(MODEL_CACHE_VERSION.encode("utf-8"))
    h.update(str(len(df)).encode("utf-8"))
    h.update(str(train_idx).encode("utf-8"))
    h.update(str(forward_days).encode("utf-8"))
    h.update("|".join(feature_cols).encode("utf-8"))

    if len(df) > 0:
        h.update(str(df.index[0]).encode("utf-8"))
        h.update(str(df.index[-1]).encode("utf-8"))

    # Hash only core market columns used to derive all engineered features.
    # This keeps key generation fast while still invalidating correctly.
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        vals = np.nan_to_num(df[col].to_numpy(dtype=np.float64), nan=0.0)
        h.update(vals.tobytes())

    return h.hexdigest()


def train_or_load_ml_model(df: pd.DataFrame, feature_cols: list, train_idx: int,
                           forward_days: int = 3,
                           cache_path: Path = None) -> tuple:
    """
    Load model from binary cache if possible, else train and persist.

    Returns:
        (model, train_accuracy, loaded_from_cache, cache_key)
    """
    cache_key = build_model_cache_key(df, feature_cols, train_idx, forward_days)
    cache_file = Path(cache_path) if cache_path is not None else None

    if cache_file is not None and cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("cache_key") == cache_key and payload.get("model") is not None:
                return (
                    payload["model"],
                    float(payload.get("train_accuracy", 0.0)),
                    True,
                    cache_key,
                )
        except Exception:
            # Corrupt/old cache should not break startup; we simply retrain.
            pass

    model, train_acc = train_ml_model(df, feature_cols, train_idx, forward_days)

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as f:
                pickle.dump(
                    {
                        "cache_key": cache_key,
                        "train_accuracy": float(train_acc),
                        "model": model,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            # Cache write failure should not stop execution.
            pass

    return model, float(train_acc), False, cache_key


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ml_model(model, df: pd.DataFrame, feature_cols: list, train_idx: int,
                      forward_days: int = 3) -> dict:
    """
    Evaluate on out-of-sample test set.
    BUG-06 FIX: Drop last `forward_days` rows — they have NaN targets because there
    is no future data to compute a 3-day forward return, and astype(int) silently
    converts them to 0, biasing test accuracy downward.
    """
    # Exclude last forward_days rows (unknowable future targets)
    safe_end = len(df) - forward_days
    X_test = _safe_fill(df[feature_cols].iloc[train_idx:safe_end])
    y_test = _build_target(df, forward_days).iloc[train_idx:safe_end]

    test_pred  = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    return {
        "test_accuracy":  float(accuracy_score(y_test, test_pred)),
        "precision":      float(precision_score(y_test, test_pred, zero_division=0)),
        "recall":         float(recall_score(y_test, test_pred, zero_division=0)),
        "f1_score":       float(f1_score(y_test, test_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, test_pred),
        "mean_buy_prob":  float(test_proba.mean()),
        "high_conf_pct":  float((test_proba > 0.55).mean()),
        "low_conf_pct":   float((test_proba < 0.45).mean()),
        "forward_days":   forward_days,
    }


def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """
    Feature importance from ensemble.
    HistGradientBoostingClassifier does not expose .feature_importances_ directly,
    so we use RF importances only (which are well-calibrated from the large forest).
    """
    rf_imp  = model.estimators_[0].feature_importances_
    return (
        pd.DataFrame({"Feature": feature_cols, "Importance": rf_imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION  (confidence-gated on 3-day prediction)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ml_signal(model, df: pd.DataFrame, feature_cols: list,
                       train_idx: int,
                       buy_threshold: float = 0.55,
                       sell_threshold: float = 0.45) -> tuple:
    """
    Generate ML signals on OOS data only using probability thresholds.
    Returns (signal series, proba series).
    """
    signal = pd.Series(np.nan, index=df.index)
    proba  = pd.Series(np.nan, index=df.index)

    X_test     = _safe_fill(df[feature_cols].iloc[train_idx:])
    proba_test = model.predict_proba(X_test)[:, 1]

    proba.iloc[train_idx:]  = proba_test
    signal.iloc[train_idx:] = np.where(
        proba_test > buy_threshold, 1,
        np.where(proba_test < sell_threshold, 0, np.nan)
    )
    return signal, proba


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_validate(df: pd.DataFrame, feature_cols: list,
                          n_folds: int = 5,
                          min_train_frac: float = 0.50,
                          forward_days: int = 3) -> dict:
    """
    5-fold anchored walk-forward CV.
    Each fold expands the training window by ~((1-min_train_frac)/n_folds) of total data.
    """
    n         = len(df)
    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = max(remaining // n_folds, 30)
    target    = _build_target(df, forward_days)

    fold_accuracies, fold_dates, fold_details = [], [], []

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end  = min(train_end + fold_size, n - forward_days - 1)

        if train_end >= test_end or (test_end - train_end) < 10:
            break

        X_tr = _safe_fill(df[feature_cols].iloc[:train_end])
        y_tr = target.iloc[:train_end]
        X_te = _safe_fill(df[feature_cols].iloc[train_end:test_end])
        y_te = target.iloc[train_end:test_end]

        m = build_ensemble()
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)

        acc  = float(accuracy_score(y_te, preds))
        fold_accuracies.append(acc)
        fold_dates.append({
            "train_end":  df.index[train_end - 1],
            "test_start": df.index[train_end],
            "test_end":   df.index[test_end - 1],
            "n_train":    train_end,
            "n_test":     test_end - train_end,
        })
        fold_details.append({
            "fold":      fold + 1,
            "accuracy":  acc,
            "precision": float(precision_score(y_te, preds, zero_division=0)),
            "recall":    float(recall_score(y_te, preds, zero_division=0)),
            "f1":        float(f1_score(y_te, preds, zero_division=0)),
        })

    return {
        "fold_accuracies": fold_accuracies,
        "fold_dates":      fold_dates,
        "fold_details":    fold_details,
        "mean_accuracy":   float(np.mean(fold_accuracies)) if fold_accuracies else 0.0,
        "std_accuracy":    float(np.std(fold_accuracies))  if fold_accuracies else 0.0,
    }
