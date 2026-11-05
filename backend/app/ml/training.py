"""Training pipelines, leak-free chronological splitting, and model factories."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("EquityLens5.Training")


class MajorityBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Zero-rule baseline: always predicts majority class observed in training data."""

    def __init__(self) -> None:
        self.majority_class_: float = 1.0
        self.positive_ratio_: float = 0.5
        self.classes_: np.ndarray = np.array([0.0, 1.0])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> MajorityBaselineClassifier:
        y_arr = np.asarray(y, dtype=float)
        self.positive_ratio_ = float(np.mean(y_arr == 1.0))
        self.majority_class_ = 1.0 if self.positive_ratio_ >= 0.5 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        p1 = self.positive_ratio_
        p0 = 1.0 - p1
        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = p0
        probs[:, 1] = p1
        return probs


class MomentumBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Financial heuristic baseline: predicts 1 if return_5d > 0 else 0."""

    def __init__(self, return_col: str = "return_5d") -> None:
        self.return_col = return_col
        self.classes_: np.ndarray = np.array([0.0, 1.0])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> MomentumBaselineClassifier:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ret = X[self.return_col].astype(float).values
        return (ret > 0.0).astype(float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ret = X[self.return_col].astype(float).values
        # Simple logistic mapping around 0 for soft probabilities
        prob_up = 1.0 / (1.0 + np.exp(-np.clip(ret * 20.0, -10.0, 10.0)))
        prob_down = 1.0 - prob_up
        return np.column_stack([prob_down, prob_up])


def build_chronological_splits(
    df_5d: pd.DataFrame,
    df_all: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> dict[str, Any]:
    """Partition dataset into chronological Train, Validation, and Test subsets with purging.

    - Splits are strictly date-based on unique panel session dates.
    - An observation at session_date t has label_end_date t+5.
    - If label_end_date crosses the split threshold, it is PURGED to prevent
      look-ahead label leakage.
    """
    # 1. Map label_end_date for each (ticker, session_date) pair using full chronological series
    label_end_map: dict[tuple[str, str], str] = {}
    for _sym, grp in df_all.groupby("ticker"):
        grp_sorted = grp.sort_values("session_date").reset_index(drop=True)
        grp_sorted["label_end_date"] = grp_sorted["session_date"].shift(-5)
        for _, row in grp_sorted.iterrows():
            if pd.notna(row["label_end_date"]):
                label_end_map[(str(row["ticker"]), str(row["session_date"]))] = str(
                    row["label_end_date"]
                )

    work_df = df_5d.copy()
    work_df["session_date"] = work_df["session_date"].astype(str)
    work_df["label_end_date"] = [
        label_end_map.get((r["ticker"], r["session_date"])) for _, r in work_df.iterrows()
    ]

    # 2. Determine global chronological boundaries
    unique_dates = sorted(work_df["session_date"].unique())
    n_dates = len(unique_dates)
    idx_train = int(n_dates * train_ratio)
    idx_val = int(n_dates * (train_ratio + val_ratio))

    train_end = unique_dates[idx_train - 1]
    val_end = unique_dates[idx_val - 1]
    test_end = unique_dates[-1]

    # Full available horizon boundary for test set
    max_avail_date = str(df_all["session_date"].max())

    # 3. Apply split definitions and purging
    # Train: session_date <= train_end
    # Purge if label_end_date > train_end
    is_train_cand = work_df["session_date"] <= train_end
    train_kept = work_df[is_train_cand & (work_df["label_end_date"] <= train_end)].copy()
    train_purged = work_df[is_train_cand & (work_df["label_end_date"] > train_end)].copy()

    # Validation: session_date > train_end and session_date <= val_end
    # Purge if label_end_date > val_end
    is_val_cand = (work_df["session_date"] > train_end) & (work_df["session_date"] <= val_end)
    val_kept = work_df[is_val_cand & (work_df["label_end_date"] <= val_end)].copy()
    val_purged = work_df[is_val_cand & (work_df["label_end_date"] > val_end)].copy()

    # Test: session_date > val_end and session_date <= test_end
    # Purge if label_end_date > max_avail_date
    is_test_cand = (work_df["session_date"] > val_end) & (work_df["session_date"] <= test_end)
    test_kept = work_df[is_test_cand & (work_df["label_end_date"] <= max_avail_date)].copy()
    test_purged = work_df[is_test_cand & (work_df["label_end_date"] > max_avail_date)].copy()

    # 4. Summary counts per symbol
    symbol_breakdown: dict[str, dict[str, int]] = {}
    tickers = sorted(work_df["ticker"].unique())
    for sym in tickers:
        symbol_breakdown[sym] = {
            "train_kept": int(len(train_kept[train_kept["ticker"] == sym])),
            "train_purged": int(len(train_purged[train_purged["ticker"] == sym])),
            "val_kept": int(len(val_kept[val_kept["ticker"] == sym])),
            "val_purged": int(len(val_purged[val_purged["ticker"] == sym])),
            "test_kept": int(len(test_kept[test_kept["ticker"] == sym])),
            "test_purged": int(len(test_purged[test_purged["ticker"] == sym])),
        }

    return {
        "train_df": train_kept,
        "val_df": val_kept,
        "test_df": test_kept,
        "purged_train_df": train_purged,
        "purged_val_df": val_purged,
        "purged_test_df": test_purged,
        "boundaries": {
            "train_start": unique_dates[0],
            "train_end": train_end,
            "val_start": unique_dates[idx_train],
            "val_end": val_end,
            "test_start": unique_dates[idx_val],
            "test_end": test_end,
            "max_available_date": max_avail_date,
        },
        "counts": {
            "train_kept": len(train_kept),
            "train_purged": len(train_purged),
            "val_kept": len(val_kept),
            "val_purged": len(val_purged),
            "test_kept": len(test_kept),
            "test_purged": len(test_purged),
            "total_evaluated": len(train_kept) + len(val_kept) + len(test_kept),
            "total_purged": len(train_purged) + len(val_purged) + len(test_purged),
        },
        "symbol_breakdown": symbol_breakdown,
    }


def create_model_candidate(model_key: str) -> Any:
    """Instantiate a model pipeline or estimator for candidate key."""
    if model_key == "B0a_majority":
        return MajorityBaselineClassifier()
    if model_key == "B0b_momentum":
        return MomentumBaselineClassifier()
    if model_key == "M0_logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        C=1.0,
                    ),
                ),
            ]
        )
    if model_key == "M1_random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        )
    if model_key == "M1_hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            min_samples_leaf=20,
            random_state=42,
        )
    msg = f"Unknown model candidate key: {model_key}"
    raise ValueError(msg)


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    candidate_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Train all requested baseline and machine learning models on training partition."""
    keys = candidate_keys or [
        "B0a_majority",
        "B0b_momentum",
        "M0_logistic_regression",
        "M1_random_forest",
        "M1_hist_gradient_boosting",
    ]

    trained_models: dict[str, Any] = {}
    for k in keys:
        logger.info("Training candidate model: %s", k)
        model = create_model_candidate(k)
        model.fit(X_train, y_train)
        trained_models[k] = model

    return trained_models
