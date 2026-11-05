"""EquityLens-5 SEC Fundamental Model Comparison (M0 vs M1 vs B0a vs B0b) (EL5-060).

Implements:
1. Retrospective exploratory ablation on EL5-050 test set.
2. Leak-free expanding-window chronological cross-validation with label purging.
3. M1 pipeline with SimpleImputer(strategy='median') fitted purely on training data.
4. Scientific metrics computation (Balanced Accuracy, Accuracy, Precision, Recall, F1, ROC-AUC).
"""

from __future__ import annotations

import logging
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from backend.app.features import FEATURE_COLUMNS
from backend.app.ml.evaluation import evaluate_model, evaluate_per_symbol
from backend.app.ml.training import (
    MajorityBaselineClassifier,
    MomentumBaselineClassifier,
    build_chronological_splits,
)
from backend.app.sec_fundamentals import SEC_FEATURE_COLUMNS

logger = logging.getLogger("EquityLens5.SECExperiment")

# Combined feature list for M1 (23 technical + 8 SEC fundamental = 31 features)
M1_FEATURE_COLUMNS: Final[list[str]] = FEATURE_COLUMNS + SEC_FEATURE_COLUMNS


def create_m0_model() -> HistGradientBoostingClassifier:
    """Instantiate M0 price-only HistGradientBoosting model (same as EL5-050)."""
    return HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        min_samples_leaf=20,
        random_state=42,
    )


def create_m1_model() -> Pipeline:
    """Instantiate M1 price + fundamental pipeline.

    Includes SimpleImputer fitted strictly on training data to handle missing values
    without information leakage. Uses identical classifier hyperparameters as M0.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=3,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )


def build_expanding_window_folds(
    df_sec: pd.DataFrame,
    df_all: pd.DataFrame,
    n_folds: int = 3,
    train_start_ratio: float = 0.50,
    val_ratio: float = 0.15,
) -> list[dict[str, Any]]:
    """Generate chronological expanding-window folds with label purging.

    For each fold k:
    - Train window expands chronologically.
    - Test window follows immediately.
    - Observations whose 5-day return horizon crosses the split boundary are PURGED.
    """
    # 1. Map label_end_date for every (ticker, session_date) pair
    label_end_map: dict[tuple[str, str], str] = {}
    for _sym, grp in df_all.groupby("ticker"):
        grp_sorted = grp.sort_values("session_date").reset_index(drop=True)
        grp_sorted["label_end_date"] = grp_sorted["session_date"].shift(-5)
        for _, row in grp_sorted.iterrows():
            if pd.notna(row["label_end_date"]):
                label_end_map[(str(row["ticker"]), str(row["session_date"]))] = str(
                    row["label_end_date"]
                )

    work_df = df_sec.copy()
    work_df["session_date"] = work_df["session_date"].astype(str)
    work_df["label_end_date"] = [
        label_end_map.get((r["ticker"], r["session_date"])) for _, r in work_df.iterrows()
    ]

    unique_dates = sorted(work_df["session_date"].unique())
    n_dates = len(unique_dates)
    max_avail_date = str(df_all["session_date"].max())

    folds: list[dict[str, Any]] = []

    # Step size between expanding train boundaries
    # E.g. with 3 folds:
    # Fold 1: train 0..50%, test 50..65%
    # Fold 2: train 0..65%, test 65..80%
    # Fold 3: train 0..80%, test 80..95% (or 100%)
    step = (1.0 - train_start_ratio - val_ratio) / max(1, n_folds - 1)

    for i in range(n_folds):
        tr_ratio = train_start_ratio + i * step
        te_ratio = tr_ratio + val_ratio

        idx_train = int(n_dates * tr_ratio)
        idx_test = min(n_dates, int(n_dates * te_ratio))

        train_end = unique_dates[idx_train - 1]
        test_start = unique_dates[idx_train]
        test_end = unique_dates[idx_test - 1]

        # Purging rule:
        # Train: session_date <= train_end, label_end_date <= train_end
        is_train = work_df["session_date"] <= train_end
        train_kept = work_df[is_train & (work_df["label_end_date"] <= train_end)].copy()
        train_purged = work_df[is_train & (work_df["label_end_date"] > train_end)].copy()

        # Test: test_start <= session_date <= test_end
        is_test = (work_df["session_date"] >= test_start) & (work_df["session_date"] <= test_end)
        test_kept = work_df[is_test & (work_df["label_end_date"] <= max_avail_date)].copy()
        test_purged = work_df[is_test & (work_df["label_end_date"] > max_avail_date)].copy()

        folds.append(
            {
                "fold_index": i + 1,
                "train_start": unique_dates[0],
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_df": train_kept,
                "train_purged_df": train_purged,
                "test_df": test_kept,
                "test_purged_df": test_purged,
                "train_count": len(train_kept),
                "train_purged_count": len(train_purged),
                "test_count": len(test_kept),
                "test_purged_count": len(test_purged),
            }
        )

    return folds


def run_retrospective_ablation(
    df_sec: pd.DataFrame,
    df_all: pd.DataFrame,
) -> dict[str, Any]:
    """Execute retrospective exploratory ablation on the exact EL5-050 test set.

    Compares M0 (price-only), M1 (price + SEC fundamentals), B0a (Majority), and B0b (Momentum).
    """
    # Use exact EL5-050 split configuration (60% train, 20% val, 20% test)
    splits = build_chronological_splits(df_sec, df_all, train_ratio=0.60, val_ratio=0.20)
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    test_df = splits["test_df"]

    # In EL5-050, the winning architecture was retrained on Train + Validation combined
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    y_train = train_val_df["direction_5d"].values
    y_test = test_df["direction_5d"].values

    # 1. B0a: Majority Baseline
    b0a = MajorityBaselineClassifier()
    b0a.fit(train_val_df[FEATURE_COLUMNS], y_train)
    b0a_metrics = evaluate_model(b0a, test_df[FEATURE_COLUMNS], y_test)

    # 2. B0b: Momentum Baseline
    b0b = MomentumBaselineClassifier()
    b0b.fit(train_val_df[FEATURE_COLUMNS], y_train)
    b0b_metrics = evaluate_model(b0b, test_df[FEATURE_COLUMNS], y_test)

    # 3. M0: HistGradientBoosting on 23 Price features
    m0 = create_m0_model()
    m0.fit(train_val_df[FEATURE_COLUMNS], y_train)
    m0_metrics = evaluate_model(m0, test_df[FEATURE_COLUMNS], y_test)
    m0_by_sym = evaluate_per_symbol(m0, test_df, FEATURE_COLUMNS, "direction_5d")

    # 4. M1: HistGradientBoosting on 31 features (Price + SEC)
    m1 = create_m1_model()
    m1.fit(train_val_df[M1_FEATURE_COLUMNS], y_train)
    m1_metrics = evaluate_model(m1, test_df[M1_FEATURE_COLUMNS], y_test)
    m1_by_sym = evaluate_per_symbol(m1, test_df, M1_FEATURE_COLUMNS, "direction_5d")

    # Differences
    m1_m0_diff = round(m1_metrics["balanced_accuracy"] - m0_metrics["balanced_accuracy"], 4)
    m1_b0a_diff = round(m1_metrics["balanced_accuracy"] - b0a_metrics["balanced_accuracy"], 4)

    return {
        "split_counts": {
            "train_val_count": len(train_val_df),
            "test_count": len(test_df),
            "purged_count": splits["counts"]["total_purged"],
        },
        "boundaries": splits["boundaries"],
        "models": {
            "B0a_majority": b0a_metrics,
            "B0b_momentum": b0b_metrics,
            "M0_price_only": m0_metrics,
            "M1_price_sec": m1_metrics,
        },
        "differences": {
            "m1_minus_m0_balanced_acc": m1_m0_diff,
            "m1_minus_b0a_balanced_acc": m1_b0a_diff,
        },
        "m0_by_symbol": m0_by_sym,
        "m1_by_symbol": m1_by_sym,
        "trained_m1_model": m1,
    }


def run_cross_validation_experiment(
    df_sec: pd.DataFrame,
    df_all: pd.DataFrame,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Execute leak-free expanding-window chronological cross-validation."""
    folds = build_expanding_window_folds(df_sec, df_all, n_folds=n_folds)

    fold_results: list[dict[str, Any]] = []

    for f in folds:
        k = f["fold_index"]
        tr_df = f["train_df"]
        te_df = f["test_df"]

        y_tr = tr_df["direction_5d"].values
        y_te = te_df["direction_5d"].values

        # B0a
        b0a = MajorityBaselineClassifier()
        b0a.fit(tr_df[FEATURE_COLUMNS], y_tr)
        b0a_res = evaluate_model(b0a, te_df[FEATURE_COLUMNS], y_te)

        # B0b
        b0b = MomentumBaselineClassifier()
        b0b.fit(tr_df[FEATURE_COLUMNS], y_tr)
        b0b_res = evaluate_model(b0b, te_df[FEATURE_COLUMNS], y_te)

        # M0
        m0 = create_m0_model()
        m0.fit(tr_df[FEATURE_COLUMNS], y_tr)
        m0_res = evaluate_model(m0, te_df[FEATURE_COLUMNS], y_te)

        # M1
        m1 = create_m1_model()
        m1.fit(tr_df[M1_FEATURE_COLUMNS], y_tr)
        m1_res = evaluate_model(m1, te_df[M1_FEATURE_COLUMNS], y_te)

        diff_m1_m0 = round(m1_res["balanced_accuracy"] - m0_res["balanced_accuracy"], 4)
        diff_m1_b0a = round(m1_res["balanced_accuracy"] - b0a_res["balanced_accuracy"], 4)

        fold_results.append(
            {
                "fold": k,
                "train_dates": f"{f['train_start']} to {f['train_end']}",
                "test_dates": f"{f['test_start']} to {f['test_end']}",
                "train_count": f["train_count"],
                "test_count": f["test_count"],
                "purged_count": f["train_purged_count"] + f["test_purged_count"],
                "B0a": b0a_res,
                "B0b": b0b_res,
                "M0": m0_res,
                "M1": m1_res,
                "diff_m1_minus_m0_bacc": diff_m1_m0,
                "diff_m1_minus_b0a_bacc": diff_m1_b0a,
            }
        )

    # Compute fold averages for each model
    metric_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc"]
    avg_results: dict[str, dict[str, float | None]] = {"B0a": {}, "B0b": {}, "M0": {}, "M1": {}}

    for m_name in ["B0a", "B0b", "M0", "M1"]:
        for m_key in metric_keys:
            vals = [fr[m_name][m_key] for fr in fold_results if fr[m_name].get(m_key) is not None]
            if vals:
                avg_results[m_name][m_key] = round(float(np.mean(vals)), 4)
            else:
                avg_results[m_name][m_key] = None

    mean_diff_m1_m0 = round(
        avg_results["M1"]["balanced_accuracy"] - avg_results["M0"]["balanced_accuracy"], 4
    )
    mean_diff_m1_b0a = round(
        avg_results["M1"]["balanced_accuracy"] - avg_results["B0a"]["balanced_accuracy"], 4
    )

    return {
        "folds": fold_results,
        "fold_averages": avg_results,
        "mean_diff_m1_minus_m0_bacc": mean_diff_m1_m0,
        "mean_diff_m1_minus_b0a_bacc": mean_diff_m1_b0a,
    }
