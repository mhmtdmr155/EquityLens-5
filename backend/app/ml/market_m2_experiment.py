"""EquityLens-5 Market/Sector M2 Model Experiment (EL5-061).

Compares:
- B0a: Majority baseline
- B0b: Momentum baseline
- M0: Price-only HistGradientBoosting (23 features)
- M1: Price + SEC Fundamentals HistGradientBoosting (31 features)
- M2: Price + SEC Fundamentals + Market/Sector ETF HistGradientBoosting (40 features)

Evaluates under:
1. Retrospective exploratory ablation on EL5-050 test set.
2. Leak-free expanding-window chronological cross-validation with label purging (3 folds).
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
from backend.app.market_features import MARKET_FEATURE_COLUMNS
from backend.app.ml.evaluation import evaluate_model, evaluate_per_symbol
from backend.app.ml.sec_experiment import build_expanding_window_folds
from backend.app.ml.training import (
    MajorityBaselineClassifier,
    MomentumBaselineClassifier,
    build_chronological_splits,
)
from backend.app.sec_fundamentals import SEC_FEATURE_COLUMNS

logger = logging.getLogger("EquityLens5.MarketM2Experiment")

# Feature sets
M0_FEATURE_COLUMNS: Final[list[str]] = FEATURE_COLUMNS  # 23 features
M1_FEATURE_COLUMNS: Final[list[str]] = FEATURE_COLUMNS + SEC_FEATURE_COLUMNS  # 31 features
M2_FEATURE_COLUMNS: Final[list[str]] = (
    FEATURE_COLUMNS + SEC_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS
)  # 40 features


def create_m0_model() -> HistGradientBoostingClassifier:
    """Instantiate M0 price-only HistGradientBoosting model (23 features)."""
    return HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        min_samples_leaf=20,
        random_state=42,
    )


def create_m1_model() -> Pipeline:
    """Instantiate M1 price + fundamental pipeline (31 features)."""
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


def create_m2_model() -> Pipeline:
    """Instantiate M2 price + fundamental + market/sector pipeline (40 features).

    Uses identical classifier hyperparameters as M0 and M1.
    Includes SimpleImputer fitted strictly on training data.
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


def run_m2_retrospective_ablation(
    df_m2: pd.DataFrame,
    df_all: pd.DataFrame,
) -> dict[str, Any]:
    """Execute retrospective exploratory ablation on the exact EL5-050 test set.

    Compares B0a, B0b, M0, M1, and M2 on identical samples and dates.
    """
    splits = build_chronological_splits(df_m2, df_all, train_ratio=0.60, val_ratio=0.20)
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    test_df = splits["test_df"]

    # Retrain winning architecture on Train + Validation combined (same as EL5-050)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    y_train = train_val_df["direction_5d"].values
    y_test = test_df["direction_5d"].values

    # 1. B0a: Majority Baseline
    b0a = MajorityBaselineClassifier()
    b0a.fit(train_val_df[M0_FEATURE_COLUMNS], y_train)
    b0a_res = evaluate_model(b0a, test_df[M0_FEATURE_COLUMNS], y_test)

    # 2. B0b: Momentum Baseline
    b0b = MomentumBaselineClassifier()
    b0b.fit(train_val_df[M0_FEATURE_COLUMNS], y_train)
    b0b_res = evaluate_model(b0b, test_df[M0_FEATURE_COLUMNS], y_test)

    # 3. M0: Price-only (23 features)
    m0 = create_m0_model()
    m0.fit(train_val_df[M0_FEATURE_COLUMNS], y_train)
    m0_res = evaluate_model(m0, test_df[M0_FEATURE_COLUMNS], y_test)
    m0_sym = evaluate_per_symbol(m0, test_df, M0_FEATURE_COLUMNS, "direction_5d")

    # 4. M1: Price + SEC (31 features)
    m1 = create_m1_model()
    m1.fit(train_val_df[M1_FEATURE_COLUMNS], y_train)
    m1_res = evaluate_model(m1, test_df[M1_FEATURE_COLUMNS], y_test)
    m1_sym = evaluate_per_symbol(m1, test_df, M1_FEATURE_COLUMNS, "direction_5d")

    # 5. M2: Price + SEC + Market (40 features)
    m2 = create_m2_model()
    m2.fit(train_val_df[M2_FEATURE_COLUMNS], y_train)
    m2_res = evaluate_model(m2, test_df[M2_FEATURE_COLUMNS], y_test)
    m2_sym = evaluate_per_symbol(m2, test_df, M2_FEATURE_COLUMNS, "direction_5d")

    diff_m2_m1 = round(m2_res["balanced_accuracy"] - m1_res["balanced_accuracy"], 4)
    diff_m2_m0 = round(m2_res["balanced_accuracy"] - m0_res["balanced_accuracy"], 4)
    diff_m2_b0a = round(m2_res["balanced_accuracy"] - b0a_res["balanced_accuracy"], 4)

    return {
        "split_counts": {
            "train_val_count": len(train_val_df),
            "test_count": len(test_df),
            "purged_count": splits["counts"]["total_purged"],
        },
        "boundaries": splits["boundaries"],
        "models": {
            "B0a_majority": b0a_res,
            "B0b_momentum": b0b_res,
            "M0_price_only": m0_res,
            "M1_price_sec": m1_res,
            "M2_price_sec_market": m2_res,
        },
        "differences": {
            "m2_minus_m1_balanced_acc": diff_m2_m1,
            "m2_minus_m0_balanced_acc": diff_m2_m0,
            "m2_minus_b0a_balanced_acc": diff_m2_b0a,
        },
        "m0_by_symbol": m0_sym,
        "m1_by_symbol": m1_sym,
        "m2_by_symbol": m2_sym,
        "trained_m2_model": m2,
    }


def run_m2_cross_validation_experiment(
    df_m2: pd.DataFrame,
    df_all: pd.DataFrame,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Execute expanding-window cross-validation for M0, M1, M2, B0a, B0b."""
    folds = build_expanding_window_folds(df_m2, df_all, n_folds=n_folds)
    fold_results: list[dict[str, Any]] = []

    for f in folds:
        k = f["fold_index"]
        tr_df = f["train_df"]
        te_df = f["test_df"]

        y_tr = tr_df["direction_5d"].values
        y_te = te_df["direction_5d"].values

        # B0a
        b0a = MajorityBaselineClassifier()
        b0a.fit(tr_df[M0_FEATURE_COLUMNS], y_tr)
        b0a_res = evaluate_model(b0a, te_df[M0_FEATURE_COLUMNS], y_te)

        # B0b
        b0b = MomentumBaselineClassifier()
        b0b.fit(tr_df[M0_FEATURE_COLUMNS], y_tr)
        b0b_res = evaluate_model(b0b, te_df[M0_FEATURE_COLUMNS], y_te)

        # M0
        m0 = create_m0_model()
        m0.fit(tr_df[M0_FEATURE_COLUMNS], y_tr)
        m0_res = evaluate_model(m0, te_df[M0_FEATURE_COLUMNS], y_te)

        # M1
        m1 = create_m1_model()
        m1.fit(tr_df[M1_FEATURE_COLUMNS], y_tr)
        m1_res = evaluate_model(m1, te_df[M1_FEATURE_COLUMNS], y_te)

        # M2
        m2 = create_m2_model()
        m2.fit(tr_df[M2_FEATURE_COLUMNS], y_tr)
        m2_res = evaluate_model(m2, te_df[M2_FEATURE_COLUMNS], y_te)

        diff_m2_m1 = round(m2_res["balanced_accuracy"] - m1_res["balanced_accuracy"], 4)
        diff_m2_m0 = round(m2_res["balanced_accuracy"] - m0_res["balanced_accuracy"], 4)
        diff_m2_b0a = round(m2_res["balanced_accuracy"] - b0a_res["balanced_accuracy"], 4)

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
                "M2": m2_res,
                "diff_m2_minus_m1_bacc": diff_m2_m1,
                "diff_m2_minus_m0_bacc": diff_m2_m0,
                "diff_m2_minus_b0a_bacc": diff_m2_b0a,
            }
        )

    # Compute fold averages
    metric_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc"]
    avg_results: dict[str, dict[str, float | None]] = {
        "B0a": {},
        "B0b": {},
        "M0": {},
        "M1": {},
        "M2": {},
    }

    for m_name in ["B0a", "B0b", "M0", "M1", "M2"]:
        for m_key in metric_keys:
            vals = [fr[m_name][m_key] for fr in fold_results if fr[m_name].get(m_key) is not None]
            if vals:
                avg_results[m_name][m_key] = round(float(np.mean(vals)), 4)
            else:
                avg_results[m_name][m_key] = None

    mean_diff_m2_m1 = round(
        avg_results["M2"]["balanced_accuracy"] - avg_results["M1"]["balanced_accuracy"], 4
    )
    mean_diff_m2_m0 = round(
        avg_results["M2"]["balanced_accuracy"] - avg_results["M0"]["balanced_accuracy"], 4
    )
    mean_diff_m2_b0a = round(
        avg_results["M2"]["balanced_accuracy"] - avg_results["B0a"]["balanced_accuracy"], 4
    )

    return {
        "folds": fold_results,
        "fold_averages": avg_results,
        "mean_diff_m2_minus_m1_bacc": mean_diff_m2_m1,
        "mean_diff_m2_minus_m0_bacc": mean_diff_m2_m0,
        "mean_diff_m2_minus_b0a_bacc": mean_diff_m2_b0a,
    }
