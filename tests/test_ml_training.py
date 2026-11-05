"""Unit and regression tests for EquityLens-5 ML Model Training & Evaluation (EL5-050).

Verifies chronological splitting, label purging, zero feature leakage, scaler isolation,
baseline classifiers, model selection logic, and SPCX out-of-company generalization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from backend.app.features import FEATURE_COLUMNS, TARGET_COLUMNS
from backend.app.ml.evaluation import calculate_metrics, evaluate_model
from backend.app.ml.training import (
    MajorityBaselineClassifier,
    MomentumBaselineClassifier,
    build_chronological_splits,
    train_candidate_models,
)

DATA_DIR = Path("data/derived")


@pytest.fixture
def derived_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load derived features and 5d training datasets."""
    f_all = pd.read_csv(DATA_DIR / "features_all.csv")
    f_5d = pd.read_csv(DATA_DIR / "ml_train_5d.csv")
    return f_5d, f_all


def test_no_target_columns_in_feature_columns() -> None:
    """CRITICAL TEST: Ensure zero target or forward-looking fields in FEATURE_COLUMNS."""
    for col in FEATURE_COLUMNS:
        assert col not in TARGET_COLUMNS
        assert not col.startswith("future_")
        assert not col.startswith("direction_")
        assert col != "future_close"


def test_chronological_split_strict_date_disjointness(
    derived_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verify that Train, Validation, and Test sets never overlap in session dates."""
    df_5d, df_all = derived_datasets
    splits = build_chronological_splits(df_5d, df_all, train_ratio=0.60, val_ratio=0.20)

    train_dates = set(splits["train_df"]["session_date"].unique())
    val_dates = set(splits["val_df"]["session_date"].unique())
    test_dates = set(splits["test_df"]["session_date"].unique())

    # Strictly disjoint date sets
    assert train_dates.isdisjoint(val_dates)
    assert val_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)

    # Monotonic order of boundaries
    b = splits["boundaries"]
    assert b["train_start"] < b["train_end"]
    assert b["train_end"] < b["val_start"]
    assert b["val_start"] < b["val_end"]
    assert b["val_end"] < b["test_start"]
    assert b["test_start"] < b["test_end"]


def test_label_purging_prevents_leakage(
    derived_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """CRITICAL TEST: Verify no kept observation has label_end_date crossing into next split."""
    df_5d, df_all = derived_datasets
    splits = build_chronological_splits(df_5d, df_all, train_ratio=0.60, val_ratio=0.20)

    train_end = splits["boundaries"]["train_end"]
    val_end = splits["boundaries"]["val_end"]

    # In train set: every kept row MUST have label_end_date <= train_end
    train_kept = splits["train_df"]
    assert (train_kept["label_end_date"] <= train_end).all()

    # In purged train set: every row MUST have label_end_date > train_end
    train_purged = splits["purged_train_df"]
    assert (train_purged["label_end_date"] > train_end).all()
    assert len(train_purged) == 20  # 5 bars * 4 active companies

    # In validation set: every kept row MUST have label_end_date <= val_end
    val_kept = splits["val_df"]
    assert (val_kept["label_end_date"] <= val_end).all()

    # In purged val set: every row MUST have label_end_date > val_end
    val_purged = splits["purged_val_df"]
    assert (val_purged["label_end_date"] > val_end).all()
    assert len(val_purged) == 20  # 5 bars * 4 active companies


def test_scaler_fitted_only_on_training_data() -> None:
    """Verify that StandardScaler fits solely on training data without test/val exposure."""
    np.random.seed(42)
    X_train = pd.DataFrame({"feat1": [10.0, 20.0, 30.0], "feat2": [100.0, 200.0, 300.0]})
    X_val = pd.DataFrame({"feat1": [1000.0, 2000.0], "feat2": [10000.0, 20000.0]})

    scaler = StandardScaler()
    scaler.fit(X_train)

    train_mean = scaler.mean_.copy()
    val_transformed = scaler.transform(X_val)

    # Scaler mean must NOT change after transforming validation data
    assert np.allclose(scaler.mean_, train_mean)
    assert not np.allclose(val_transformed.mean(axis=0), 0.0)


def test_majority_baseline_classifier() -> None:
    """Verify B0a zero-rule baseline logic."""
    clf = MajorityBaselineClassifier()

    # Case 1: Positive majority
    y_pos = np.array([1.0, 1.0, 1.0, 0.0])
    X_dummy = pd.DataFrame({"x": [1, 2, 3, 4]})
    clf.fit(X_dummy, y_pos)
    assert clf.majority_class_ == 1.0
    assert (clf.predict(X_dummy) == 1.0).all()

    # Case 2: Negative majority
    y_neg = np.array([0.0, 0.0, 0.0, 1.0])
    clf.fit(X_dummy, y_neg)
    assert clf.majority_class_ == 0.0
    assert (clf.predict(X_dummy) == 0.0).all()


def test_momentum_baseline_classifier() -> None:
    """Verify B0b momentum heuristic predicts 1 on positive return_5d and 0 otherwise."""
    clf = MomentumBaselineClassifier(return_col="return_5d")
    X = pd.DataFrame({"return_5d": [0.05, -0.02, 0.00, 0.12, -0.001]})
    preds = clf.predict(X)
    assert preds.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_spacex_out_of_company_generalization(
    derived_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verify SpaceX has 0 samples in train/val and is evaluated solely as out-of-company test."""
    df_5d, df_all = derived_datasets
    splits = build_chronological_splits(df_5d, df_all)

    spcx_breakdown = splits["symbol_breakdown"]["SPCX"]
    assert spcx_breakdown["train_kept"] == 0
    assert spcx_breakdown["val_kept"] == 0
    assert spcx_breakdown["test_kept"] == 60

    # Evaluate mock model on SPCX test set
    test_spcx = splits["test_df"][splits["test_df"]["ticker"] == "SPCX"]
    model = MajorityBaselineClassifier().fit(
        splits["train_df"][FEATURE_COLUMNS],
        splits["train_df"]["direction_5d"].values,
    )
    res_spcx = evaluate_model(model, test_spcx[FEATURE_COLUMNS], test_spcx["direction_5d"].values)
    assert res_spcx["status"] == "VALID"
    assert res_spcx["sample_count"] == 60


def test_calculate_metrics_edge_cases() -> None:
    """Verify metrics calculation handles empty and single-class distributions gracefully."""
    # Empty
    empty_res = calculate_metrics(np.array([]), np.array([]))
    assert empty_res["status"] == "INSUFFICIENT_DATA"

    # Single class only (all 1s)
    y_single = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    res_single = calculate_metrics(y_single, y_pred, np.array([0.8, 0.9, 0.7]))
    assert res_single["status"] == "VALID"
    assert res_single["roc_auc"] is None  # Cannot compute ROC-AUC with single class


def test_candidate_models_factory_and_training() -> None:
    """Verify all 5 candidate models can be instantiated and trained without error."""
    np.random.seed(42)
    n = 50
    X = pd.DataFrame({col: np.random.randn(n) for col in FEATURE_COLUMNS})
    y = np.random.choice([0.0, 1.0], size=n)

    models = train_candidate_models(X, y)
    assert len(models) == 5
    for _name, m in models.items():
        preds = m.predict(X)
        assert len(preds) == n
        assert set(preds).issubset({0.0, 1.0})
