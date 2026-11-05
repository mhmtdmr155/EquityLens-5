"""Evaluation metrics and scientific scoring utilities for EquityLens-5 ML models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("EquityLens5.Evaluation")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive classification metrics for binary direction targets.

    Handles single-class test subsets safely without raising exceptions.
    """
    n_samples = int(len(y_true))
    if n_samples == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "sample_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "confusion_matrix": None,
            "positive_ratio": None,
            "negative_ratio": None,
        }

    pos_count = int(np.sum(y_true == 1.0))
    neg_count = int(np.sum(y_true == 0.0))
    pos_ratio = round(pos_count / n_samples, 4)
    neg_ratio = round(neg_count / n_samples, 4)

    # Basic classification metrics
    unique_classes = np.unique(y_true)
    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    if len(unique_classes) < 2:
        bacc = acc
    else:
        bacc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
    prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    # ROC-AUC calculation requires at least two distinct true classes
    if len(unique_classes) >= 2 and y_prob is not None:
        try:
            auc = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            auc = None
    else:
        auc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0])
    cm_dict = {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }

    return {
        "status": "VALID",
        "sample_count": n_samples,
        "positive_ratio": pos_ratio,
        "negative_ratio": neg_ratio,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm_dict,
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a trained model or baseline pipeline on feature matrix X and ground truth y."""
    if len(y) == 0:
        return calculate_metrics(y, np.array([]), None)

    y_pred = model.predict(X)

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            # Binary probabilities for positive class (column 1)
            if probs.shape[1] >= 2:
                y_prob = probs[:, 1]
            else:
                y_prob = probs[:, 0]
        except Exception as exc:
            logger.debug("Failed extracting probabilities: %s", exc)
            y_prob = None

    return calculate_metrics(y, y_pred, y_prob)


def evaluate_per_symbol(
    model: Any,
    df_split: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "direction_5d",
) -> dict[str, Any]:
    """Evaluate model performance broken down by individual company."""
    results: dict[str, Any] = {}
    tickers = sorted(df_split["ticker"].unique())

    for sym in tickers:
        sub_df = df_split[df_split["ticker"] == sym]
        if len(sub_df) == 0:
            results[sym] = {"status": "INSUFFICIENT_DATA", "sample_count": 0}
            continue

        X_sym = sub_df[feature_cols]
        y_sym = sub_df[target_col].values
        sym_metrics = evaluate_model(model, X_sym, y_sym)
        results[sym] = sym_metrics

    return results
