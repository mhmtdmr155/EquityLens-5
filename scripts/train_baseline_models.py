"""EquityLens-5 ML Baseline Training and Scientific Evaluation Pipeline (EL5-050).

Implements strict chronological train/validation/test splitting with label purging,
evaluates zero-rule (B0a) and momentum (B0b) baselines alongside Logistic Regression
(M0) and Tree-based ensembles (M1), selects the best candidate purely on validation
balanced accuracy, evaluates once on the test set, and exports the serialized model.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.features import FEATURE_COLUMNS
from backend.app.ml.evaluation import evaluate_model, evaluate_per_symbol
from backend.app.ml.training import (
    build_chronological_splits,
    create_model_candidate,
    train_candidate_models,
)

logger = logging.getLogger("EquityLens5.ModelTraining")


def run_pipeline(
    data_dir: Path = PROJECT_ROOT / "data" / "derived",
    output_dir: Path = PROJECT_ROOT / "models",
) -> dict[str, Any]:
    """Execute complete ML training, scientific evaluation, and serialization workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load derived datasets
    f_all_path = data_dir / "features_all.csv"
    train_5d_path = data_dir / "ml_train_5d.csv"

    if not f_all_path.exists() or not train_5d_path.exists():
        msg = f"Required CSV datasets not found in {data_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading feature datasets from %s...", data_dir)
    df_all = pd.read_csv(f_all_path)
    df_5d = pd.read_csv(train_5d_path)

    # 2. Build chronological splits with label purging
    logger.info("Building chronological partitions with label purging...")
    splits = build_chronological_splits(df_5d, df_all, train_ratio=0.60, val_ratio=0.20)

    train_df = splits["train_df"]
    val_df = splits["val_df"]
    test_df = splits["test_df"]

    logger.info(
        "Split counts: Train=%d (purged=%d), Val=%d (purged=%d), Test=%d (purged=%d)",
        splits["counts"]["train_kept"],
        splits["counts"]["train_purged"],
        splits["counts"]["val_kept"],
        splits["counts"]["val_purged"],
        splits["counts"]["test_kept"],
        splits["counts"]["test_purged"],
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["direction_5d"].values

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["direction_5d"].values

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["direction_5d"].values

    # 3. Train all candidates on training partition
    candidate_keys = [
        "B0a_majority",
        "B0b_momentum",
        "M0_logistic_regression",
        "M1_random_forest",
        "M1_hist_gradient_boosting",
    ]

    logger.info("Training %d candidate models on Train set...", len(candidate_keys))
    models = train_candidate_models(X_train, y_train, candidate_keys)

    # 4. Evaluate candidates on VALIDATION set
    logger.info("Evaluating candidates on Validation set...")
    val_results: dict[str, Any] = {}
    for key, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        val_results[key] = metrics
        logger.info(
            "Validation [%s] -> BalancedAcc=%.4f, Acc=%.4f, F1=%.4f, AUC=%s",
            key,
            metrics["balanced_accuracy"],
            metrics["accuracy"],
            metrics["f1_score"],
            str(metrics["roc_auc"]),
        )

    # 5. Scientific Model Selection on Validation Balanced Accuracy
    best_key = max(
        candidate_keys,
        key=lambda k: (
            val_results[k]["balanced_accuracy"]
            if val_results[k]["balanced_accuracy"] is not None
            else -1.0,
            val_results[k]["f1_score"] if val_results[k]["f1_score"] is not None else -1.0,
        ),
    )
    logger.info("Selected winning model architecture based on validation: %s", best_key)

    # 6. Retrain winning architecture on Train + Validation combined
    logger.info("Retraining winning architecture (%s) on Train + Validation...", best_key)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df[FEATURE_COLUMNS]
    y_train_val = train_val_df["direction_5d"].values

    eval_model = create_model_candidate(best_key)
    eval_model.fit(X_train_val, y_train_val)

    # 7. Final single evaluation on TEST set
    logger.info("Evaluating selected model on TEST set (exactly once)...")
    test_metrics = evaluate_model(eval_model, X_test, y_test)
    test_by_symbol = evaluate_per_symbol(eval_model, test_df, FEATURE_COLUMNS, "direction_5d")

    # Also evaluate all other candidates on TEST set for complete scientific comparison table
    all_test_metrics: dict[str, Any] = {}
    for key, m in models.items():
        all_test_metrics[key] = evaluate_model(m, X_test, y_test)

    logger.info(
        "Final Test [%s] -> BalancedAcc=%.4f, Acc=%.4f, F1=%.4f, AUC=%s",
        best_key,
        test_metrics["balanced_accuracy"],
        test_metrics["accuracy"],
        test_metrics["f1_score"],
        str(test_metrics["roc_auc"]),
    )

    # 8. Train full inference model on all available labeled data (train + val + test)
    # This model is specifically tagged for production inference API serving
    logger.info("Training full inference model across all 1,868 non-purged labeled rows...")
    full_labeled_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    X_full = full_labeled_df[FEATURE_COLUMNS]
    y_full = full_labeled_df["direction_5d"].values

    inference_model = create_model_candidate(best_key)
    inference_model.fit(X_full, y_full)

    # 9. Save artifacts to models/
    model_path = output_dir / "direction_5d_model.joblib"
    joblib.dump(inference_model, model_path)
    logger.info("Saved serialized inference model to %s", model_path)

    metadata = {
        "model_name": best_key,
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "target_variable": "direction_5d",
        "horizon_trading_days": 5,
        "primary_selection_metric": "balanced_accuracy",
        "selected_architecture": best_key,
        "candidate_architectures": candidate_keys,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "split_boundaries": splits["boundaries"],
        "sample_counts": splits["counts"],
        "symbol_breakdown": splits["symbol_breakdown"],
        "validation_comparison": val_results,
        "test_comparison": all_test_metrics,
        "selected_model_test_metrics": test_metrics,
        "selected_model_test_per_symbol": test_by_symbol,
        "model_artifact": str(model_path.name),
    }

    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    eval_path = output_dir / "evaluation_metrics.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": best_key,
                "validation_metrics": val_results[best_key],
                "test_metrics": test_metrics,
                "test_by_symbol": test_by_symbol,
                "all_models_validation": val_results,
                "all_models_test": all_test_metrics,
            },
            f,
            indent=2,
        )

    logger.info("Saved metadata and evaluation metrics to %s", output_dir)
    return metadata


def main() -> int:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="EquityLens-5 ML Baseline Training Pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived",
        help="Path to derived datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to save serialized models and metrics.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        run_pipeline(args.data_dir, args.output_dir)
        return 0
    except Exception as exc:
        logger.exception("Model training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
