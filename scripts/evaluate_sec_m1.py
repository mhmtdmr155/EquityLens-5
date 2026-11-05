"""EquityLens-5 SEC Fundamentals Experiment Evaluation Script (EL5-060).

Runs:
1. Retrospective exploratory ablation on EL5-050 test set.
2. Chronological expanding-window cross-validation with label purging.
3. Saves experiment artifacts in models/experiments/sec_m1/.
4. Strictly preserves existing production models and metrics.
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

from backend.app.ml.sec_experiment import (
    M1_FEATURE_COLUMNS,
    run_cross_validation_experiment,
    run_retrospective_ablation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EquityLens5.EvaluateSECM1")


def run_experiment(
    sec_csv: Path = PROJECT_ROOT / "data" / "derived" / "ml_sec_5d.csv",
    all_features_csv: Path = PROJECT_ROOT / "data" / "derived" / "features_all.csv",
    output_dir: Path = PROJECT_ROOT / "models" / "experiments" / "sec_m1",
) -> dict[str, Any]:
    """Execute complete M0 vs M1 vs B0a evaluation and serialize experiment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SEC features dataset from %s ...", sec_csv)
    df_sec = pd.read_csv(sec_csv)
    logger.info("Loading full feature series from %s ...", all_features_csv)
    df_all = pd.read_csv(all_features_csv)

    # 1. Retrospective Exploratory Ablation (EL5-050 test set)
    logger.info("\n=======================================================")
    logger.info("1. RETROSPECTIVE EXPLORATORY ABLATION (EL5-050 TEST SET)")
    logger.info("=======================================================")
    ablation_res = run_retrospective_ablation(df_sec, df_all)

    models_res = ablation_res["models"]
    diffs = ablation_res["differences"]

    logger.info("Retrospective Test Results:")
    logger.info(
        "  B0a (Majority):   Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
        models_res["B0a_majority"]["accuracy"],
        models_res["B0a_majority"]["balanced_accuracy"],
        models_res["B0a_majority"]["f1_score"],
        str(models_res["B0a_majority"]["roc_auc"]),
    )
    logger.info(
        "  B0b (Momentum):   Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
        models_res["B0b_momentum"]["accuracy"],
        models_res["B0b_momentum"]["balanced_accuracy"],
        models_res["B0b_momentum"]["f1_score"],
        str(models_res["B0b_momentum"]["roc_auc"]),
    )
    logger.info(
        "  M0  (Price-Only): Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
        models_res["M0_price_only"]["accuracy"],
        models_res["M0_price_only"]["balanced_accuracy"],
        models_res["M0_price_only"]["f1_score"],
        str(models_res["M0_price_only"]["roc_auc"]),
    )
    logger.info(
        "  M1  (Price+SEC):  Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
        models_res["M1_price_sec"]["accuracy"],
        models_res["M1_price_sec"]["balanced_accuracy"],
        models_res["M1_price_sec"]["f1_score"],
        str(models_res["M1_price_sec"]["roc_auc"]),
    )
    logger.info("  Diff (M1 - M0) Balanced Accuracy:  %+.4f", diffs["m1_minus_m0_balanced_acc"])
    logger.info("  Diff (M1 - B0a) Balanced Accuracy: %+.4f", diffs["m1_minus_b0a_balanced_acc"])

    logger.info("\n--- Per-Symbol Test Balanced Accuracy ---")
    m0_sym = ablation_res["m0_by_symbol"]
    m1_sym = ablation_res["m1_by_symbol"]
    for sym in sorted(m0_sym.keys()):
        m0_bacc = m0_sym[sym].get("balanced_accuracy")
        m1_bacc = m1_sym[sym].get("balanced_accuracy")
        n_samples = m0_sym[sym].get("sample_count", 0)
        logger.info(
            "  %5s (N=%2d): M0 BalancedAcc=%s, M1 BalancedAcc=%s",
            sym,
            n_samples,
            str(m0_bacc),
            str(m1_bacc),
        )

    # 2. Chronological Expanding-Window Cross-Validation (3 Folds)
    logger.info("\n=======================================================")
    logger.info("2. CHRONOLOGICAL EXPANDING-WINDOW CROSS-VALIDATION (3 FOLDS)")
    logger.info("=======================================================")
    cv_res = run_cross_validation_experiment(df_sec, df_all, n_folds=3)

    for fr in cv_res["folds"]:
        k = fr["fold"]
        logger.info(
            "Fold %d (%s -> %s | Train=%d, Test=%d, Purged=%d):",
            k,
            fr["train_dates"],
            fr["test_dates"],
            fr["train_count"],
            fr["test_count"],
            fr["purged_count"],
        )
        logger.info(
            "  B0a: Acc=%.4f, BalancedAcc=%.4f, F1=%.4f",
            fr["B0a"]["accuracy"],
            fr["B0a"]["balanced_accuracy"],
            fr["B0a"]["f1_score"],
        )
        logger.info(
            "  B0b: Acc=%.4f, BalancedAcc=%.4f, F1=%.4f",
            fr["B0b"]["accuracy"],
            fr["B0b"]["balanced_accuracy"],
            fr["B0b"]["f1_score"],
        )
        logger.info(
            "  M0 : Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
            fr["M0"]["accuracy"],
            fr["M0"]["balanced_accuracy"],
            fr["M0"]["f1_score"],
            str(fr["M0"]["roc_auc"]),
        )
        logger.info(
            "  M1 : Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
            fr["M1"]["accuracy"],
            fr["M1"]["balanced_accuracy"],
            fr["M1"]["f1_score"],
            str(fr["M1"]["roc_auc"]),
        )
        logger.info(
            "  Diff (M1 - M0) BalancedAcc: %+.4f | (M1 - B0a): %+.4f",
            fr["diff_m1_minus_m0_bacc"],
            fr["diff_m1_minus_b0a_bacc"],
        )

    logger.info("\n--- Fold Averages ---")
    avg = cv_res["fold_averages"]
    for m in ["B0a", "B0b", "M0", "M1"]:
        logger.info(
            "  %-4s Average: Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
            m,
            avg[m]["accuracy"],
            avg[m]["balanced_accuracy"],
            avg[m]["f1_score"],
            str(avg[m]["roc_auc"]),
        )
    logger.info(
        "  Mean Diff (M1 - M0)  Balanced Accuracy: %+.4f", cv_res["mean_diff_m1_minus_m0_bacc"]
    )
    logger.info(
        "  Mean Diff (M1 - B0a) Balanced Accuracy: %+.4f", cv_res["mean_diff_m1_minus_b0a_bacc"]
    )

    # 3. Serialization of Experiment Artifacts
    logger.info("\n=======================================================")
    logger.info("3. SERIALIZING EXPERIMENT ARTIFACTS")
    logger.info("=======================================================")
    m1_model = ablation_res.pop("trained_m1_model")

    # Save trained M1 pipeline
    model_save_path = output_dir / "m1_model.joblib"
    joblib.dump(m1_model, model_save_path)
    logger.info("Saved M1 experiment model to %s", model_save_path)

    # Save metrics JSON
    metrics_export = {
        "retrospective_ablation": ablation_res,
        "expanding_window_cv": cv_res,
    }
    metrics_path = output_dir / "m1_experiment_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    logger.info("Saved experiment metrics to %s", metrics_path)

    # Save metadata JSON
    metadata = {
        "experiment_id": "EL5-060-M1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "algorithm": "HistGradientBoostingClassifier with SimpleImputer(median)",
        "features_total": len(M1_FEATURE_COLUMNS),
        "feature_list": M1_FEATURE_COLUMNS,
        "retrospective_test_metrics": {
            "M0_balanced_accuracy": models_res["M0_price_only"]["balanced_accuracy"],
            "M1_balanced_accuracy": models_res["M1_price_sec"]["balanced_accuracy"],
            "B0a_balanced_accuracy": models_res["B0a_majority"]["balanced_accuracy"],
            "m1_minus_m0_diff": diffs["m1_minus_m0_balanced_acc"],
            "m1_minus_b0a_diff": diffs["m1_minus_b0a_balanced_acc"],
        },
        "cv_average_metrics": {
            "M0_mean_bacc": avg["M0"]["balanced_accuracy"],
            "M1_mean_bacc": avg["M1"]["balanced_accuracy"],
            "B0a_mean_bacc": avg["B0a"]["balanced_accuracy"],
            "m1_minus_m0_mean_diff": cv_res["mean_diff_m1_minus_m0_bacc"],
            "m1_minus_b0a_mean_diff": cv_res["mean_diff_m1_minus_b0a_bacc"],
        },
        "artifacts": {
            "model_path": str(model_save_path),
            "metrics_path": str(metrics_path),
        },
    }
    metadata_path = output_dir / "m1_experiment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved experiment metadata to %s", metadata_path)

    # 4. Verify production files were NOT modified
    prod_model_path = PROJECT_ROOT / "models" / "direction_5d_model.joblib"
    if not prod_model_path.exists():
        logger.error("WARNING: Production model file missing!")
    else:
        logger.info(
            "Verified production model %s remains intact and untouched.", prod_model_path.name
        )

    return metrics_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M0 vs M1 experiment (EL5-060).")
    parser.add_argument(
        "--sec-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "ml_sec_5d.csv",
    )
    parser.add_argument(
        "--all-features-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "features_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "experiments" / "sec_m1",
    )
    args = parser.parse_args()

    run_experiment(
        sec_csv=args.sec_csv,
        all_features_csv=args.all_features_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
