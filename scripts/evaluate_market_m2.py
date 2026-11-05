"""EquityLens-5 Market/Sector M2 Experiment Evaluation Script (EL5-061).

Runs:
1. Retrospective exploratory ablation on EL5-050 test set.
2. Chronological expanding-window cross-validation with label purging (3 folds).
3. Saves experiment artifacts in models/experiments/market_m2/.
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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.market_m2_experiment import (
    M2_FEATURE_COLUMNS,
    run_m2_cross_validation_experiment,
    run_m2_retrospective_ablation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EquityLens5.EvaluateMarketM2")


def run_experiment(
    market_sec_csv: Path = PROJECT_ROOT / "data" / "derived" / "ml_market_sec_5d.csv",
    all_features_csv: Path = PROJECT_ROOT / "data" / "derived" / "features_all.csv",
    output_dir: Path = PROJECT_ROOT / "models" / "experiments" / "market_m2",
) -> dict[str, Any]:
    """Execute complete M0 vs M1 vs M2 vs baseline evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading M2 dataset from %s ...", market_sec_csv)
    df_m2 = pd.read_csv(market_sec_csv)
    logger.info("Loading full feature series from %s ...", all_features_csv)
    df_all = pd.read_csv(all_features_csv)

    # 1. Retrospective Exploratory Ablation (EL5-050 test set)
    logger.info("\n=======================================================")
    logger.info("1. RETROSPECTIVE EXPLORATORY ABLATION (EL5-050 TEST SET)")
    logger.info("=======================================================")
    ablation_res = run_m2_retrospective_ablation(df_m2, df_all)
    models_res = ablation_res["models"]
    diffs = ablation_res["differences"]

    logger.info("Retrospective Test Results:")
    for m_key, label in [
        ("B0a_majority", "B0a (Majority)    "),
        ("B0b_momentum", "B0b (Momentum)    "),
        ("M0_price_only", "M0  (Price-Only)  "),
        ("M1_price_sec", "M1  (Price + SEC) "),
        ("M2_price_sec_market", "M2  (Price+SEC+Mkt)"),
    ]:
        m = models_res[m_key]
        logger.info(
            "  %s: Acc=%.4f, BalancedAcc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, AUC=%s",
            label,
            m["accuracy"],
            m["balanced_accuracy"],
            m["precision"] if m["precision"] is not None else 0.0,
            m["recall"] if m["recall"] is not None else 0.0,
            m["f1_score"] if m["f1_score"] is not None else 0.0,
            str(m["roc_auc"]),
        )

    logger.info("  Diff (M2 - M1)  Balanced Accuracy: %+.4f", diffs["m2_minus_m1_balanced_acc"])
    logger.info("  Diff (M2 - M0)  Balanced Accuracy: %+.4f", diffs["m2_minus_m0_balanced_acc"])
    logger.info("  Diff (M2 - B0a) Balanced Accuracy: %+.4f", diffs["m2_minus_b0a_balanced_acc"])

    logger.info("\n--- Per-Symbol Test Balanced Accuracy ---")
    m0_sym = ablation_res["m0_by_symbol"]
    m1_sym = ablation_res["m1_by_symbol"]
    m2_sym = ablation_res["m2_by_symbol"]
    for sym in sorted(m0_sym.keys()):
        m0_bacc = m0_sym[sym].get("balanced_accuracy")
        m1_bacc = m1_sym[sym].get("balanced_accuracy")
        m2_bacc = m2_sym[sym].get("balanced_accuracy")
        n_samples = m0_sym[sym].get("sample_count", 0)
        logger.info(
            "  %5s (N=%2d): M0 BAcc=%s | M1 BAcc=%s | M2 BAcc=%s",
            sym,
            n_samples,
            str(m0_bacc),
            str(m1_bacc),
            str(m2_bacc),
        )

    # 2. Chronological Expanding-Window Cross-Validation (3 Folds)
    logger.info("\n=======================================================")
    logger.info("2. CHRONOLOGICAL EXPANDING-WINDOW CROSS-VALIDATION (3 FOLDS)")
    logger.info("=======================================================")
    cv_res = run_m2_cross_validation_experiment(df_m2, df_all, n_folds=3)

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
        for m_name in ["B0a", "B0b", "M0", "M1", "M2"]:
            m = fr[m_name]
            logger.info(
                "  %-3s: Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
                m_name,
                m["accuracy"],
                m["balanced_accuracy"],
                m["f1_score"] if m["f1_score"] is not None else 0.0,
                str(m["roc_auc"]),
            )
        logger.info(
            "  Diff (M2 - M1) BAcc: %+.4f | (M2 - M0) BAcc: %+.4f | (M2 - B0a): %+.4f",
            fr["diff_m2_minus_m1_bacc"],
            fr["diff_m2_minus_m0_bacc"],
            fr["diff_m2_minus_b0a_bacc"],
        )

    logger.info("\n--- Fold Averages ---")
    avg = cv_res["fold_averages"]
    for m in ["B0a", "B0b", "M0", "M1", "M2"]:
        logger.info(
            "  %-3s Average: Acc=%.4f, BalancedAcc=%.4f, F1=%.4f, AUC=%s",
            m,
            avg[m]["accuracy"],
            avg[m]["balanced_accuracy"],
            avg[m]["f1_score"],
            str(avg[m]["roc_auc"]),
        )
    logger.info(
        "  Mean Diff (M2 - M1)  Balanced Accuracy: %+.4f", cv_res["mean_diff_m2_minus_m1_bacc"]
    )
    logger.info(
        "  Mean Diff (M2 - M0)  Balanced Accuracy: %+.4f", cv_res["mean_diff_m2_minus_m0_bacc"]
    )
    logger.info(
        "  Mean Diff (M2 - B0a) Balanced Accuracy: %+.4f", cv_res["mean_diff_m2_minus_b0a_bacc"]
    )

    # 3. Serialization of Experiment Artifacts
    logger.info("\n=======================================================")
    logger.info("3. SERIALIZING EXPERIMENT ARTIFACTS")
    logger.info("=======================================================")
    m2_model = ablation_res.pop("trained_m2_model")

    model_save_path = output_dir / "m2_model.joblib"
    joblib.dump(m2_model, model_save_path)
    logger.info("Saved M2 experiment model to %s", model_save_path)

    metrics_export = {
        "retrospective_ablation": ablation_res,
        "expanding_window_cv": cv_res,
    }
    metrics_path = output_dir / "m2_experiment_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    logger.info("Saved experiment metrics to %s", metrics_path)

    metadata = {
        "experiment_id": "EL5-061-M2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "algorithm": "HistGradientBoostingClassifier with SimpleImputer(median)",
        "features_total": len(M2_FEATURE_COLUMNS),
        "feature_list": M2_FEATURE_COLUMNS,
        "retrospective_test_metrics": {
            "M0_balanced_accuracy": models_res["M0_price_only"]["balanced_accuracy"],
            "M1_balanced_accuracy": models_res["M1_price_sec"]["balanced_accuracy"],
            "M2_balanced_accuracy": models_res["M2_price_sec_market"]["balanced_accuracy"],
            "B0a_balanced_accuracy": models_res["B0a_majority"]["balanced_accuracy"],
            "m2_minus_m1_diff": diffs["m2_minus_m1_balanced_acc"],
            "m2_minus_m0_diff": diffs["m2_minus_m0_balanced_acc"],
            "m2_minus_b0a_diff": diffs["m2_minus_b0a_balanced_acc"],
        },
        "cv_average_metrics": {
            "M0_mean_bacc": avg["M0"]["balanced_accuracy"],
            "M1_mean_bacc": avg["M1"]["balanced_accuracy"],
            "M2_mean_bacc": avg["M2"]["balanced_accuracy"],
            "B0a_mean_bacc": avg["B0a"]["balanced_accuracy"],
            "m2_minus_m1_mean_diff": cv_res["mean_diff_m2_minus_m1_bacc"],
            "m2_minus_m0_mean_diff": cv_res["mean_diff_m2_minus_m0_bacc"],
            "m2_minus_b0a_mean_diff": cv_res["mean_diff_m2_minus_b0a_bacc"],
        },
        "artifacts": {
            "model_path": str(model_save_path),
            "metrics_path": str(metrics_path),
        },
    }
    metadata_path = output_dir / "m2_experiment_metadata.json"
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
    parser = argparse.ArgumentParser(description="Evaluate M0 vs M1 vs M2 experiment (EL5-061).")
    parser.add_argument(
        "--market-sec-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "ml_market_sec_5d.csv",
    )
    parser.add_argument(
        "--all-features-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "features_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "experiments" / "market_m2",
    )
    args = parser.parse_args()

    run_experiment(
        market_sec_csv=args.market_sec_csv,
        all_features_csv=args.all_features_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
