"""EquityLens-5 ML Dataset Preparation Pipeline (EL5-040).

Extracts raw daily OHLCV price bars from SQLite database (data/equitylens_dev.db),
computes 23 technical indicators, generates forward-looking 5-day and 20-day
classification targets, strictly separates training from prediction-ready sets,
and exports reproducible datasets to data/derived/.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.features import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RAW_PRICE_COLUMNS,
    TARGET_COLUMNS,
    build_feature_pipeline,
    split_train_and_prediction_sets,
)

logger = logging.getLogger("EquityLens5.DatasetPrep")


def load_raw_data_from_sqlite(db_path: Path) -> pd.DataFrame:
    """Load raw daily OHLCV bars for all tracked securities from SQLite database."""
    if not db_path.exists():
        msg = f"Database file not found: {db_path}"
        raise FileNotFoundError(msg)

    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT
            s.ticker,
            c.company_name,
            p.session_date,
            CAST(p.open AS REAL) AS open,
            CAST(p.high AS REAL) AS high,
            CAST(p.low AS REAL) AS low,
            CAST(p.close AS REAL) AS close,
            CAST(p.volume AS INTEGER) AS volume,
            p.provider,
            p.feed
        FROM price_bars p
        JOIN securities s ON s.id = p.security_id
        JOIN companies c ON c.id = s.company_id
        ORDER BY s.ticker ASC, p.session_date ASC
    """
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    logger.info(
        "Loaded %d raw price bars across %d symbols from %s",
        len(df),
        df["ticker"].nunique(),
        db_path,
    )
    return df


def generate_dataset_metadata(
    splits: dict[str, pd.DataFrame],
    source_db: Path,
) -> dict[str, Any]:
    """Generate comprehensive audit metadata for exported datasets."""
    features_all = splits["features_all"]
    train_5d = splits["ml_train_5d"]
    train_20d = splits["ml_train_20d"]
    pred_ready = splits["prediction_ready"]

    symbol_stats: dict[str, Any] = {}
    tickers = sorted(features_all["ticker"].unique())

    for sym in tickers:
        f_sym = features_all[features_all["ticker"] == sym]
        t5_sym = train_5d[train_5d["ticker"] == sym]
        t20_sym = train_20d[train_20d["ticker"] == sym]
        pr_sym = pred_ready[pred_ready["ticker"] == sym]

        symbol_stats[sym] = {
            "total_raw_bars": int(len(f_sym)),
            "start_date": str(f_sym["session_date"].min()),
            "end_date": str(f_sym["session_date"].max()),
            "train_5d_samples": int(len(t5_sym)),
            "train_20d_samples": int(len(t20_sym)),
            "prediction_ready_samples": int(len(pr_sym)),
            "target_5d_positive_ratio": round(float((t5_sym["direction_5d"] == 1.0).mean()), 4)
            if len(t5_sym) > 0
            else None,
            "target_20d_positive_ratio": round(float((t20_sym["direction_20d"] == 1.0).mean()), 4)
            if len(t20_sym) > 0
            else None,
        }

    # Class balance summary
    cb_5d_pos = (
        round(float((train_5d["direction_5d"] == 1.0).mean()), 4) if len(train_5d) > 0 else 0.0
    )
    cb_20d_pos = (
        round(float((train_20d["direction_20d"] == 1.0).mean()), 4) if len(train_20d) > 0 else 0.0
    )

    metadata = {
        "dataset_name": "EquityLens-5 Technical Features & ML Targets",
        "dataset_version": "1.0.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_database": str(source_db.resolve()),
        "data_provider": "alpaca",
        "data_feed": "iex",
        "price_adjustment": "raw",
        "corporate_actions_note": (
            "Prices are raw (unadjusted for splits and cash dividends). "
            "One diagnostic jump observed (DELL on 2026-05-29 at +32.75% due to earnings). "
            "No unadjusted stock split distortions identified within 2024-11 to 2026-10 window."
        ),
        "total_symbols": len(tickers),
        "symbols": tickers,
        "spacex_ipo_constraint": (
            "SPCX data strictly constrained to post-IPO sessions (>= 2026-06-12)."
        ),
        "column_specifications": {
            "metadata_columns": METADATA_COLUMNS,
            "raw_price_columns": RAW_PRICE_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "total_feature_count": len(FEATURE_COLUMNS),
            "total_target_count": len(TARGET_COLUMNS),
        },
        "row_counts": {
            "features_all": int(len(features_all)),
            "ml_train_5d": int(len(train_5d)),
            "ml_train_20d": int(len(train_20d)),
            "prediction_ready": int(len(pred_ready)),
        },
        "target_class_balance": {
            "direction_5d": {
                "positive_label_1": cb_5d_pos,
                "non_positive_label_0": round(1.0 - cb_5d_pos, 4),
            },
            "direction_20d": {
                "positive_label_1": cb_20d_pos,
                "non_positive_label_0": round(1.0 - cb_20d_pos, 4),
            },
        },
        "symbol_breakdown": symbol_stats,
    }

    return metadata


def run_dataset_preparation(db_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute complete dataset extraction, computation, and file persistence."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_df = load_raw_data_from_sqlite(db_path)

    # 2. Build feature and target pipeline
    logger.info("Computing technical features and targets across securities...")
    pipeline_df = build_feature_pipeline(raw_df)

    # 3. Partition into training sets and prediction-ready set
    splits = split_train_and_prediction_sets(pipeline_df)

    # 4. Save CSV files
    all_path = output_dir / "features_all.csv"
    train_5d_path = output_dir / "ml_train_5d.csv"
    train_20d_path = output_dir / "ml_train_20d.csv"
    pred_path = output_dir / "prediction_ready.csv"
    meta_path = output_dir / "dataset_metadata.json"

    logger.info("Saving features_all.csv (%d rows)...", len(splits["features_all"]))
    splits["features_all"].to_csv(all_path, index=False, float_format="%.6f")

    logger.info("Saving ml_train_5d.csv (%d rows)...", len(splits["ml_train_5d"]))
    splits["ml_train_5d"].to_csv(train_5d_path, index=False, float_format="%.6f")

    logger.info("Saving ml_train_20d.csv (%d rows)...", len(splits["ml_train_20d"]))
    splits["ml_train_20d"].to_csv(train_20d_path, index=False, float_format="%.6f")

    logger.info("Saving prediction_ready.csv (%d rows)...", len(splits["prediction_ready"]))
    splits["prediction_ready"].to_csv(pred_path, index=False, float_format="%.6f")

    # 5. Generate and save metadata JSON
    metadata = generate_dataset_metadata(splits, db_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata JSON to %s", meta_path)

    return metadata


def main() -> int:
    """CLI entrypoint for ML dataset preparation."""
    parser = argparse.ArgumentParser(description="EquityLens-5 ML Dataset Preparation Pipeline")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "equitylens_dev.db",
        help="Path to SQLite development database.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived",
        help="Output directory for derived CSVs and metadata.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting EquityLens-5 ML Dataset Preparation Pipeline...")

    try:
        meta = run_dataset_preparation(args.db_path, args.out_dir)
        logger.info("Dataset preparation successfully completed.")
        logger.info(
            "Summary: total_raw=%d, train_5d=%d, train_20d=%d, prediction_ready=%d",
            meta["row_counts"]["features_all"],
            meta["row_counts"]["ml_train_5d"],
            meta["row_counts"]["ml_train_20d"],
            meta["row_counts"]["prediction_ready"],
        )
        return 0
    except Exception as exc:
        logger.exception("Dataset preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
