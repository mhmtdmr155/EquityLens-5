"""Script to generate M2 dataset with market and sector ETF features (EL5-061).

Merges 9 ETF market/sector features with ml_sec_5d.csv to produce
data/derived/ml_market_sec_5d.csv (1,908 rows).
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.market_features import (
    MARKET_FEATURE_COLUMNS,
    compute_market_etf_features,
    merge_market_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EquityLens5.PrepareMarketFeatures")


def prepare_market_dataset(
    sec_csv: Path = PROJECT_ROOT / "data" / "derived" / "ml_sec_5d.csv",
    market_cache_dir: Path = PROJECT_ROOT / "data" / "market_cache",
    output_csv: Path = PROJECT_ROOT / "data" / "derived" / "ml_market_sec_5d.csv",
) -> pd.DataFrame:
    """Build and save M2 dataset with market/sector features."""
    if not sec_csv.is_file():
        msg = f"Input M1 dataset not found: {sec_csv}"
        raise FileNotFoundError(msg)

    if not market_cache_dir.is_dir():
        msg = f"Market cache directory not found: {market_cache_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading M1 dataset from %s ...", sec_csv)
    df_sec = pd.read_csv(sec_csv)
    n_original = len(df_sec)
    logger.info("Loaded %d rows from M1 dataset.", n_original)

    logger.info("Computing 9 market and sector features from %s ...", market_cache_dir)
    market_df = compute_market_etf_features(market_cache_dir)
    logger.info(
        "Market features computed for %d distinct session dates (%s to %s).",
        len(market_df),
        market_df["session_date"].min(),
        market_df["session_date"].max(),
    )

    logger.info("Merging market features into panel dataset ...")
    df_m2 = merge_market_features(df_sec, market_df)

    # 1. Structural Validation
    if len(df_m2) != n_original:
        msg = f"Row count mismatch! Expected {n_original}, got {len(df_m2)}"
        raise ValueError(msg)

    # 2. Target Integrity Validation
    pos_count = int((df_m2["direction_5d"] == 1.0).sum())
    neg_count = int((df_m2["direction_5d"] == 0.0).sum())
    if pos_count != 1107 or neg_count != 801:
        msg = f"Target integrity violated! Expected 1107/801, got {pos_count}/{neg_count}"
        raise ValueError(msg)
    logger.info(
        "Target integrity PASSED: 1,107 positive / 801 negative (58.02%% positive ratio).",
    )

    # 3. Missingness check for market features
    logger.info("\n--- Market & Sector Feature Coverage ---")
    for col in MARKET_FEATURE_COLUMNS:
        null_count = int(df_m2[col].isna().sum())
        coverage_pct = round((n_original - null_count) / n_original * 100, 2)
        logger.info(
            "  %25s: %4d / %4d non-null (coverage: %6.2f%%, nulls: %d)",
            col,
            n_original - null_count,
            n_original,
            coverage_pct,
            null_count,
        )

    # 4. Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_m2.to_csv(output_csv, index=False)
    logger.info(
        "Successfully saved M2 dataset to %s (%d rows, %d columns).",
        output_csv,
        len(df_m2),
        len(df_m2.columns),
    )

    return df_m2


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare M2 dataset with market/sector features.")
    parser.add_argument(
        "--sec-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "ml_sec_5d.csv",
    )
    parser.add_argument(
        "--market-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "market_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "ml_market_sec_5d.csv",
    )
    args = parser.parse_args()

    prepare_market_dataset(
        sec_csv=args.sec_csv,
        market_cache_dir=args.market_cache,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
