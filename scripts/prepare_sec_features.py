"""EquityLens-5 SEC Feature Preparation Script (EL5-060).

Merges SEC fundamental features with the 1,908-row ml_train_5d dataset under
strict Point-in-Time alignment (session_date > filed_date) and generates
data/derived/ml_sec_5d.csv.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.sec_fundamentals import (
    SEC_FEATURE_COLUMNS,
    align_sec_fundamentals_point_in_time,
    load_cached_company_filings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EquityLens5.PrepareSECFeatures")


def prepare_sec_dataset(
    input_csv: Path = PROJECT_ROOT / "data" / "derived" / "ml_train_5d.csv",
    cache_dir: Path = PROJECT_ROOT / "data" / "sec_cache",
    output_csv: Path = PROJECT_ROOT / "data" / "derived" / "ml_sec_5d.csv",
) -> pd.DataFrame:
    """Build and save M1 dataset with point-in-time SEC fundamentals."""
    if not input_csv.is_file():
        msg = f"Input dataset not found: {input_csv}"
        raise FileNotFoundError(msg)

    if not cache_dir.is_dir():
        msg = f"SEC cache directory not found: {cache_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading baseline 5-day ML dataset from %s ...", input_csv)
    df_base = pd.read_csv(input_csv)
    original_row_count = len(df_base)
    logger.info(
        "Baseline dataset loaded with %d rows, %d columns.",
        original_row_count,
        len(df_base.columns),
    )

    logger.info("Loading cached SEC filings from %s ...", cache_dir)
    filings_by_symbol = load_cached_company_filings(cache_dir)
    for sym, filings in filings_by_symbol.items():
        logger.info("  %s: %d filings parsed", sym, len(filings))

    logger.info("Performing point-in-time SEC alignment (session_date > filed) ...")
    df_sec = align_sec_fundamentals_point_in_time(df_base, filings_by_symbol)

    # 1. Structural Validation
    if len(df_sec) != original_row_count:
        msg = f"Row count mismatch! Expected {original_row_count}, got {len(df_sec)}"
        raise ValueError(msg)

    # 2. Strict Point-in-Time Look-Ahead Validation
    leak_violations = 0
    for idx, row in df_sec.iterrows():
        filed_d = row.get("sec_filed_date")
        sess_d = str(row["session_date"])
        if pd.notna(filed_d) and str(filed_d) >= sess_d:
            logger.error(
                "LOOK-AHEAD VIOLATION at row %d (%s): filed %s >= session %s",
                idx,
                row["ticker"],
                filed_d,
                sess_d,
            )
            leak_violations += 1

    if leak_violations > 0:
        msg = f"Validation failed with {leak_violations} look-ahead violations!"
        raise ValueError(msg)

    logger.info(
        "Point-in-time look-ahead validation PASSED: 0 violations across %d rows.",
        len(df_sec),
    )

    # 3. Missingness and Coverage Reporting
    total_rows = len(df_sec)
    logger.info("\n--- SEC Feature Coverage & Missingness ---")
    for col in SEC_FEATURE_COLUMNS:
        null_count = int(df_sec[col].isna().sum())
        non_null = total_rows - null_count
        pct_missing = round(null_count / total_rows * 100, 2)
        pct_coverage = round(non_null / total_rows * 100, 2)
        logger.info(
            "  %30s: %4d / %4d non-null (coverage: %6.2f%%, missing: %5.2f%%)",
            col,
            non_null,
            total_rows,
            pct_coverage,
            pct_missing,
        )

    logger.info("\n--- Symbol Breakdown ---")
    for sym, grp in df_sec.groupby("ticker"):
        n_sym = len(grp)
        assets_null = int(grp["sec_total_assets"].isna().sum())
        assets_avail = n_sym - assets_null
        logger.info(
            "  %5s: total=%4d, SEC available=%4d (%.1f%%), SEC missing=%2d (%.1f%%)",
            sym,
            n_sym,
            assets_avail,
            assets_avail / n_sym * 100,
            assets_null,
            assets_null / n_sym * 100,
        )

    # 4. Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sec.to_csv(output_csv, index=False)
    logger.info(
        "Successfully saved M1 dataset to %s (%d rows, %d cols).",
        output_csv,
        len(df_sec),
        len(df_sec.columns),
    )

    return df_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SEC fundamental features dataset (M1).")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "ml_train_5d.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sec_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "ml_sec_5d.csv",
    )
    args = parser.parse_args()

    prepare_sec_dataset(input_csv=args.input, cache_dir=args.cache_dir, output_csv=args.output)


if __name__ == "__main__":
    main()
