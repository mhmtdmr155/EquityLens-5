"""EquityLens-5 Historical Market Data Backfill Runner (EL5-030).

Fetches historical daily OHLCV bars (feed='iex', adjustment='raw') from Alpaca
for the 5 tracked companies with strict pagination, loop prevention, and SpaceX IPO boundary:
- AAPL, NVDA, MU, DELL: 2024-11-01 to 2026-10-30
- SPCX: 2026-06-12 to 2026-10-30 (SpaceX IPO boundary strictly enforced)

Stores verified bars in the local SQLite development database (data/equitylens_dev.db).
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from backend.app.config import get_settings
from backend.app.ingestion import run_historical_backfill
from backend.app.models import Base, PriceBar, Security
from backend.app.seed import seed_companies
from scripts.market_data_probe import (
    AlpacaConfig,
    AlpacaDataClient,
    AuthenticationError,
)

logger = logging.getLogger("EquityLens5.BackfillRunner")


def audit_database_contents(db) -> dict[str, Any]:
    """Inspect stored price bars across all tracked securities."""
    stmt = (
        select(
            Security.ticker,
            func.count(PriceBar.id).label("total_bars"),
            func.min(PriceBar.session_date).label("earliest_date"),
            func.max(PriceBar.session_date).label("latest_date"),
        )
        .join(PriceBar, Security.id == PriceBar.security_id)
        .group_by(Security.ticker)
        .order_by(Security.ticker)
    )
    rows = db.execute(stmt).all()

    audit_res = {}
    for r in rows:
        audit_res[r.ticker] = {
            "total_bars": r.total_bars,
            "earliest_date": str(r.earliest_date),
            "latest_date": str(r.latest_date),
        }

    total_all = db.execute(select(func.count(PriceBar.id))).scalar_one()
    audit_res["_TOTAL_"] = total_all
    return audit_res


def main() -> int:
    """Execute controlled historical backfill pipeline."""
    parser = argparse.ArgumentParser(description="EquityLens-5 Historical Data Backfill")
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="Run pilot range (2 weeks) or full target range (2024-11 to 2026-10)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("EquityLens-5 Historical Data Backfill (EL5-030) - Mode: %s", args.mode.upper())

    # 1. Credentials
    try:
        config = AlpacaConfig.from_env()
        client = AlpacaDataClient(config)
    except AuthenticationError as exc:
        logger.error("Alpaca credentials missing or invalid: %s", exc)
        return 1

    # 2. Database setup
    settings = get_settings()
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    logger.info("Connecting to database: %s", settings.masked_database_url)
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # 3. Determine date ranges
    if args.mode == "pilot":
        # 2-week pilot verification
        ranges = {
            "AAPL": ("2026-10-12", "2026-10-23"),
            "NVDA": ("2026-10-12", "2026-10-23"),
            "MU": ("2026-10-12", "2026-10-23"),
            "DELL": ("2026-10-12", "2026-10-23"),
            "SPCX": ("2026-10-12", "2026-10-23"),
        }
    else:
        # Full project target ranges
        ranges = {
            "AAPL": ("2024-11-01", "2026-10-30"),
            "NVDA": ("2024-11-01", "2026-10-30"),
            "MU": ("2024-11-01", "2026-10-30"),
            "DELL": ("2024-11-01", "2026-10-30"),
            "SPCX": ("2026-06-12", "2026-10-30"),
        }

    with session_factory() as db:
        # Seed companies
        seed_companies(db)
        logger.info("Core companies seeded.")

        # Run backfill
        logger.info("Starting historical data fetch from Alpaca (feed='iex', adjustment='raw')...")
        summary = run_historical_backfill(
            db=db,
            client=client,
            date_ranges=ranges,
            feed="iex",
            max_total_requests=50,
        )

        logger.info(
            "Backfill completed: %d total requests, %d bars fetched, "
            "%d inserted, %d skipped duplicates.",
            summary["total_requests_made"],
            summary["total_bars_fetched"],
            summary["total_bars_inserted"],
            summary["total_bars_skipped"],
        )

        # Audit database state
        audit = audit_database_contents(db)
        logger.info("--- DATABASE AUDIT SUMMARY ---")
        for sym, details in audit.items():
            if sym != "_TOTAL_":
                logger.info(
                    "  %s: %d bars (range: %s to %s)",
                    sym,
                    details["total_bars"],
                    details["earliest_date"],
                    details["latest_date"],
                )
        logger.info("Total price bars in database: %d", audit["_TOTAL_"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
