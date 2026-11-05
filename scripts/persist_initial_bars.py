"""EquityLens-5 Initial Price Data Persistence Runner (EL5-021).

Fetches sample daily OHLCV bars (2026-10-19 to 2026-10-23, feed='iex')
from Alpaca Market Data for the 5 tracked companies (AAPL, NVDA, MU, DELL, SPCX),
validates all records, resolves securities, and idempotently stores them.
"""

# ruff: noqa: E402
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path for script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import get_settings
from backend.app.ingestion import persist_symbol_bars
from backend.app.models import Base, PriceBar
from backend.app.seed import seed_companies
from scripts.market_data_probe import (
    EQUITYLENS_SYMBOLS,
    SPACEX_IPO_DATE_STR,
    AlpacaConfig,
    AlpacaDataClient,
    AuthenticationError,
    MarketDataError,
)

logger = logging.getLogger("EquityLens5.PersistenceRunner")


def run_persistence_pipeline(
    db: Session,
    client: AlpacaDataClient,
    symbols: tuple[str, ...] = EQUITYLENS_SYMBOLS,
    start_date: str = "2026-10-19",
    end_date: str = "2026-10-23",
    feed: str = "iex",
) -> dict[str, Any]:
    """Execute live or mock fetch and persist pipeline with strict validation."""
    summary: dict[str, Any] = {
        "symbols_processed": [],
        "total_bars_fetched": 0,
        "total_bars_inserted": 0,
        "total_bars_skipped": 0,
        "symbol_details": {},
    }

    for sym in symbols:
        query_start = start_date
        # Strict boundary for SPCX
        if sym.upper() == "SPCX" and query_start < SPACEX_IPO_DATE_STR:
            query_start = SPACEX_IPO_DATE_STR

        try:
            raw_response = client.get_stock_bars(
                symbols=[sym],
                timeframe="1Day",
                start=query_start,
                end=end_date,
                feed=feed,
                limit=5,
            )
            bars = raw_response.get("bars", {}).get(sym, [])
            res = persist_symbol_bars(db, sym, bars, provider="alpaca", feed=feed)

            summary["symbols_processed"].append(sym)
            summary["total_bars_fetched"] += len(bars)
            summary["total_bars_inserted"] += res["bars_inserted"]
            summary["total_bars_skipped"] += res["bars_skipped_duplicate"]
            summary["symbol_details"][sym] = res

        except (AuthenticationError, MarketDataError, Exception) as exc:
            logger.error("Failed persisting bars for %s: %s", sym, exc)
            summary["symbol_details"][sym] = {"status": "ERROR", "error": str(exc)}

    return summary


def main() -> int:
    """Run initial price bar persistence flow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("EquityLens-5 Initial Price Data Persistence Runner (EL5-021)")

    # 1. Initialize client
    try:
        config = AlpacaConfig.from_env()
        client = AlpacaDataClient(config)
    except AuthenticationError as exc:
        logger.error("Alpaca credentials missing or invalid: %s", exc)
        return 1

    # 2. Initialize database
    settings = get_settings()
    is_sqlite = settings.database_url.startswith("sqlite")
    logger.info("Target database dialect: %s", "sqlite" if is_sqlite else "postgresql")

    try:
        engine = create_engine(settings.database_url)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)
        return 1

    with session_factory() as db:
        # Seed companies
        seed_companies(db)
        logger.info("Companies and securities seeded successfully.")

        # First run: Ingest initial bars
        res1 = run_persistence_pipeline(db, client)
        logger.info(
            "First ingestion run: Fetched %d bars, Inserted %d bars, Skipped %d duplicates.",
            res1["total_bars_fetched"],
            res1["total_bars_inserted"],
            res1["total_bars_skipped"],
        )

        # Second run: Verify idempotency
        res2 = run_persistence_pipeline(db, client)
        logger.info(
            "Second ingestion run (idempotency test): Fetched %d bars, "
            "Inserted %d bars, Skipped %d duplicates.",
            res2["total_bars_fetched"],
            res2["total_bars_inserted"],
            res2["total_bars_skipped"],
        )

        # Final audit
        total_in_db = db.execute(select(func.count(PriceBar.id))).scalar_one()
        logger.info("Total price_bars in database: %d", total_in_db)

    return 0


if __name__ == "__main__":
    sys.exit(main())
