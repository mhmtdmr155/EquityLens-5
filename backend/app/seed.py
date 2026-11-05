"""EquityLens-5 Database Seeding Mechanism for Core Companies.

Seeds the initial 5 tracked companies and their primary securities:
1. Apple Inc. (AAPL - NASDAQ) - SEC CIK 0000320193
2. NVIDIA Corporation (NVDA - NASDAQ) - SEC CIK 0001045810
3. Micron Technology, Inc. (MU - NASDAQ) - SEC CIK 0000723125
4. Dell Technologies Inc. (DELL - NYSE) - SEC CIK 0001571996
5. Space Exploration Technologies Corp. (SPCX - NASDAQ) - SEC CIK 0001181412
   (Strictly enforced SpaceX IPO validity date: 2026-06-12)

Designed to be completely idempotent: multiple executions will not produce duplicates.
"""

from __future__ import annotations

import datetime
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Company, Security

logger = logging.getLogger("EquityLens5.Seed")

CORE_COMPANIES_SEED = [
    {
        "company_name": "Apple Inc.",
        "sec_cik": "0000320193",
        "ticker": "AAPL",
        "exchange": "NASDAQ",
        "valid_from": datetime.date(1980, 12, 12),
        "valid_to": None,
    },
    {
        "company_name": "NVIDIA Corporation",
        "sec_cik": "0001045810",
        "ticker": "NVDA",
        "exchange": "NASDAQ",
        "valid_from": datetime.date(1999, 1, 22),
        "valid_to": None,
    },
    {
        "company_name": "Micron Technology, Inc.",
        "sec_cik": "0000723125",
        "ticker": "MU",
        "exchange": "NASDAQ",
        "valid_from": datetime.date(1984, 6, 1),
        "valid_to": None,
    },
    {
        "company_name": "Dell Technologies Inc.",
        "sec_cik": "0001571996",
        "ticker": "DELL",
        "exchange": "NYSE",
        "valid_from": datetime.date(2018, 12, 28),
        "valid_to": None,
    },
    {
        "company_name": "Space Exploration Technologies Corp.",
        "sec_cik": "0001181412",
        "ticker": "SPCX",
        "exchange": "NASDAQ",
        "valid_from": datetime.date(2026, 6, 12),
        "valid_to": None,
    },
]


def seed_companies(db: Session) -> list[Company]:
    """Idempotently seed the 5 tracked companies and their securities into the database."""
    results: list[Company] = []

    for entry in CORE_COMPANIES_SEED:
        # Check if company exists by SEC CIK or name
        stmt = select(Company).where(
            (Company.sec_cik == entry["sec_cik"]) | (Company.company_name == entry["company_name"])
        )
        company = db.execute(stmt).scalars().first()

        if company is None:
            company = Company(
                company_name=entry["company_name"],
                sec_cik=entry["sec_cik"],
            )
            db.add(company)
            db.flush()
            logger.info(
                "Created company record: %s (CIK: %s)",
                company.company_name,
                company.sec_cik,
            )

        # Check if security exists for this company
        sec_stmt = select(Security).where(
            Security.company_id == company.id,
            Security.ticker == entry["ticker"],
            Security.exchange == entry["exchange"],
        )
        security = db.execute(sec_stmt).scalars().first()

        if security is None:
            security = Security(
                company_id=company.id,
                ticker=entry["ticker"],
                exchange=entry["exchange"],
                valid_from=entry["valid_from"],
                valid_to=entry["valid_to"],
            )
            db.add(security)
            db.flush()
            logger.info(
                "Created security record: %s on %s for %s",
                security.ticker,
                security.exchange,
                company.company_name,
            )

        results.append(company)

    db.commit()
    # Refresh objects to ensure relationships are accessible
    for c in results:
        db.refresh(c)

    return results
