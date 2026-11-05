"""EquityLens-5 Historical Stock Price Query API Route."""

from __future__ import annotations

import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from backend.app.database import get_db
from backend.app.models import PriceBar, Security
from backend.app.schemas import PriceBarDetail, PriceSeriesResponse

router = APIRouter(prefix="/prices", tags=["Prices"])


@router.get(
    "/{ticker}",
    response_model=PriceSeriesResponse,
    summary="Query historical daily OHLCV price bars for a symbol",
)
def get_historical_prices(
    ticker: str,
    db: Annotated[Session, Depends(get_db)],
    start_date: Annotated[
        datetime.date | None,
        Query(description="Start date filter (YYYY-MM-DD)"),
    ] = None,
    end_date: Annotated[
        datetime.date | None,
        Query(description="End date filter (YYYY-MM-DD)"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of bars to return"),
    ] = 100,
) -> PriceSeriesResponse:
    """Retrieve historical daily price bars for a tracked company ticker.

    Supports optional date filtering (start_date, end_date) and pagination limit.
    Returns bars sorted in chronological order (session_date ascending).
    """
    clean_ticker = ticker.upper().strip()

    # Parameter validation
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Invalid date range: start_date '{start_date}' "
                f"cannot be after end_date '{end_date}'."
            ),
        )

    # Resolve Security entity
    sec_stmt = (
        select(Security)
        .options(joinedload(Security.company))
        .where(Security.ticker == clean_ticker)
    )
    security = db.execute(sec_stmt).scalars().first()

    if security is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{clean_ticker}' not found.",
        )

    # Query price bars
    bars_stmt = select(PriceBar).where(PriceBar.security_id == security.id)
    if start_date:
        bars_stmt = bars_stmt.where(PriceBar.session_date >= start_date)
    if end_date:
        bars_stmt = bars_stmt.where(PriceBar.session_date <= end_date)

    bars_stmt = bars_stmt.order_by(PriceBar.session_date.asc()).limit(limit)
    bars = list(db.execute(bars_stmt).scalars().all())

    # Build response
    provider = bars[0].provider if bars else "alpaca"
    feed = bars[0].feed if bars else "iex"
    company_name = security.company.company_name if security.company else clean_ticker

    return PriceSeriesResponse(
        ticker=security.ticker,
        company_name=company_name,
        provider=provider,
        feed=feed,
        count=len(bars),
        bars=[PriceBarDetail.model_validate(b) for b in bars],
    )
