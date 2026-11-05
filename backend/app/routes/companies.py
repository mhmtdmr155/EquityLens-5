"""EquityLens-5 Company and Security Exploration Endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.database import get_db
from backend.app.models import Company
from backend.app.schemas import CompanyRead

router = APIRouter(prefix="/companies", tags=["Companies"])


@router.get("", response_model=list[CompanyRead], summary="List all tracked companies")
def list_companies(db: Annotated[Session, Depends(get_db)]) -> list[CompanyRead]:
    """Retrieve all corporate entities with their associated securities."""
    stmt = select(Company).options(selectinload(Company.securities)).order_by(Company.id)
    companies = db.execute(stmt).scalars().all()
    return list(companies)


@router.get(
    "/{company_id}",
    response_model=CompanyRead,
    summary="Get company details by ID",
)
def get_company(
    company_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> CompanyRead:
    """Retrieve single company details and associated securities by internal ID."""
    stmt = select(Company).options(selectinload(Company.securities)).where(Company.id == company_id)
    company = db.execute(stmt).scalars().first()

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with ID {company_id} not found.",
        )

    return company
