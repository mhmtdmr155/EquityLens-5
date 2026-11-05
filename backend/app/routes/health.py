"""EquityLens-5 Health Check API Endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.database import check_database_health, get_db
from backend.app.schemas import DatabaseHealthResponse, HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Application operational status")
def get_app_health() -> HealthResponse:
    """Return application process health and metadata."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get(
    "/db",
    response_model=DatabaseHealthResponse,
    summary="Database connectivity diagnostic",
)
def get_db_health(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> DatabaseHealthResponse:
    """Probe database connectivity and report real operational status.

    If the database is unreachable, responds with HTTP 503 Service Unavailable
    to ensure callers are not misled into believing all services are healthy.
    """
    health = check_database_health(db)
    if not health["connected"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return DatabaseHealthResponse(
        status=health["status"],
        connected=health["connected"],
        dialect=health["dialect"],
        error=health["error"],
    )
