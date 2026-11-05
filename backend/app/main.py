"""EquityLens-5 FastAPI Application Entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.routes.companies import router as companies_router
from backend.app.routes.health import router as health_router
from backend.app.routes.predictions import router as predictions_router
from backend.app.routes.prices import router as prices_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "EquityLens-5 Financial Analysis & Machine Learning Decision Support System. "
        "Provides REST endpoints for corporate entities, security mappings, and market data."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for future React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routers
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(companies_router, prefix=settings.api_prefix)
app.include_router(prices_router, prefix=settings.api_prefix)
app.include_router(predictions_router, prefix=settings.api_prefix)


@app.get("/", tags=["Root"])
def root_info() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api_v1_health": f"{settings.api_prefix}/health",
        "api_v1_companies": f"{settings.api_prefix}/companies",
        "api_v1_prices": f"{settings.api_prefix}/prices/{{ticker}}",
        "api_v1_predictions": f"{settings.api_prefix}/predictions/{{ticker}}",
    }
