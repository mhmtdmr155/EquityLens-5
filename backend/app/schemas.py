"""EquityLens-5 Pydantic Schemas for API Serialization."""

from __future__ import annotations

import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """General application health status response."""

    status: str = Field(..., description="Application operational status")
    version: str = Field(..., description="API application version")
    environment: str = Field(..., description="Runtime environment name")


class DatabaseHealthResponse(BaseModel):
    """Database connectivity diagnostic response."""

    status: str = Field(..., description="Database status: healthy or unhealthy")
    connected: bool = Field(..., description="Whether database query succeeded")
    dialect: str = Field(..., description="SQL dialect in use (e.g., postgresql, sqlite)")
    error: str | None = Field(default=None, description="Diagnostic error message if disconnected")


class SecurityBase(BaseModel):
    """Base schema for security instrument data."""

    ticker: str
    exchange: str
    valid_from: datetime.date
    valid_to: datetime.date | None = None


class SecurityRead(SecurityBase):
    """Read schema for security records."""

    id: int
    company_id: int

    model_config = ConfigDict(from_attributes=True)


class CompanyBase(BaseModel):
    """Base schema for corporate entity data."""

    company_name: str
    sec_cik: str


class CompanyRead(CompanyBase):
    """Read schema for company records including associated securities."""

    id: int
    created_at: datetime.datetime
    securities: list[SecurityRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class PriceBarBase(BaseModel):
    """Base schema for daily OHLCV price bar."""

    provider: str
    feed: str
    session_date: datetime.date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class PriceBarRead(PriceBarBase):
    """Read schema for price bar records."""

    id: int
    security_id: int
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class PriceBarDetail(BaseModel):
    """Clean OHLCV detail representation for price query endpoints."""

    session_date: datetime.date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    model_config = ConfigDict(from_attributes=True)


class PriceSeriesResponse(BaseModel):
    """Response payload for stock price queries."""

    ticker: str
    company_name: str
    provider: str
    feed: str
    count: int
    bars: list[PriceBarDetail] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    """Response payload for machine learning 5-day direction predictions."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full legal corporate name")
    model_name: str = Field(..., description="Identified machine learning algorithm")
    model_version: str = Field(..., description="Model version tag")
    horizon_trading_days: int = Field(default=5, description="Forecast horizon in trading sessions")
    as_of_date: str = Field(..., description="Effective historical date of input features")
    predicted_direction: int = Field(
        ..., description="Predicted direction: 1 for up, 0 for flat/down"
    )
    probability_up: float = Field(
        ..., description="Model confidence score for positive direction [0.0 - 1.0]"
    )
    data_source: str = Field(..., description="Underlying price bar data source and feed")
    warning: str = Field(..., description="Regulatory, freshness, and uncalibrated risk warning")
