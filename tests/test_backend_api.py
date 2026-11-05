"""EquityLens-5 FastAPI Backend Integration Tests (EL5-020).

Tests API endpoints:
- GET /
- GET /api/v1/health
- GET /api/v1/health/db
- GET /api/v1/companies
- GET /api/v1/companies/{id}

Ensures proper status codes, payload serialization, error handling,
and guarantees that no credentials or secrets are leaked in responses.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Generator
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import get_db
from backend.app.main import app
from backend.app.models import Base, PriceBar, Security
from backend.app.seed import seed_companies


@pytest.fixture
def client_with_seeded_db() -> Generator[TestClient, None, None]:
    """Provide a TestClient connected to an in-memory SQLite database seeded with 5 companies."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Seed the database
    with session_factory() as db:
        seed_companies(db)
        aapl_sec = db.execute(select(Security).where(Security.ticker == "AAPL")).scalars().first()
        if aapl_sec:
            b1 = PriceBar(
                security_id=aapl_sec.id,
                provider="alpaca",
                feed="iex",
                session_date=datetime.date(2026, 10, 20),
                open=Decimal("230.00"),
                high=Decimal("235.00"),
                low=Decimal("229.00"),
                close=Decimal("234.00"),
                volume=1000000,
            )
            b2 = PriceBar(
                security_id=aapl_sec.id,
                provider="alpaca",
                feed="iex",
                session_date=datetime.date(2026, 10, 21),
                open=Decimal("234.00"),
                high=Decimal("238.00"),
                low=Decimal("233.00"),
                close=Decimal("237.50"),
                volume=1200000,
            )
            db.add_all([b1, b2])
            db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def test_root_endpoint(client_with_seeded_db: TestClient) -> None:
    """Verify that the root endpoint returns service metadata."""
    response = client_with_seeded_db.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "EquityLens-5" in data["service"]
    assert data["version"] == "0.1.0"
    assert "/docs" in data["docs"]


def test_health_endpoint(client_with_seeded_db: TestClient) -> None:
    """Verify that the general health check endpoint returns 200 OK."""
    response = client_with_seeded_db.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "development"


def test_health_db_endpoint_healthy(client_with_seeded_db: TestClient) -> None:
    """Verify that the database health check reports connected when operational."""
    response = client_with_seeded_db.get("/api/v1/health/db")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["connected"] is True
    assert data["dialect"] == "sqlite"
    assert data["error"] is None


def test_health_db_endpoint_unhealthy() -> None:
    """Verify that an unreachable database returns HTTP 503 and accurate failure status."""
    mock_db = MagicMock()
    mock_db.bind.dialect.name = "postgresql"
    mock_db.execute.side_effect = OperationalError("Connection refused", {}, None)

    def broken_get_db() -> Generator[Session, None, None]:
        yield mock_db

    app.dependency_overrides[get_db] = broken_get_db

    with TestClient(app) as client:
        response = client.get("/api/v1/health/db")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["connected"] is False
        assert "unreachable" in data["error"].lower()

    app.dependency_overrides.clear()


def test_get_companies_list(client_with_seeded_db: TestClient) -> None:
    """Verify listing all 5 seeded companies with their securities."""
    response = client_with_seeded_db.get("/api/v1/companies")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5

    tickers = {c["securities"][0]["ticker"] for c in data if c.get("securities")}
    assert tickers == {"AAPL", "NVDA", "MU", "DELL", "SPCX"}


def test_get_company_by_id_success(client_with_seeded_db: TestClient) -> None:
    """Verify retrieving a single company by its ID."""
    # First get the list to obtain valid IDs
    list_resp = client_with_seeded_db.get("/api/v1/companies")
    first_company = list_resp.json()[0]
    target_id = first_company["id"]

    response = client_with_seeded_db.get(f"/api/v1/companies/{target_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == target_id
    assert data["company_name"] == first_company["company_name"]
    assert data["sec_cik"] == first_company["sec_cik"]
    assert len(data["securities"]) >= 1


def test_get_company_by_id_not_found(client_with_seeded_db: TestClient) -> None:
    """Verify that querying a non-existent company returns HTTP 404."""
    response = client_with_seeded_db.get("/api/v1/companies/999999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_spacex_company_and_security_details(client_with_seeded_db: TestClient) -> None:
    """Verify that SpaceX details and strict 2026-06-12 IPO boundary are surfaced correctly."""
    response = client_with_seeded_db.get("/api/v1/companies")
    assert response.status_code == 200
    companies = response.json()

    spacex = next(
        (c for c in companies if c["sec_cik"] == "0001181412"),
        None,
    )
    assert spacex is not None
    assert "Space Exploration Technologies" in spacex["company_name"]
    assert len(spacex["securities"]) == 1

    security = spacex["securities"][0]
    assert security["ticker"] == "SPCX"
    assert security["exchange"] == "NASDAQ"
    assert security["valid_from"] == "2026-06-12"
    assert security["valid_to"] is None


def test_api_responses_contain_no_secrets(client_with_seeded_db: TestClient) -> None:
    """Verify that API payloads never leak passwords, raw DB URLs, or secret keys."""
    endpoints = [
        "/",
        "/api/v1/health",
        "/api/v1/health/db",
        "/api/v1/companies",
        "/api/v1/companies/1",
        "/api/v1/prices/AAPL",
    ]
    forbidden_terms = [
        "secret",
        "password",
        "api_secret_key",
        "apca_api_secret_key",
        "postgres:postgres",
    ]

    for path in endpoints:
        response = client_with_seeded_db.get(path)
        payload_str = json.dumps(response.json()).lower()
        for forbidden in forbidden_terms:
            assert forbidden not in payload_str, (
                f"Endpoint '{path}' leaked sensitive term '{forbidden}' in payload: {payload_str}"
            )


def test_get_prices_endpoint_success(client_with_seeded_db: TestClient) -> None:
    """Verify querying prices for a valid ticker returns bars in ascending order."""
    response = client_with_seeded_db.get("/api/v1/prices/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["company_name"] == "Apple Inc."
    assert data["provider"] == "alpaca"
    assert data["feed"] == "iex"
    assert data["count"] == 2
    assert len(data["bars"]) == 2
    assert data["bars"][0]["session_date"] == "2026-10-20"
    assert data["bars"][1]["session_date"] == "2026-10-21"


def test_get_prices_endpoint_date_filtering(client_with_seeded_db: TestClient) -> None:
    """Verify start_date and end_date query parameters correctly filter price bars."""
    # Filter only second date
    resp1 = client_with_seeded_db.get("/api/v1/prices/AAPL?start_date=2026-10-21")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["count"] == 1
    assert data1["bars"][0]["session_date"] == "2026-10-21"

    # Filter only first date
    resp2 = client_with_seeded_db.get("/api/v1/prices/AAPL?end_date=2026-10-20")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["count"] == 1
    assert data2["bars"][0]["session_date"] == "2026-10-20"


def test_get_prices_endpoint_unknown_ticker_404(client_with_seeded_db: TestClient) -> None:
    """Verify querying prices for an unknown ticker returns HTTP 404."""
    response = client_with_seeded_db.get("/api/v1/prices/NONEXISTENT")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_get_prices_endpoint_invalid_date_range_422(client_with_seeded_db: TestClient) -> None:
    """Verify that start_date > end_date returns HTTP 422 Unprocessable Content."""
    response = client_with_seeded_db.get(
        "/api/v1/prices/AAPL?start_date=2026-10-25&end_date=2026-10-20"
    )
    assert response.status_code == 422
    data = response.json()
    assert "cannot be after end_date" in data["detail"]


def test_get_prices_endpoint_empty_results(client_with_seeded_db: TestClient) -> None:
    """Verify querying date range with no bars returns 200 with empty list."""
    response = client_with_seeded_db.get(
        "/api/v1/prices/AAPL?start_date=2025-01-01&end_date=2025-01-05"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["bars"] == []


def test_get_prices_endpoint_order_desc(client_with_seeded_db: TestClient) -> None:
    """Verify order=desc query parameter sorts bars with latest date first."""
    response = client_with_seeded_db.get("/api/v1/prices/AAPL?order=desc")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["bars"][0]["session_date"] == "2026-10-21"
    assert data["bars"][1]["session_date"] == "2026-10-20"


def test_get_prices_endpoint_invalid_order_422(client_with_seeded_db: TestClient) -> None:
    """Verify order values other than asc/desc return 422."""
    response = client_with_seeded_db.get("/api/v1/prices/AAPL?order=random")
    assert response.status_code == 422


def test_cors_headers_configured(client_with_seeded_db: TestClient) -> None:
    """Verify that CORS middleware responds appropriately for Vite dev server origin."""
    response = client_with_seeded_db.get(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_get_experiment_metrics_endpoint(client_with_seeded_db: TestClient) -> None:
    """Verify GET /api/v1/experiments/metrics returns valid experiment scores."""
    response = client_with_seeded_db.get("/api/v1/experiments/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "retrospective_models" in data
    assert "expanding_cv_averages" in data
    assert "B0a_majority" in data["retrospective_models"]
    assert "M0_price_only" in data["retrospective_models"]
    assert "M1_price_sec" in data["retrospective_models"]
    assert "M2_price_sec_market" in data["retrospective_models"]
    assert "disclaimer" in data
    # Check that CV fold averages contain expected keys
    assert "M0" in data["expanding_cv_averages"]
    assert "M1" in data["expanding_cv_averages"]
    assert "M2" in data["expanding_cv_averages"]
