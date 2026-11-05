"""FastAPI Prediction API Endpoint Integration Tests (EL5-050)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import get_db
from backend.app.main import app
from backend.app.models import Base, Company, PriceBar, Security


@pytest.fixture
def api_client() -> TestClient:
    """Provide a TestClient for FastAPI application."""
    return TestClient(app)


def test_root_endpoint_includes_predictions(api_client: TestClient) -> None:
    """Verify root info route exposes predictions endpoint."""
    response = api_client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "api_v1_predictions" in data
    assert data["api_v1_predictions"] == "/api/v1/predictions/{ticker}"


def test_get_prediction_success(api_client: TestClient) -> None:
    """Verify successful 5-day direction prediction on AAPL."""
    response = api_client.get("/api/v1/predictions/AAPL")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["ticker"] == "AAPL"
    assert data["company_name"] == "Apple Inc."
    assert data["horizon_trading_days"] == 5
    assert data["as_of_date"] == "2026-10-30"
    assert data["predicted_direction"] in [0, 1]
    assert 0.0 <= data["probability_up"] <= 1.0
    assert "HistGradientBoosting" in data["model_name"]
    assert "Alpaca Market Data" in data["data_source"]
    assert "Notice: Prediction is derived from historical end-of-day data" in data["warning"]


def test_get_prediction_all_five_companies(api_client: TestClient) -> None:
    """Verify predictions succeed for all 5 core tracked companies including SPCX."""
    for ticker in ["AAPL", "NVDA", "MU", "DELL", "SPCX"]:
        response = api_client.get(f"/api/v1/predictions/{ticker}")
        assert response.status_code == status.HTTP_200_OK, f"Failed for {ticker}: {response.text}"
        data = response.json()
        assert data["ticker"] == ticker
        assert data["as_of_date"] == "2026-10-30"


def test_get_prediction_unknown_ticker_404(api_client: TestClient) -> None:
    """Verify 404 returned for untracked ticker symbol."""
    response = api_client.get("/api/v1/predictions/INVALIDTICKER")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_get_prediction_missing_model_503(api_client: TestClient) -> None:
    """Verify 503 returned when model artifact file is absent."""
    missing_path = Path("models/non_existent_model.joblib")
    with patch("backend.app.routes.predictions.MODEL_PATH", missing_path):
        response = api_client.get("/api/v1/predictions/AAPL")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Model artifact not found" in response.json()["detail"]


def test_get_prediction_insufficient_history_422(tmp_path: Path) -> None:
    """Verify 422 returned when fewer than 34 bars exist for feature warmup."""
    db_file = tmp_path / "test_short_history.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    with session_factory() as db:
        c = Company(company_name="Short History Corp", sec_cik="0009999999")
        db.add(c)
        db.flush()
        s = Security(
            company_id=c.id,
            ticker="SHORT",
            exchange="NASDAQ",
            valid_from=datetime.date(2026, 1, 1),
        )
        db.add(s)
        db.flush()
        # Add only 10 bars (less than 34 required)
        for i in range(10):
            bar = PriceBar(
                security_id=s.id,
                provider="alpaca",
                feed="iex",
                session_date=datetime.date(2026, 1, 10 + i),
                open=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("95.00"),
                close=Decimal("102.00"),
                volume=10000,
            )
            db.add(bar)
        db.commit()

    def override_get_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        test_client = TestClient(app)
        response = test_client.get("/api/v1/predictions/SHORT")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert "requires at least 34 bars" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_prediction_responses_contain_no_secrets(api_client: TestClient) -> None:
    """Verify prediction endpoint responses contain zero exposed credentials."""
    response = api_client.get("/api/v1/predictions/AAPL")
    content = response.text
    forbidden_terms = ["PK", "SECRET", "Bearer", "api_key", "password"]
    for term in forbidden_terms:
        assert term not in content
