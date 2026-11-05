"""Unit tests for SEC Fundamentals Extraction, Point-in-Time Alignment, and M1 Pipeline (EL5-060).

All tests are strictly offline and hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.features import FEATURE_COLUMNS
from backend.app.main import app
from backend.app.ml.sec_experiment import (
    M1_FEATURE_COLUMNS,
    create_m1_model,
)
from backend.app.sec_fundamentals import (
    SEC_FEATURE_COLUMNS,
    align_sec_fundamentals_point_in_time,
    parse_sec_facts_json,
    select_best_concept_records,
)

client = TestClient(app)


@pytest.fixture
def sample_company_facts_json() -> dict[str, Any]:
    """Provide realistic synthetic SEC CompanyFacts JSON with multiple forms and restatements."""
    return {
        "cik": 1045810,
        "entityName": "TEST CORP",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "label": "Revenue from Contract with Customer",
                    "units": {
                        "USD": [
                            # Q1 10-Q (3-month flow)
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "val": 10000000.0,
                                "accn": "0001-25-0001",
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-04-15",
                            },
                            # Q2 10-Q (6-month cumulative YTD - 180 days - should be IGNORED)
                            {
                                "start": "2025-01-01",
                                "end": "2025-06-30",
                                "val": 22000000.0,
                                "accn": "0001-25-0002",
                                "fy": 2025,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2025-07-15",
                            },
                            # Q2 10-Q (3-month pure quarter - 90 days - should be KEPT)
                            {
                                "start": "2025-04-01",
                                "end": "2025-06-30",
                                "val": 12000000.0,
                                "accn": "0001-25-0002",
                                "fy": 2025,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2025-07-15",
                            },
                            # Subsequent restatement filed much later - should NOT leak
                            {
                                "start": "2025-04-01",
                                "end": "2025-06-30",
                                "val": 12500000.0,
                                "accn": "0001-25-0099",
                                "fy": 2025,
                                "fp": "Q2",
                                "form": "10-Q/A",
                                "filed": "2025-11-01",
                            },
                        ]
                    },
                },
                "NetIncomeLoss": {
                    "label": "Net Income Loss",
                    "units": {
                        "USD": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-03-31",
                                "val": 2000000.0,
                                "accn": "0001-25-0001",
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-04-15",
                            },
                            {
                                "start": "2025-04-01",
                                "end": "2025-06-30",
                                "val": 3000000.0,
                                "accn": "0001-25-0002",
                                "fy": 2025,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2025-07-15",
                            },
                        ]
                    },
                },
                "Assets": {
                    "label": "Assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2025-03-31",
                                "val": 50000000.0,
                                "accn": "0001-25-0001",
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-04-15",
                            },
                            {
                                "end": "2025-06-30",
                                "val": 55000000.0,
                                "accn": "0001-25-0002",
                                "fy": 2025,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2025-07-15",
                            },
                        ]
                    },
                },
                "Liabilities": {
                    "label": "Liabilities",
                    "units": {
                        "USD": [
                            {
                                "end": "2025-03-31",
                                "val": 20000000.0,
                                "accn": "0001-25-0001",
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-04-15",
                            },
                            {
                                "end": "2025-06-30",
                                "val": 22000000.0,
                                "accn": "0001-25-0002",
                                "fy": 2025,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2025-07-15",
                            },
                        ]
                    },
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "label": "Cash",
                    "units": {
                        "USD": [
                            {
                                "end": "2025-03-31",
                                "val": 15000000.0,
                                "accn": "0001-25-0001",
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-04-15",
                            },
                            {
                                "end": "2025-06-30",
                                "val": 18000000.0,
                                "accn": "0001-25-0002",
                                "fy": 2025,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2025-07-15",
                            },
                        ]
                    },
                },
            }
        },
    }


def test_select_best_concept_records_priority() -> None:
    """Verify modern ASC 606 concept is chosen over legacy deprecated concepts."""
    us_gaap = {
        "SalesRevenueNet": {
            "units": {
                "USD": [
                    {"form": "10-K", "val": 100, "filed": "2017-01-01"},
                    {"form": "10-K", "val": 110, "filed": "2017-12-31"},
                ]
            }
        },
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {
                "USD": [
                    {"form": "10-Q", "val": 500, "filed": "2025-05-01"},
                ]
            }
        },
    }
    concept, records = select_best_concept_records(
        us_gaap,
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    )
    assert concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert len(records) == 1
    assert records[0]["val"] == 500


def test_parse_sec_facts_json_duration_filtering(sample_company_facts_json: dict[str, Any]) -> None:
    """Verify 3-month flow is separated from cumulative YTD flow."""
    filings = parse_sec_facts_json(sample_company_facts_json)
    assert len(filings) >= 2

    # Check Q1 filing (filed 2025-04-15)
    f_q1 = next(f for f in filings if f["filed"] == "2025-04-15")
    assert f_q1["revenue"] == 10000000.0
    assert f_q1["net_income"] == 2000000.0
    assert f_q1["net_margin"] == 0.20  # 2M / 10M
    assert f_q1["debt_to_assets"] == 0.40  # 20M / 50M
    assert np.isnan(f_q1["revenue_growth_qoq"])  # first quarter, no prev

    # Check Q2 filing (filed 2025-07-15)
    f_q2 = next(f for f in filings if f["filed"] == "2025-07-15")
    # Must use 12,000,000 (quarterly), NOT 22,000,000 (cumulative 6-month)
    assert f_q2["revenue"] == 12000000.0
    assert f_q2["net_income"] == 3000000.0
    assert f_q2["net_margin"] == 0.25  # 3M / 12M
    assert f_q2["revenue_growth_qoq"] == 0.20  # (12M - 10M) / 10M = +20%


def test_point_in_time_alignment_strict_boundary(sample_company_facts_json: dict[str, Any]) -> None:
    """Verify filings are strictly NOT observable on or before their filed date."""
    filings = parse_sec_facts_json(sample_company_facts_json)
    company_filings = {"TEST": filings}

    # Simulate price bars: before filing, on filing date, and strictly after filing date
    price_df = pd.DataFrame(
        [
            {"ticker": "TEST", "session_date": "2025-04-14", "close": 100.0},
            {"ticker": "TEST", "session_date": "2025-04-15", "close": 101.0},  # filed date!
            {"ticker": "TEST", "session_date": "2025-04-16", "close": 102.0},  # strictly after!
            {"ticker": "TEST", "session_date": "2025-07-15", "close": 105.0},  # Q2 filed date!
            {"ticker": "TEST", "session_date": "2025-07-16", "close": 106.0},  # strictly after Q2!
        ]
    )

    merged = align_sec_fundamentals_point_in_time(price_df, company_filings)
    assert len(merged) == 5

    # Day 1 (2025-04-14): BEFORE filing -> Must be NaN
    assert pd.isna(merged.iloc[0]["sec_quarterly_revenue"])
    assert pd.isna(merged.iloc[0]["sec_filed_date"])

    # Day 2 (2025-04-15): ON filing date -> Must be NaN (filing hour unknown, conservative rule)
    assert pd.isna(merged.iloc[1]["sec_quarterly_revenue"])
    assert pd.isna(merged.iloc[1]["sec_filed_date"])

    # Day 3 (2025-04-16): Strictly AFTER Q1 filing -> Must reflect Q1
    assert merged.iloc[2]["sec_quarterly_revenue"] == 10000000.0
    assert merged.iloc[2]["sec_filed_date"] == "2025-04-15"
    assert merged.iloc[2]["sec_form"] == "10-Q"

    # Day 4 (2025-07-15): ON Q2 filing date -> Still reflects Q1 (Q2 not yet effective)
    assert merged.iloc[3]["sec_quarterly_revenue"] == 10000000.0
    assert merged.iloc[3]["sec_filed_date"] == "2025-04-15"

    # Day 5 (2025-07-16): Strictly AFTER Q2 filing -> Reflects Q2
    assert merged.iloc[4]["sec_quarterly_revenue"] == 12000000.0
    assert merged.iloc[4]["sec_filed_date"] == "2025-07-15"


def test_restatement_leak_prevention(sample_company_facts_json: dict[str, Any]) -> None:
    """Verify an amended filing (10-Q/A) does not leak prior to its filed date."""
    filings = parse_sec_facts_json(sample_company_facts_json)
    company_filings = {"TEST": filings}

    price_df = pd.DataFrame(
        [
            {"ticker": "TEST", "session_date": "2025-08-01", "close": 110.0},  # before amendment
            {"ticker": "TEST", "session_date": "2025-11-02", "close": 120.0},  # after amendment
        ]
    )

    merged = align_sec_fundamentals_point_in_time(price_df, company_filings)

    # Prior to 2025-11-01, original Q2 revenue was 12M
    assert merged.iloc[0]["sec_quarterly_revenue"] == 12000000.0
    assert merged.iloc[0]["sec_accn"] == "0001-25-0002"

    # After 2025-11-01, amended revenue becomes visible
    assert merged.iloc[1]["sec_quarterly_revenue"] == 12500000.0
    assert merged.iloc[1]["sec_accn"] == "0001-25-0099"


def test_spcx_unfiled_sessions_handled_gracefully() -> None:
    """Verify that sessions prior to a company's first filing have NaN without crashing."""
    price_df = pd.DataFrame(
        [
            {"ticker": "SPCX", "session_date": "2026-07-15", "close": 50.0},
            {"ticker": "SPCX", "session_date": "2026-07-16", "close": 51.0},
        ]
    )
    # Company with empty filings list
    merged = align_sec_fundamentals_point_in_time(price_df, {"SPCX": []})
    assert len(merged) == 2
    for col in SEC_FEATURE_COLUMNS:
        assert merged[col].isna().all()


def test_m1_dataset_matches_m0_structure() -> None:
    """Verify ml_sec_5d.csv preserves exact rows, labels, and price features of ml_train_5d.csv."""
    p_base = Path("data/derived/ml_train_5d.csv")
    p_sec = Path("data/derived/ml_sec_5d.csv")

    if not p_sec.exists():
        pytest.skip("ml_sec_5d.csv has not been generated yet.")

    df_base = pd.read_csv(p_base)
    df_sec = pd.read_csv(p_sec)

    assert len(df_sec) == len(df_base)
    assert len(df_sec) == 1908
    assert (df_sec["direction_5d"] == df_base["direction_5d"]).all()
    assert (df_sec["ticker"] == df_base["ticker"]).all()
    assert (df_sec["session_date"] == df_base["session_date"]).all()

    # Check 23 original features exist and are unaltered
    for col in FEATURE_COLUMNS:
        assert col in df_sec.columns
        np.testing.assert_allclose(df_sec[col].values, df_base[col].values, rtol=1e-5, atol=1e-5)

    # Check 8 SEC features exist
    for col in SEC_FEATURE_COLUMNS:
        assert col in df_sec.columns


def test_m1_pipeline_train_only_imputer() -> None:
    """Verify SimpleImputer fits exclusively on training data and doesn't leak into test."""
    m1_pipe = create_m1_model()

    # Synthetic train with median = 100.0 for missing feature
    X_train = pd.DataFrame({col: np.ones(10) * 1.0 for col in FEATURE_COLUMNS})
    for col in SEC_FEATURE_COLUMNS:
        X_train[col] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, np.nan]
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    m1_pipe.fit(X_train[M1_FEATURE_COLUMNS], y_train)

    # Test sample with all NaNs in SEC features
    X_test = pd.DataFrame({col: np.ones(2) * 1.0 for col in FEATURE_COLUMNS})
    for col in SEC_FEATURE_COLUMNS:
        X_test[col] = [np.nan, np.nan]

    # Predict must succeed without raising NaN errors
    preds = m1_pipe.predict(X_test[M1_FEATURE_COLUMNS])
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

    # Imputer statistics must equal 100.0 from training data
    imputer = m1_pipe.named_steps["imputer"]
    sec_imputed_medians = imputer.statistics_[len(FEATURE_COLUMNS) :]
    assert all(med == 100.0 for med in sec_imputed_medians)


def test_production_prediction_endpoint_regression() -> None:
    """Verify existing prediction endpoint /api/v1/predictions/{ticker} remains functional."""
    resp = client.get("/api/v1/predictions/AAPL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert "predicted_direction" in data
    assert "probability_up" in data
    assert data["horizon_trading_days"] == 5
    assert data["model_name"] == "HistGradientBoostingClassifier"


def test_production_model_files_unmodified() -> None:
    """Verify production model and metadata files are intact."""
    prod_model = Path("models/direction_5d_model.joblib")
    prod_meta = Path("models/model_metadata.json")
    prod_eval = Path("models/evaluation_metrics.json")

    assert prod_model.is_file()
    assert prod_meta.is_file()
    assert prod_eval.is_file()

    with open(prod_meta, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["target_variable"] == "direction_5d"
    assert meta["feature_count"] == 23  # Production model strictly retains 23 features
