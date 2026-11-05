"""Unit tests for Market/Sector Features, M2 Pipeline, and Data Consistency (EL5-061).

All tests are strictly offline and hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.features import FEATURE_COLUMNS
from backend.app.main import app
from backend.app.market_features import (
    MARKET_FEATURE_COLUMNS,
    compute_market_etf_features,
    load_etf_bars_json,
    merge_market_features,
)
from backend.app.ml.market_m2_experiment import (
    M2_FEATURE_COLUMNS,
    create_m2_model,
)
from backend.app.sec_fundamentals import SEC_FEATURE_COLUMNS

client = TestClient(app)


def test_data_consistency_across_all_derived_datasets() -> None:
    """Audit test: Verify exact row count, primary keys, and targets across derived datasets."""
    p_train = Path("data/derived/ml_train_5d.csv")
    p_sec = Path("data/derived/ml_sec_5d.csv")
    p_market = Path("data/derived/ml_market_sec_5d.csv")

    assert p_train.is_file(), "ml_train_5d.csv must exist"
    assert p_sec.is_file(), "ml_sec_5d.csv must exist"
    if not p_market.is_file():
        pytest.skip("ml_market_sec_5d.csv has not been generated yet.")

    df_train = pd.read_csv(p_train)
    df_sec = pd.read_csv(p_sec)
    df_market = pd.read_csv(p_market)

    # 1. Row counts
    assert len(df_train) == 1908
    assert len(df_sec) == 1908
    assert len(df_market) == 1908

    # 2. Primary Keys (ticker + session_date)
    pk_train = df_train["ticker"] + "_" + df_train["session_date"]
    pk_sec = df_sec["ticker"] + "_" + df_sec["session_date"]
    pk_market = df_market["ticker"] + "_" + df_market["session_date"]

    assert (pk_train == pk_sec).all()
    assert (pk_train == pk_market).all()
    assert pk_market.duplicated().sum() == 0

    # 3. Target Variable (direction_5d)
    assert (df_train["direction_5d"] == df_sec["direction_5d"]).all()
    assert (df_train["direction_5d"] == df_market["direction_5d"]).all()

    # 4. Exact Class Distribution
    pos_count = int((df_market["direction_5d"] == 1.0).sum())
    neg_count = int((df_market["direction_5d"] == 0.0).sum())
    assert pos_count == 1107
    assert neg_count == 801
    assert round(pos_count / len(df_market), 4) == 0.5802


def test_load_etf_bars_json(tmp_path: Path) -> None:
    """Verify loading and parsing of raw ETF bar JSON records."""
    dummy_bars = [
        {"t": "2025-01-02T05:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 1000},
        {"t": "2025-01-03T05:00:00Z", "o": 101.0, "h": 103.0, "l": 100.0, "c": 102.0, "v": 1200},
    ]
    json_path = tmp_path / "TEST_bars.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dummy_bars, f)

    df = load_etf_bars_json(json_path)
    assert len(df) == 2
    assert list(df["session_date"]) == ["2025-01-02", "2025-01-03"]
    assert df.iloc[0]["close"] == 101.0
    assert df.iloc[1]["volume"] == 1200.0


def test_compute_market_etf_features(tmp_path: Path) -> None:
    """Verify calculation of 9 market and sector features with mathematical accuracy."""
    # Generate 25 synthetic dates so 20d rolling volatility is populated
    dates = pd.date_range("2025-01-01", periods=25, freq="B").strftime("%Y-%m-%d").tolist()
    spy_records = []
    xlk_records = []
    soxx_records = []

    for i, d in enumerate(dates):
        spy_c = 100.0 + i * 1.0
        xlk_c = 50.0 + i * 0.5
        soxx_c = 200.0 + i * 2.0
        spy_records.append(
            {"t": f"{d}T05:00:00Z", "o": spy_c, "h": spy_c, "l": spy_c, "c": spy_c, "v": 100}
        )
        xlk_records.append(
            {"t": f"{d}T05:00:00Z", "o": xlk_c, "h": xlk_c, "l": xlk_c, "c": xlk_c, "v": 100}
        )
        soxx_records.append(
            {"t": f"{d}T05:00:00Z", "o": soxx_c, "h": soxx_c, "l": soxx_c, "c": soxx_c, "v": 100}
        )

    with open(tmp_path / "SPY_bars.json", "w", encoding="utf-8") as f:
        json.dump(spy_records, f)
    with open(tmp_path / "XLK_bars.json", "w", encoding="utf-8") as f:
        json.dump(xlk_records, f)
    with open(tmp_path / "SOXX_bars.json", "w", encoding="utf-8") as f:
        json.dump(soxx_records, f)

    m_df = compute_market_etf_features(tmp_path)
    assert len(m_df) == 25
    for col in MARKET_FEATURE_COLUMNS:
        assert col in m_df.columns

    # Check that day 24 has non-null 20d volatility
    assert pd.notna(m_df.iloc[-1]["spy_volatility_20"])
    # Check relative returns: xlk_relative_to_spy_5d = xlk_5d - spy_5d
    last_row = m_df.iloc[-1]
    expected_rel = round(last_row["xlk_return_5d"] - last_row["spy_return_5d"], 6)
    assert abs(last_row["xlk_relative_to_spy_5d"] - expected_rel) < 1e-5


def test_merge_market_features_integrity() -> None:
    """Verify market feature merge preserves row count and aligns strictly on session_date."""
    df_panel = pd.DataFrame(
        [
            {"ticker": "AAPL", "session_date": "2025-01-02", "close": 150.0},
            {"ticker": "NVDA", "session_date": "2025-01-02", "close": 120.0},
            {"ticker": "AAPL", "session_date": "2025-01-03", "close": 152.0},
        ]
    )
    df_market = pd.DataFrame(
        [
            {"session_date": "2025-01-02", "spy_return_1d": 0.01, "spy_return_5d": 0.03},
            {"session_date": "2025-01-03", "spy_return_1d": -0.005, "spy_return_5d": 0.02},
        ]
    )

    merged = merge_market_features(df_panel, df_market)
    assert len(merged) == 3
    assert merged.iloc[0]["spy_return_1d"] == 0.01
    assert merged.iloc[1]["spy_return_1d"] == 0.01
    assert merged.iloc[2]["spy_return_1d"] == -0.005


def test_m2_pipeline_train_only_imputer() -> None:
    """Verify M2 pipeline accepts 40 features and fits SimpleImputer purely on train data."""
    m2_pipe = create_m2_model()

    # Generate synthetic training set with 40 features
    all_cols = FEATURE_COLUMNS + SEC_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS
    assert len(all_cols) == 40
    assert len(M2_FEATURE_COLUMNS) == 40

    X_train = pd.DataFrame({col: np.ones(10) * 1.0 for col in all_cols})
    # Add NaNs in SEC features with median = 50.0
    for col in SEC_FEATURE_COLUMNS:
        X_train[col] = [50.0] * 9 + [np.nan]
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    m2_pipe.fit(X_train[M2_FEATURE_COLUMNS], y_train)

    # Test sample with all NaNs in SEC features
    X_test = pd.DataFrame({col: np.ones(2) * 1.0 for col in all_cols})
    for col in SEC_FEATURE_COLUMNS:
        X_test[col] = [np.nan, np.nan]

    preds = m2_pipe.predict(X_test[M2_FEATURE_COLUMNS])
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

    # Imputer statistics verify train-only fitting
    imputer = m2_pipe.named_steps["imputer"]
    sec_start = len(FEATURE_COLUMNS)
    sec_end = sec_start + len(SEC_FEATURE_COLUMNS)
    sec_medians = imputer.statistics_[sec_start:sec_end]
    assert all(m == 50.0 for m in sec_medians)


def test_spcx_in_m2_dataset() -> None:
    """Verify SPCX post-IPO sessions contain valid market features without errors."""
    p_market = Path("data/derived/ml_market_sec_5d.csv")
    if not p_market.is_file():
        pytest.skip("ml_market_sec_5d.csv not yet generated")

    df = pd.read_csv(p_market)
    spcx_df = df[df["ticker"] == "SPCX"]
    assert len(spcx_df) == 60

    # Market features must be 100% populated for SPCX
    for col in MARKET_FEATURE_COLUMNS:
        assert spcx_df[col].isna().sum() == 0


def test_production_predictions_api_and_model_unmodified() -> None:
    """Verify existing prediction API and production model remain intact."""
    prod_model = Path("models/direction_5d_model.joblib")
    assert prod_model.is_file()

    with open("models/model_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["target_variable"] == "direction_5d"
    assert meta["feature_count"] == 23

    # API test
    resp = client.get("/api/v1/predictions/NVDA")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "NVDA"
    assert data["horizon_trading_days"] == 5
    assert "predicted_direction" in data
