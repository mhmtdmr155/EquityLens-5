"""Unit and regression tests for EquityLens-5 Feature Engineering (EL5-040).

Verifies mathematical accuracy of technical indicators, strict chronological
isolation, zero look-ahead bias in features, proper tail NaN handling for targets,
per-symbol grouping, and dataset export reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.features import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RAW_PRICE_COLUMNS,
    TARGET_COLUMNS,
    build_feature_pipeline,
    compute_features_for_series,
    compute_rsi,
    compute_targets_for_series,
    split_train_and_prediction_sets,
)
from scripts.prepare_ml_dataset import run_dataset_preparation


def _create_sample_series(
    ticker: str = "TEST",
    n_days: int = 100,
    base_price: float = 100.0,
    daily_growth: float = 0.001,
) -> pd.DataFrame:
    """Generate deterministic synthetic chronological price series for testing."""
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B").strftime("%Y-%m-%d")
    prices = [base_price * ((1.0 + daily_growth) ** i) for i in range(n_days)]

    df = pd.DataFrame(
        {
            "ticker": ticker,
            "company_name": f"{ticker} Test Corp",
            "session_date": dates,
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1_000_000 + (i * 1000) for i in range(n_days)],
        }
    )
    return df


def test_schema_specifications() -> None:
    """Verify feature and target column counts and completeness."""
    assert len(FEATURE_COLUMNS) == 23
    assert len(TARGET_COLUMNS) == 4
    assert len(METADATA_COLUMNS) == 3
    assert len(RAW_PRICE_COLUMNS) == 5
    assert len(ALL_COLUMNS) == 35

    # Check key column existence
    assert "rsi_14" in FEATURE_COLUMNS
    assert "macd_12_26" in FEATURE_COLUMNS
    assert "macd_signal_9" in FEATURE_COLUMNS
    assert "macd_histogram" in FEATURE_COLUMNS
    assert "rolling_volatility_5" in FEATURE_COLUMNS
    assert "bollinger_position" in FEATURE_COLUMNS
    assert "direction_5d" in TARGET_COLUMNS
    assert "direction_20d" in TARGET_COLUMNS


def test_rsi_calculation_math_and_boundaries() -> None:
    """Verify Wilder's RSI on strictly rising, strictly falling, and mixed data."""
    # Strictly rising series: RSI must approach 100
    rising = pd.Series([10.0 + i for i in range(30)])
    rsi_rising = compute_rsi(rising, period=14)
    assert rsi_rising.iloc[:14].isna().all()
    assert rsi_rising.iloc[14] == 100.0
    assert rsi_rising.iloc[-1] == 100.0

    # Strictly falling series: RSI must be 0
    falling = pd.Series([100.0 - i for i in range(30)])
    rsi_falling = compute_rsi(falling, period=14)
    assert rsi_falling.iloc[:14].isna().all()
    assert rsi_falling.iloc[14] == 0.0
    assert rsi_falling.iloc[-1] == 0.0

    # Constant series: zero gain and zero loss evaluates to 50
    flat = pd.Series([50.0] * 30)
    rsi_flat = compute_rsi(flat, period=14)
    assert rsi_flat.iloc[:14].isna().all()
    assert rsi_flat.iloc[14] == 50.0

    # Boundary check: values must remain strictly within [0, 100]
    np.random.seed(42)
    random_prices = pd.Series(100.0 + np.random.randn(100).cumsum())
    rsi_rand = compute_rsi(random_prices, period=14)
    valid_rsi = rsi_rand.dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_moving_averages_math() -> None:
    """Verify SMA and EMA calculations against manual ground truth."""
    df = _create_sample_series(n_days=50)
    feat = compute_features_for_series(df)

    # First 4 rows of sma_5 must be NaN, 5th row must match mean of first 5 closes
    assert feat["sma_5"].iloc[:4].isna().all()
    expected_sma_5 = df["close"].iloc[:5].mean()
    assert pytest.approx(feat["sma_5"].iloc[4], rel=1e-6) == expected_sma_5

    # First 19 rows of sma_20 must be NaN, 20th row must match mean of first 20 closes
    assert feat["sma_20"].iloc[:19].isna().all()
    expected_sma_20 = df["close"].iloc[:20].mean()
    assert pytest.approx(feat["sma_20"].iloc[19], rel=1e-6) == expected_sma_20

    # EMA warmup behavior
    assert feat["ema_5"].iloc[:4].isna().all()
    assert feat["ema_20"].iloc[:19].isna().all()


def test_macd_and_histogram_math() -> None:
    """Verify MACD line, signal line, and histogram relations."""
    df = _create_sample_series(n_days=60)
    feat = compute_features_for_series(df)

    # macd_12_26 needs 26 periods
    assert feat["macd_12_26"].iloc[:25].isna().all()
    assert not feat["macd_12_26"].iloc[25:].isna().any()

    # macd_signal_9 needs 9 periods of macd_12_26, so first 33 rows are NaN
    assert feat["macd_signal_9"].iloc[:33].isna().all()
    assert not feat["macd_signal_9"].iloc[33:].isna().any()

    # Histogram identity: histogram = macd - signal
    valid_idx = feat["macd_histogram"].dropna().index
    diff = feat.loc[valid_idx, "macd_histogram"] - (
        feat.loc[valid_idx, "macd_12_26"] - feat.loc[valid_idx, "macd_signal_9"]
    )
    assert (diff.abs() < 1e-10).all()


def test_volatility_and_bollinger_bands() -> None:
    """Verify rolling volatility and Bollinger band envelopes."""
    df = _create_sample_series(n_days=50)
    feat = compute_features_for_series(df)

    # Volatility warmup
    assert feat["rolling_volatility_5"].iloc[:5].isna().all()
    assert feat["rolling_volatility_20"].iloc[:20].isna().all()

    # Bollinger Bands
    assert feat["bollinger_mid_20"].iloc[:19].isna().all()
    valid_idx = feat["bollinger_upper_20"].dropna().index

    # Upper must be >= Mid, Mid must be >= Lower
    upper_gte_mid = (
        feat.loc[valid_idx, "bollinger_upper_20"] >= feat.loc[valid_idx, "bollinger_mid_20"]
    ).all()
    mid_gte_lower = (
        feat.loc[valid_idx, "bollinger_mid_20"] >= feat.loc[valid_idx, "bollinger_lower_20"]
    ).all()
    assert upper_gte_mid
    assert mid_gte_lower


def test_volume_and_normalized_features() -> None:
    """Verify volume ratios and normalized distance features."""
    df = _create_sample_series(n_days=50)
    feat = compute_features_for_series(df)

    valid_idx = feat["volume_ratio_20"].dropna().index
    assert (feat.loc[valid_idx, "volume_ratio_20"] > 0).all()

    # close_to_sma_20 identity: (close / sma_20) - 1
    c = feat.loc[valid_idx, "close"]
    sma = feat.loc[valid_idx, "sma_20"]
    expected_c_to_sma = (c / sma) - 1.0
    assert (feat.loc[valid_idx, "close_to_sma_20"] - expected_c_to_sma).abs().max() < 1e-10


def test_target_variables_calculation_and_tail_nan() -> None:
    """Verify forward-looking targets and ensure tail rows NEVER become 0."""
    df = _create_sample_series(n_days=50)
    targets = compute_targets_for_series(df)

    # Exactly the last 5 rows must have NaN 5d targets
    assert targets["future_return_5d"].iloc[-5:].isna().all()
    assert targets["direction_5d"].iloc[-5:].isna().all()

    # Exactly the last 20 rows must have NaN 20d targets
    assert targets["future_return_20d"].iloc[-20:].isna().all()
    assert targets["direction_20d"].iloc[-20:].isna().all()

    # Verify math for index 0: future_return_5d = close[5] / close[0] - 1
    expected_ret_5 = (df["close"].iloc[5] / df["close"].iloc[0]) - 1.0
    assert pytest.approx(targets["future_return_5d"].iloc[0], rel=1e-6) == expected_ret_5
    expected_dir_5 = 1.0 if expected_ret_5 > 0 else 0.0
    assert targets["direction_5d"].iloc[0] == expected_dir_5


def test_lookahead_leakage_strict_isolation() -> None:
    """CRITICAL TEST: Perturbing future price must NOT alter past features.

    However, it MUST change the past target label (future_return and direction).
    """
    df1 = _create_sample_series(n_days=60, base_price=100.0, daily_growth=0.01)
    df2 = df1.copy()

    target_idx = 20
    future_idx = target_idx + 5

    # Introduce a massive price shock at future_idx (+500%)
    df2.loc[future_idx, "close"] = df2.loc[future_idx, "close"] * 5.0
    df2.loc[future_idx, "high"] = df2.loc[future_idx, "close"] * 1.05

    pipe1 = build_feature_pipeline(df1)
    pipe2 = build_feature_pipeline(df2)

    # 1. All features at target_idx must be 100% IDENTICAL (zero leak)
    feat1 = pipe1.loc[target_idx, FEATURE_COLUMNS]
    feat2 = pipe2.loc[target_idx, FEATURE_COLUMNS]
    diff = (feat1 - feat2).abs().max()
    assert diff == 0.0 or diff < 1e-12, (
        "Look-ahead detected! Feature values changed when future price changed."
    )

    # 2. But the forward targets at target_idx MUST reflect the future change
    assert pipe1.loc[target_idx, "future_return_5d"] != pipe2.loc[target_idx, "future_return_5d"]


def test_per_symbol_isolation() -> None:
    """Verify that processing multiple symbols produces identical results to single symbol."""
    df_aapl = _create_sample_series(ticker="AAPL", n_days=50, base_price=200.0)
    df_nvda = _create_sample_series(ticker="NVDA", n_days=50, base_price=100.0)
    combined = pd.concat([df_aapl, df_nvda], ignore_index=True)

    res_combined = build_feature_pipeline(combined)
    res_aapl_single = build_feature_pipeline(df_aapl)

    comb_aapl = res_combined[res_combined["ticker"] == "AAPL"].reset_index(drop=True)
    diff = (comb_aapl[FEATURE_COLUMNS] - res_aapl_single[FEATURE_COLUMNS]).abs().max().max()
    assert diff < 1e-10, "Cross-symbol contamination detected in feature calculations."


def test_spacex_short_history_and_warmup() -> None:
    """Verify SPCX 98 bars are handled safely with correct sample counts."""
    df_spcx = _create_sample_series(ticker="SPCX", n_days=98, base_price=100.0)
    res = build_feature_pipeline(df_spcx)
    splits = split_train_and_prediction_sets(res)

    train_5d = splits["ml_train_5d"]
    train_20d = splits["ml_train_20d"]
    pred_ready = splits["prediction_ready"]

    # 98 total bars - 33 warmup bars - 5 tail bars = 60 valid 5d training samples
    assert len(train_5d) == 60
    # 98 total bars - 33 warmup bars - 20 tail bars = 45 valid 20d training samples
    assert len(train_20d) == 45
    # Exactly 5 prediction-ready samples
    assert len(pred_ready) == 5


def test_split_train_and_prediction_sets_completeness() -> None:
    """Verify training sets contain zero nulls in features and targets."""
    df = _create_sample_series(n_days=100)
    res = build_feature_pipeline(df)
    splits = split_train_and_prediction_sets(res)

    # Train 5d must have NO nulls in features or direction_5d
    assert not splits["ml_train_5d"][FEATURE_COLUMNS].isna().any().any()
    assert not splits["ml_train_5d"]["direction_5d"].isna().any()

    # Train 20d must have NO nulls in features or direction_20d
    assert not splits["ml_train_20d"][FEATURE_COLUMNS].isna().any().any()
    assert not splits["ml_train_20d"]["direction_20d"].isna().any()

    # Prediction-ready must have valid features but null direction_5d
    assert not splits["prediction_ready"][FEATURE_COLUMNS].isna().any().any()
    assert splits["prediction_ready"]["direction_5d"].isna().all()


def test_reproducible_dataset_generation(tmp_path: Path) -> None:
    """Verify prepare_ml_dataset produces deterministic identical outputs on repeated runs."""
    db_path = Path("data/equitylens_dev.db")
    if not db_path.exists():
        pytest.skip("Local development database not found.")

    out_dir1 = tmp_path / "run1"
    out_dir2 = tmp_path / "run2"

    meta1 = run_dataset_preparation(db_path, out_dir1)
    meta2 = run_dataset_preparation(db_path, out_dir2)

    # Check identical row counts
    assert meta1["row_counts"] == meta2["row_counts"]
    assert meta1["target_class_balance"] == meta2["target_class_balance"]

    # Check byte-level identical CSV files
    target_files = [
        "features_all.csv",
        "ml_train_5d.csv",
        "ml_train_20d.csv",
        "prediction_ready.csv",
    ]
    for fname in target_files:
        content1 = (out_dir1 / fname).read_text(encoding="utf-8")
        content2 = (out_dir2 / fname).read_text(encoding="utf-8")
        assert content1 == content2, f"Nondeterministic CSV export in {fname}"


def test_metadata_json_schema() -> None:
    """Verify dataset_metadata.json contains all required audit fields."""
    meta_path = Path("data/derived/dataset_metadata.json")
    if not meta_path.exists():
        pytest.skip("Dataset metadata JSON not found.")

    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_name"] == "EquityLens-5 Technical Features & ML Targets"
    assert data["total_symbols"] == 5
    assert set(data["symbols"]) == {"AAPL", "DELL", "MU", "NVDA", "SPCX"}
    assert data["row_counts"]["features_all"] == 2098
    assert data["row_counts"]["ml_train_5d"] == 1908
    assert data["row_counts"]["ml_train_20d"] == 1833
    assert data["row_counts"]["prediction_ready"] == 25
    assert "direction_5d" in data["target_class_balance"]
    assert "direction_20d" in data["target_class_balance"]
