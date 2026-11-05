"""EquityLens-5 Machine Learning Feature Engineering and Target Generation.

This module provides reproducible, leak-free feature extraction and forward-looking
target generation for daily OHLCV equity price bars. All rolling indicators are
computed strictly in chronological order per security, with zero look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger("EquityLens5.Features")

# Metadata and raw columns
METADATA_COLUMNS: Final[list[str]] = ["ticker", "company_name", "session_date"]
RAW_PRICE_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]

# Feature columns (23 features)
FEATURE_COLUMNS: Final[list[str]] = [
    # A. Price Returns
    "return_1d",
    "return_5d",
    "return_10d",
    # B. Trend
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_5",
    "ema_10",
    "ema_20",
    # C. Momentum
    "rsi_14",
    "macd_12_26",
    "macd_signal_9",
    "macd_histogram",
    # D. Volatility
    "rolling_volatility_5",
    "rolling_volatility_20",
    "bollinger_mid_20",
    "bollinger_upper_20",
    "bollinger_lower_20",
    # E. Volume
    "volume_sma_20",
    "volume_ratio_20",
    # F. Normalized Features
    "close_to_sma_20",
    "close_to_ema_20",
    "bollinger_position",
]

# Target columns (4 target columns)
TARGET_COLUMNS: Final[list[str]] = [
    "future_return_5d",
    "direction_5d",
    "future_return_20d",
    "direction_20d",
]

# Complete schema
ALL_COLUMNS: Final[list[str]] = (
    METADATA_COLUMNS + RAW_PRICE_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
)


def compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's Relative Strength Index (RSI).

    Warmup values (< period) evaluate to NaN.
    Output is strictly bounded in [0.0, 100.0].
    """
    delta = close_series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Relative Strength
    rs = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Boundary handling for zero loss or zero gain
    zero_loss = (avg_loss == 0.0) & (avg_gain > 0.0)
    zero_gain = (avg_gain == 0.0) & (avg_loss > 0.0)
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)

    rsi = rsi.where(~zero_loss, 100.0)
    rsi = rsi.where(~zero_gain, 0.0)
    rsi = rsi.where(~both_zero, 50.0)

    # Ensure NaN during warmup
    return rsi.where(~avg_gain.isna(), np.nan)


def compute_features_for_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 23 technical features for a single security's chronological series.

    Strictly backwards-looking; never accesses rows at t+1 or beyond.
    """
    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    # A. Price Returns
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)

    # B. Trend
    out["sma_5"] = close.rolling(window=5, min_periods=5).mean()
    out["sma_10"] = close.rolling(window=10, min_periods=10).mean()
    out["sma_20"] = close.rolling(window=20, min_periods=20).mean()

    out["ema_5"] = close.ewm(span=5, adjust=False, min_periods=5).mean()
    out["ema_10"] = close.ewm(span=10, adjust=False, min_periods=10).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()

    # C. Momentum
    out["rsi_14"] = compute_rsi(close, period=14)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_12_26 = ema_12 - ema_26
    macd_signal_9 = macd_12_26.ewm(span=9, adjust=False, min_periods=9).mean()

    out["macd_12_26"] = macd_12_26
    out["macd_signal_9"] = macd_signal_9
    out["macd_histogram"] = macd_12_26 - macd_signal_9

    # D. Volatility
    out["rolling_volatility_5"] = out["return_1d"].rolling(window=5, min_periods=5).std(ddof=1)
    out["rolling_volatility_20"] = out["return_1d"].rolling(window=20, min_periods=20).std(ddof=1)

    bollinger_mid = out["sma_20"]
    rolling_std_20 = close.rolling(window=20, min_periods=20).std(ddof=1)
    out["bollinger_mid_20"] = bollinger_mid
    out["bollinger_upper_20"] = bollinger_mid + (2.0 * rolling_std_20)
    out["bollinger_lower_20"] = bollinger_mid - (2.0 * rolling_std_20)

    # E. Volume
    out["volume_sma_20"] = volume.rolling(window=20, min_periods=20).mean()
    out["volume_ratio_20"] = volume / out["volume_sma_20"].where(out["volume_sma_20"] > 0, np.nan)

    # F. Normalized Features
    out["close_to_sma_20"] = (close / out["sma_20"].where(out["sma_20"] > 0, np.nan)) - 1.0
    out["close_to_ema_20"] = (close / out["ema_20"].where(out["ema_20"] > 0, np.nan)) - 1.0

    band_width = out["bollinger_upper_20"] - out["bollinger_lower_20"]
    out["bollinger_position"] = (close - out["bollinger_lower_20"]) / band_width.where(
        band_width > 0, np.nan
    )

    return out


def compute_targets_for_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward-looking prediction targets for a single security's series.

    Trading-session shifts shift(-5) and shift(-20) are used.
    Unknown tail horizons remain NaN and are NEVER imputed as 0.
    """
    out = df.copy()
    close = out["close"].astype(float)

    # 5-trading-day ahead horizon
    f_ret_5 = (close.shift(-5) / close) - 1.0
    out["future_return_5d"] = f_ret_5
    out["direction_5d"] = np.where(f_ret_5.isna(), np.nan, (f_ret_5 > 0.0).astype(float))

    # 20-trading-day ahead horizon
    f_ret_20 = (close.shift(-20) / close) - 1.0
    out["future_return_20d"] = f_ret_20
    out["direction_20d"] = np.where(f_ret_20.isna(), np.nan, (f_ret_20 > 0.0).astype(float))

    return out


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical features and target variables across all securities.

    Input DataFrame must contain: ticker, session_date, open, high, low, close, volume.
    Processes each security independently in strict chronological order.
    """
    required_cols = {"ticker", "session_date", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Input DataFrame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    # Copy and ensure sorting
    work_df = df.copy()
    work_df["session_date"] = work_df["session_date"].astype(str)
    work_df = work_df.sort_values(by=["ticker", "session_date"]).reset_index(drop=True)

    processed_groups: list[pd.DataFrame] = []

    for _, group in work_df.groupby("ticker", sort=False):
        g_feat = compute_features_for_series(group)
        g_both = compute_targets_for_series(g_feat)
        processed_groups.append(g_both)

    result_df = pd.concat(processed_groups, ignore_index=True)

    # Ensure all expected columns are present in deterministic order
    final_cols = [c for c in ALL_COLUMNS if c in result_df.columns]
    # Add any extra columns from input at the beginning/end
    extras = [c for c in result_df.columns if c not in final_cols]
    return result_df[final_cols + extras]


def split_train_and_prediction_sets(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split processed DataFrame into clean training sets and prediction-ready set.

    - ml_train_5d: All 23 features valid AND direction_5d not null.
    - ml_train_20d: All 23 features valid AND direction_20d not null.
    - prediction_ready: All 23 features valid AND (direction_5d is null OR direction_20d is null).
    """
    valid_features_mask = ~df[FEATURE_COLUMNS].isna().any(axis=1)

    train_5d = df[valid_features_mask & ~df["direction_5d"].isna()].copy()
    train_20d = df[valid_features_mask & ~df["direction_20d"].isna()].copy()

    # Prediction-ready: features are fully populated, but future direction is unknown
    pred_ready = df[valid_features_mask & df["direction_5d"].isna()].copy()

    return {
        "features_all": df.copy(),
        "ml_train_5d": train_5d,
        "ml_train_20d": train_20d,
        "prediction_ready": pred_ready,
    }
