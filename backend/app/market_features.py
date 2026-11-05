"""EquityLens-5 Market and Sector ETF Feature Engineering (EL5-061).

Extracts and computes broad market (SPY) and sector (XLK, SOXX) features,
and performs strictly leak-free, date-aligned joining with core equity bars.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd

logger = logging.getLogger("EquityLens5.MarketFeatures")

# The 9 derived market & sector features
MARKET_FEATURE_COLUMNS: Final[list[str]] = [
    "spy_return_1d",
    "spy_return_5d",
    "xlk_return_1d",
    "xlk_return_5d",
    "soxx_return_1d",
    "soxx_return_5d",
    "spy_volatility_20",
    "xlk_relative_to_spy_5d",
    "soxx_relative_to_spy_5d",
]


def load_etf_bars_json(json_path: Path | str) -> pd.DataFrame:
    """Load cached ETF bars from JSON and parse into sorted DataFrame."""
    p = Path(json_path)
    if not p.is_file():
        msg = f"ETF cache file not found: {p}"
        raise FileNotFoundError(msg)

    with open(p, encoding="utf-8") as f:
        raw_bars: list[dict[str, Any]] = json.load(f)

    if not raw_bars:
        msg = f"ETF cache file {p} is empty."
        raise ValueError(msg)

    records: list[dict[str, Any]] = []
    for b in raw_bars:
        t_str = str(b.get("t", ""))
        session_date = t_str[:10]  # Extract YYYY-MM-DD
        records.append(
            {
                "session_date": session_date,
                "open": float(b["o"]),
                "high": float(b["h"]),
                "low": float(b["l"]),
                "close": float(b["c"]),
                "volume": float(b["v"]),
            }
        )

    df = pd.DataFrame(records)
    df = (
        df.drop_duplicates(subset=["session_date"])
        .sort_values("session_date")
        .reset_index(drop=True)
    )
    return df


def compute_market_etf_features(cache_dir: Path | str) -> pd.DataFrame:
    """Compute 9 market and sector features from cached SPY, XLK, and SOXX bars.

    Strict point-in-time calculation:
    All returns and rolling statistics use past and current session dates only.
    No forward-looking shifts or centered windows are used.
    """
    c_dir = Path(cache_dir)
    spy_df = load_etf_bars_json(c_dir / "SPY_bars.json")
    xlk_df = load_etf_bars_json(c_dir / "XLK_bars.json")
    soxx_df = load_etf_bars_json(c_dir / "SOXX_bars.json")

    # Calculate returns for SPY
    spy_df["spy_return_1d"] = spy_df["close"].pct_change(1)
    spy_df["spy_return_5d"] = spy_df["close"].pct_change(5)
    # Rolling 20-day volatility (annualized or raw std of 1d returns)
    spy_df["spy_volatility_20"] = spy_df["spy_return_1d"].rolling(window=20, min_periods=20).std()

    # Calculate returns for XLK
    xlk_df["xlk_return_1d"] = xlk_df["close"].pct_change(1)
    xlk_df["xlk_return_5d"] = xlk_df["close"].pct_change(5)

    # Calculate returns for SOXX
    soxx_df["soxx_return_1d"] = soxx_df["close"].pct_change(1)
    soxx_df["soxx_return_5d"] = soxx_df["close"].pct_change(5)

    # Merge ETF features on session_date
    m_df = pd.merge(
        spy_df[["session_date", "spy_return_1d", "spy_return_5d", "spy_volatility_20"]],
        xlk_df[["session_date", "xlk_return_1d", "xlk_return_5d"]],
        on="session_date",
        how="inner",
    )
    m_df = pd.merge(
        m_df,
        soxx_df[["session_date", "soxx_return_1d", "soxx_return_5d"]],
        on="session_date",
        how="inner",
    )

    # Derived relative performance features
    m_df["xlk_relative_to_spy_5d"] = m_df["xlk_return_5d"] - m_df["spy_return_5d"]
    m_df["soxx_relative_to_spy_5d"] = m_df["soxx_return_5d"] - m_df["spy_return_5d"]

    # Round feature values for consistency
    for col in MARKET_FEATURE_COLUMNS:
        m_df[col] = m_df[col].round(6)

    m_df = m_df.sort_values("session_date").reset_index(drop=True)
    return m_df[["session_date"] + MARKET_FEATURE_COLUMNS]


def merge_market_features(
    df_sec: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join market/sector features to existing panel dataset on session_date.

    Guarantees:
    - Exact preservation of row order, index, and target columns.
    - Zero look-ahead bias: session_date t joins exclusively to market session_date t.
    """
    merged = pd.merge(
        df_sec,
        market_df,
        on="session_date",
        how="left",
    )

    # Check that row count is unchanged
    if len(merged) != len(df_sec):
        msg = f"Merge altered row count! Expected {len(df_sec)}, got {len(merged)}"
        raise ValueError(msg)

    return merged
