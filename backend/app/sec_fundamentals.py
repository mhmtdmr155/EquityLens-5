"""EquityLens-5 SEC Fundamentals Parsing and Point-in-Time Alignment (EL5-060).

Extracts and parses US-GAAP financial facts from official SEC CompanyFacts JSON,
calculates standardized financial ratios, and performs strictly leak-free,
as-of point-in-time alignment with historical price session dates.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

logger = logging.getLogger("EquityLens5.SECFundamentals")

# Concept candidates in priority order (ASC 606 standard first, fallback to historical)
CONCEPT_PRIORITY: Final[dict[str, list[str]]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
}

# The 8 derived SEC fundamental features
SEC_FEATURE_COLUMNS: Final[list[str]] = [
    "sec_quarterly_revenue",
    "sec_quarterly_net_income",
    "sec_total_assets",
    "sec_total_liabilities",
    "sec_cash",
    "sec_net_margin",
    "sec_debt_to_assets",
    "sec_revenue_growth_qoq",
]

# Informational metadata columns (not used in model training)
SEC_METADATA_COLUMNS: Final[list[str]] = [
    "sec_filed_date",
    "sec_form",
    "sec_accn",
]


def parse_date_str(d_str: str | None) -> datetime | None:
    """Parse YYYY-MM-DD date string safely."""
    if not d_str:
        return None
    try:
        return datetime.strptime(str(d_str).strip(), "%Y-%m-%d")
    except ValueError:
        return None


def select_best_concept_records(
    us_gaap: dict[str, Any], candidate_names: list[str]
) -> tuple[str | None, list[dict[str, Any]]]:
    """Select the most appropriate US-GAAP concept records based on priority and recency.

    Prefers modern ASC 606 concepts if they contain recent records.
    Filters for legitimate SEC forms (10-K, 10-Q, 10-K/A, 10-Q/A).
    """
    for name in candidate_names:
        if name in us_gaap:
            units = us_gaap[name].get("units", {})
            records = units.get("USD", [])
            valid_forms = [
                r
                for r in records
                if r.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A")
                and r.get("val") is not None
                and r.get("filed") is not None
            ]
            # Check for modern records (filed in or after 2023)
            has_recent = any(
                str(r.get("filed", ""))[:4] in ("2023", "2024", "2025", "2026", "2027")
                for r in valid_forms
            )
            if has_recent or (valid_forms and name == candidate_names[-1]):
                return name, valid_forms

    return None, []


def parse_sec_facts_json(facts_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse raw SEC CompanyFacts JSON and extract chronological filing events.

    Rules enforced:
    - Balance sheet items (Assets, Liabilities, Cash) are instantaneous snapshots.
    - Income statement items (Revenue, Net Income) strictly isolate 3-month quarterly flow
      (duration between 60 and 115 days) to avoid mixing cumulative YTD or annual figures.
    - Derives Net Profit Margin, Liabilities / Assets ratio, and QoQ revenue growth.
    - Each event records filed date, accession number, and form type for provenance.
    """
    us_gaap = facts_data.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return []

    _rev_concept, rev_records = select_best_concept_records(us_gaap, CONCEPT_PRIORITY["revenue"])
    _ni_concept, ni_records = select_best_concept_records(us_gaap, CONCEPT_PRIORITY["net_income"])
    _ast_concept, assets_records = select_best_concept_records(us_gaap, CONCEPT_PRIORITY["assets"])
    _lia_concept, liab_records = select_best_concept_records(
        us_gaap, CONCEPT_PRIORITY["liabilities"]
    )
    _csh_concept, cash_records = select_best_concept_records(us_gaap, CONCEPT_PRIORITY["cash"])

    # Group records by unique filing: (accn, filed, form)
    filings_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    def get_or_create_filing(r: dict[str, Any]) -> dict[str, Any]:
        key = (str(r["accn"]), str(r["filed"]), str(r["form"]))
        if key not in filings_map:
            filings_map[key] = {
                "accn": str(r["accn"]),
                "filed": str(r["filed"]),
                "form": str(r["form"]),
                "fy": r.get("fy"),
                "fp": r.get("fp"),
                "period_end": str(r.get("end")),
                "revenue": None,
                "net_income": None,
                "assets": None,
                "liabilities": None,
                "cash": None,
                "_rev_end": None,
                "_ni_end": None,
            }
        return filings_map[key]

    # 1. Balance sheet items (instantaneous snapshot)
    for r in assets_records:
        entry = get_or_create_filing(r)
        if entry["assets"] is None or r.get("end") == entry["period_end"]:
            entry["assets"] = float(r["val"])

    for r in liab_records:
        entry = get_or_create_filing(r)
        if entry["liabilities"] is None or r.get("end") == entry["period_end"]:
            entry["liabilities"] = float(r["val"])

    for r in cash_records:
        entry = get_or_create_filing(r)
        if entry["cash"] is None or r.get("end") == entry["period_end"]:
            entry["cash"] = float(r["val"])

    # 2. Income flow items (strictly 3-month quarterly flow: 60 <= duration <= 115 days)
    for r in rev_records:
        start_d = parse_date_str(r.get("start"))
        end_d = parse_date_str(r.get("end"))
        if start_d and end_d:
            duration_days = (end_d - start_d).days
            if 60 <= duration_days <= 115:
                entry = get_or_create_filing(r)
                if entry["revenue"] is None or str(r.get("end")) >= (entry["_rev_end"] or ""):
                    entry["revenue"] = float(r["val"])
                    entry["_rev_end"] = str(r.get("end"))

    for r in ni_records:
        start_d = parse_date_str(r.get("start"))
        end_d = parse_date_str(r.get("end"))
        if start_d and end_d:
            duration_days = (end_d - start_d).days
            if 60 <= duration_days <= 115:
                entry = get_or_create_filing(r)
                if entry["net_income"] is None or str(r.get("end")) >= (entry["_ni_end"] or ""):
                    entry["net_income"] = float(r["val"])
                    entry["_ni_end"] = str(r.get("end"))

    # Sort strictly by filed date, then by accn
    filings_list = sorted(filings_map.values(), key=lambda x: (x["filed"], x["accn"]))

    # 3. Derive ratios and QoQ growth across chronological filings
    prev_rev: float | None = None
    for f in filings_list:
        # Net Profit Margin: net_income / revenue
        if f["revenue"] is not None and f["revenue"] > 0 and f["net_income"] is not None:
            f["net_margin"] = round(float(f["net_income"]) / float(f["revenue"]), 6)
        else:
            f["net_margin"] = np.nan

        # Liabilities to Assets: liabilities / assets
        if f["assets"] is not None and f["assets"] > 0 and f["liabilities"] is not None:
            f["debt_to_assets"] = round(float(f["liabilities"]) / float(f["assets"]), 6)
        else:
            f["debt_to_assets"] = np.nan

        # Revenue Growth QoQ: (rev_curr - rev_prev) / rev_prev
        if f["revenue"] is not None:
            if prev_rev is not None and prev_rev > 0:
                f["revenue_growth_qoq"] = round((float(f["revenue"]) - prev_rev) / prev_rev, 6)
            else:
                f["revenue_growth_qoq"] = np.nan
            prev_rev = float(f["revenue"])
        else:
            f["revenue_growth_qoq"] = np.nan

    return filings_list


def align_sec_fundamentals_point_in_time(
    price_df: pd.DataFrame,
    company_filings: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    """Perform strictly leak-free point-in-time join between prices and SEC filings.

    Point-in-time guarantee:
    A filing with filed date D is ONLY observable on trading session dates where
    session_date > D (strictly after filing date). Zero look-ahead bias is permitted.
    If multiple filings were published before session_date, the most recently filed
    one is used as the prevailing state of financial knowledge.
    """
    merged_groups: list[pd.DataFrame] = []

    # Process each ticker independently
    for ticker, group in price_df.groupby("ticker", as_index=False):
        group_sorted = group.sort_values("session_date").copy()
        filings = company_filings.get(str(ticker), [])

        sec_rows: list[dict[str, Any]] = []
        for _, row in group_sorted.iterrows():
            session_date = str(row["session_date"])

            # Find all filings available strictly before session_date
            available = [f for f in filings if f["filed"] < session_date]

            if available:
                latest = available[-1]

                # If latest filing is 10-K and only updated balance sheet,
                # carry forward the latest known quarterly revenue & net income
                rev_val = latest["revenue"]
                ni_val = latest["net_income"]
                rev_growth = latest.get("revenue_growth_qoq", np.nan)
                net_marg = latest.get("net_margin", np.nan)

                if rev_val is None:
                    rev_filings = [f for f in available if f["revenue"] is not None]
                    if rev_filings:
                        last_rev_f = rev_filings[-1]
                        rev_val = last_rev_f["revenue"]
                        if ni_val is None:
                            ni_val = last_rev_f["net_income"]
                        rev_growth = last_rev_f.get("revenue_growth_qoq", np.nan)
                        net_marg = last_rev_f.get("net_margin", np.nan)

                sec_rows.append(
                    {
                        "sec_filed_date": latest["filed"],
                        "sec_form": latest["form"],
                        "sec_accn": latest["accn"],
                        "sec_quarterly_revenue": rev_val,
                        "sec_quarterly_net_income": ni_val,
                        "sec_total_assets": latest["assets"],
                        "sec_total_liabilities": latest["liabilities"],
                        "sec_cash": latest["cash"],
                        "sec_net_margin": net_marg,
                        "sec_debt_to_assets": latest.get("debt_to_assets", np.nan),
                        "sec_revenue_growth_qoq": rev_growth,
                    }
                )
            else:
                # No SEC filing available prior to this session date (e.g. SPCX pre-filing)
                sec_rows.append(
                    {
                        "sec_filed_date": None,
                        "sec_form": None,
                        "sec_accn": None,
                        "sec_quarterly_revenue": np.nan,
                        "sec_quarterly_net_income": np.nan,
                        "sec_total_assets": np.nan,
                        "sec_total_liabilities": np.nan,
                        "sec_cash": np.nan,
                        "sec_net_margin": np.nan,
                        "sec_debt_to_assets": np.nan,
                        "sec_revenue_growth_qoq": np.nan,
                    }
                )

        sec_df = pd.DataFrame(sec_rows, index=group_sorted.index)
        combined_group = pd.concat([group_sorted, sec_df], axis=1)
        merged_groups.append(combined_group)

    final_merged = pd.concat(merged_groups, ignore_index=True)
    # Restore original row ordering by sorting on ticker and session_date
    final_merged = final_merged.sort_values(["ticker", "session_date"]).reset_index(drop=True)
    return final_merged


def load_cached_company_filings(cache_dir: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Load and parse CompanyFacts JSON from local cache directory for all symbols."""
    cache_path = Path(cache_dir)
    filings_by_symbol: dict[str, list[dict[str, Any]]] = {}

    for json_file in cache_path.glob("*_facts.json"):
        symbol = json_file.name.split("_")[0]
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            filings = parse_sec_facts_json(data)
            filings_by_symbol[symbol] = filings
            logger.info(
                "Loaded %d SEC filings for %s from %s", len(filings), symbol, json_file.name
            )
        except Exception as exc:
            logger.error("Failed loading SEC cache for %s: %s", symbol, exc)
            filings_by_symbol[symbol] = []

    return filings_by_symbol
