"""EquityLens-5 Market Data Ingestion & Persistence Service.

Provides a robust, multi-layer validation pipeline for market price data:
1. Raw bar validation (types, positive prices, OHLC consistency, non-negative volume).
2. Security resolution (matches active valid_from/valid_to range, preventing ticker misattribution).
3. SpaceX pre-IPO protection (rejects any data dated before 2026-06-12).
4. Idempotent persistence (detects existing bars and prevents duplicate key errors).
"""

from __future__ import annotations

import datetime
import logging
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import PriceBar, Security

logger = logging.getLogger("EquityLens5.Ingestion")
SPACEX_IPO_DATE_STR = "2026-06-12"


class IngestionError(Exception):
    """Base exception for ingestion pipeline errors."""

    pass


class BarValidationError(IngestionError):
    """Raised when an incoming price bar violates OHLC or range rules."""

    pass


class SecurityResolutionError(IngestionError):
    """Raised when a ticker cannot be safely mapped to a valid Security entity."""

    pass


def parse_session_date(date_val: str | datetime.date | datetime.datetime) -> datetime.date:
    """Parse session date from ISO string, datetime, or date."""
    if isinstance(date_val, datetime.datetime):
        return date_val.date()
    if isinstance(date_val, datetime.date):
        return date_val
    if isinstance(date_val, str):
        cleaned = date_val.strip()
        try:
            if "T" in cleaned:
                return datetime.datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date()
            return datetime.date.fromisoformat(cleaned)
        except ValueError as exc:
            raise BarValidationError(f"Invalid date format: {date_val!r}") from exc
    raise BarValidationError(f"Invalid date format: {date_val!r}")


def validate_raw_bar(
    raw_bar: dict[str, Any],
    provider: str = "alpaca",
    feed: str = "iex",
) -> dict[str, Any]:
    """Validate raw OHLCV bar dictionary and return normalized types."""
    # Date extraction (supports Alpaca 't' or standard 'session_date')
    raw_t = raw_bar.get("t") or raw_bar.get("session_date")
    if not raw_t:
        raise BarValidationError("Bar is missing required timestamp/date field ('t').")
    session_date = parse_session_date(raw_t)

    # Price fields (supports Alpaca 'o', 'h', 'l', 'c' or 'open', 'high', etc.)
    try:
        open_val = Decimal(str(raw_bar.get("o") if "o" in raw_bar else raw_bar.get("open")))
        high_val = Decimal(str(raw_bar.get("h") if "h" in raw_bar else raw_bar.get("high")))
        low_val = Decimal(str(raw_bar.get("l") if "l" in raw_bar else raw_bar.get("low")))
        close_val = Decimal(str(raw_bar.get("c") if "c" in raw_bar else raw_bar.get("close")))
    except (InvalidOperation, TypeError) as exc:
        raise BarValidationError(f"Invalid numeric price values in bar: {raw_bar}") from exc

    # Volume extraction
    raw_v = raw_bar.get("v") if "v" in raw_bar else raw_bar.get("volume", 0)
    try:
        volume_val = int(raw_v)
    except (ValueError, TypeError) as exc:
        raise BarValidationError(f"Invalid volume value '{raw_v}' in bar.") from exc

    # Validation rules
    if open_val <= 0 or high_val <= 0 or low_val <= 0 or close_val <= 0:
        raise BarValidationError(
            f"Prices must be strictly positive. Got o={open_val}, h={high_val}, "
            f"l={low_val}, c={close_val}"
        )

    if volume_val < 0:
        raise BarValidationError(f"Volume must be non-negative. Got {volume_val}")

    if high_val < low_val:
        raise BarValidationError(f"OHLC inconsistency: high ({high_val}) < low ({low_val})")

    if high_val < open_val or high_val < close_val:
        raise BarValidationError(
            f"OHLC inconsistency: high ({high_val}) must be >= "
            f"open ({open_val}) and close ({close_val})"
        )

    if low_val > open_val or low_val > close_val:
        raise BarValidationError(
            f"OHLC inconsistency: low ({low_val}) must be <= "
            f"open ({open_val}) and close ({close_val})"
        )

    return {
        "session_date": session_date,
        "open": open_val,
        "high": high_val,
        "low": low_val,
        "close": close_val,
        "volume": volume_val,
        "provider": provider.lower().strip(),
        "feed": feed.lower().strip(),
    }


def resolve_active_security(
    db: Session,
    ticker: str,
    session_date: datetime.date,
) -> Security:
    """Resolve the active Security record matching ticker and validity range."""
    cleaned_ticker = ticker.upper().strip()
    stmt = (
        select(Security)
        .where(
            Security.ticker == cleaned_ticker,
            Security.valid_from <= session_date,
            (Security.valid_to.is_(None) | (Security.valid_to >= session_date)),
        )
        .order_by(Security.valid_from.desc())
    )
    security = db.execute(stmt).scalars().first()

    if security is None:
        raise SecurityResolutionError(
            f"No valid Security entity found for ticker '{cleaned_ticker}' on date {session_date}. "
            "Pre-IPO or unlisted dates cannot be resolved to this instrument."
        )

    return security


def persist_symbol_bars(
    db: Session,
    ticker: str,
    bars_data: list[dict[str, Any]],
    provider: str = "alpaca",
    feed: str = "iex",
) -> dict[str, Any]:
    """Validate and idempotently persist a list of price bars for a single ticker."""
    inserted_count = 0
    skipped_duplicate_count = 0
    resolved_security_id: int | None = None

    for raw_bar in bars_data:
        validated = validate_raw_bar(raw_bar, provider=provider, feed=feed)
        session_date = validated["session_date"]

        # Resolve security matching date validity
        security = resolve_active_security(db, ticker, session_date)
        resolved_security_id = security.id

        # Check for existing bar (idempotency check)
        exists_stmt = select(PriceBar.id).where(
            PriceBar.security_id == security.id,
            PriceBar.provider == validated["provider"],
            PriceBar.feed == validated["feed"],
            PriceBar.session_date == session_date,
        )
        existing_id = db.execute(exists_stmt).scalars().first()

        if existing_id is not None:
            skipped_duplicate_count += 1
            continue

        # Create model instance
        bar = PriceBar(
            security_id=security.id,
            provider=validated["provider"],
            feed=validated["feed"],
            session_date=session_date,
            open=validated["open"],
            high=validated["high"],
            low=validated["low"],
            close=validated["close"],
            volume=validated["volume"],
        )
        db.add(bar)
        inserted_count += 1

    db.commit()

    return {
        "ticker": ticker.upper().strip(),
        "security_id": resolved_security_id,
        "total_bars_received": len(bars_data),
        "bars_inserted": inserted_count,
        "bars_skipped_duplicate": skipped_duplicate_count,
        "status": "SUCCESS",
    }


def fetch_and_backfill_symbol(
    db: Session,
    client: Any,
    ticker: str,
    start_date: str,
    end_date: str,
    feed: str = "iex",
    page_limit: int = 1000,
    max_requests: int = 15,
) -> dict[str, Any]:
    """Fetch paginated historical daily bars from Alpaca and persist idempotently.

    Enforces strict pagination controls:
    - Tracks seen page tokens to prevent infinite request loops.
    - Limits total HTTP requests per symbol.
    - Enforces SpaceX pre-IPO date boundary (start >= 2026-06-12).
    """
    clean_ticker = ticker.upper().strip()
    effective_start = start_date
    if clean_ticker == "SPCX" and effective_start < SPACEX_IPO_DATE_STR:
        effective_start = SPACEX_IPO_DATE_STR

    total_fetched = 0
    total_inserted = 0
    total_skipped = 0
    requests_count = 0
    seen_tokens: set[str] = set()
    current_token: str | None = None
    all_dates: list[datetime.date] = []

    while requests_count < max_requests:
        requests_count += 1
        response = client.get_stock_bars(
            symbols=[clean_ticker],
            timeframe="1Day",
            start=effective_start,
            end=end_date,
            feed=feed,
            limit=page_limit,
            page_token=current_token,
        )
        bars = response.get("bars", {}).get(clean_ticker, [])
        total_fetched += len(bars)

        if bars:
            res = persist_symbol_bars(db, clean_ticker, bars, provider="alpaca", feed=feed)
            total_inserted += res["bars_inserted"]
            total_skipped += res["bars_skipped_duplicate"]
            for b in bars:
                raw_t = b.get("t") or b.get("session_date")
                if raw_t:
                    all_dates.append(parse_session_date(raw_t))

        next_token = response.get("next_page_token")
        if not next_token or next_token in seen_tokens:
            break

        seen_tokens.add(next_token)
        current_token = next_token

    min_date = min(all_dates).isoformat() if all_dates else None
    max_date = max(all_dates).isoformat() if all_dates else None

    return {
        "ticker": clean_ticker,
        "start_date": effective_start,
        "end_date": end_date,
        "feed": feed,
        "requests_made": requests_count,
        "bars_fetched": total_fetched,
        "bars_inserted": total_inserted,
        "bars_skipped_duplicate": total_skipped,
        "min_session_date": min_date,
        "max_session_date": max_date,
        "status": "SUCCESS",
    }


def run_historical_backfill(
    db: Session,
    client: Any,
    date_ranges: dict[str, tuple[str, str]] | None = None,
    feed: str = "iex",
    max_total_requests: int = 50,
) -> dict[str, Any]:
    """Execute complete historical backfill pipeline across all target companies."""
    targets = date_ranges or {
        "AAPL": ("2024-11-01", "2026-10-30"),
        "NVDA": ("2024-11-01", "2026-10-30"),
        "MU": ("2024-11-01", "2026-10-30"),
        "DELL": ("2024-11-01", "2026-10-30"),
        "SPCX": ("2026-06-12", "2026-10-30"),
    }

    summary: dict[str, Any] = {
        "symbols_processed": [],
        "total_requests_made": 0,
        "total_bars_fetched": 0,
        "total_bars_inserted": 0,
        "total_bars_skipped": 0,
        "symbol_details": {},
    }

    remaining_requests = max_total_requests

    for sym, (s_date, e_date) in targets.items():
        if remaining_requests <= 0:
            logger.warning(
                "Reached maximum allowed API requests (%d). Halting backfill.",
                max_total_requests,
            )
            break

        symbol_res = fetch_and_backfill_symbol(
            db=db,
            client=client,
            ticker=sym,
            start_date=s_date,
            end_date=e_date,
            feed=feed,
            max_requests=min(15, remaining_requests),
        )

        reqs = symbol_res["requests_made"]
        remaining_requests -= reqs

        summary["symbols_processed"].append(sym)
        summary["total_requests_made"] += reqs
        summary["total_bars_fetched"] += symbol_res["bars_fetched"]
        summary["total_bars_inserted"] += symbol_res["bars_inserted"]
        summary["total_bars_skipped"] += symbol_res["bars_skipped_duplicate"]
        summary["symbol_details"][sym] = symbol_res

    return summary
