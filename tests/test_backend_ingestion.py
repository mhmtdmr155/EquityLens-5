"""EquityLens-5 Ingestion Pipeline and Data Persistence Tests (EL5-021).

Tests:
- Raw bar validation (OHLC rules, positive prices, non-negative volume, date formats)
- Security resolution and ticker date validity boundaries
- SpaceX pre-IPO protection across ingestion and model layers
- Idempotent persistence (duplicate bars skipped without error)
- Full 5-company mock persistence pipeline
- Direct security_id ORM event validation
All tests operate on an isolated in-memory SQLite database, completely offline
and independent of environment variables.
"""

from __future__ import annotations

import datetime
import sqlite3
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.ingestion import (
    BarValidationError,
    SecurityResolutionError,
    fetch_and_backfill_symbol,
    persist_symbol_bars,
    resolve_active_security,
    run_historical_backfill,
    validate_raw_bar,
)
from backend.app.models import Base, PriceBar, Security
from backend.app.seed import seed_companies
from scripts.persist_initial_bars import run_persistence_pipeline


@pytest.fixture
def db_session() -> Session:
    """Provide a clean in-memory SQLite database seeded with core companies."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = ON;")

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    seed_companies(session)
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_validate_raw_bar_valid() -> None:
    """Verify raw Alpaca bar validation and type normalization."""
    raw = {
        "t": "2026-10-19T04:00:00Z",
        "o": 235.5,
        "h": 237.0,
        "l": 234.0,
        "c": 236.2,
        "v": 1250000,
    }
    validated = validate_raw_bar(raw, provider="Alpaca", feed="IEX")
    assert validated["session_date"] == datetime.date(2026, 10, 19)
    assert validated["open"] == Decimal("235.5")
    assert validated["high"] == Decimal("237.0")
    assert validated["low"] == Decimal("234.0")
    assert validated["close"] == Decimal("236.2")
    assert validated["volume"] == 1250000
    assert validated["provider"] == "alpaca"
    assert validated["feed"] == "iex"


def test_validate_raw_bar_negative_or_zero_price() -> None:
    """Verify that zero or negative prices raise BarValidationError."""
    bad_zero = {"t": "2026-10-19", "o": 0, "h": 10, "l": 0, "c": 5, "v": 100}
    with pytest.raises(BarValidationError, match="strictly positive"):
        validate_raw_bar(bad_zero)

    bad_neg = {"t": "2026-10-19", "o": 10, "h": 15, "l": -1, "c": 12, "v": 100}
    with pytest.raises(BarValidationError, match="strictly positive"):
        validate_raw_bar(bad_neg)


def test_validate_raw_bar_invalid_ohlc() -> None:
    """Verify that inconsistent OHLC combinations are rejected."""
    # high < low
    with pytest.raises(BarValidationError, match="high.*< low"):
        validate_raw_bar({"t": "2026-10-19", "o": 10, "h": 8, "l": 9, "c": 8.5, "v": 100})

    # high < open
    with pytest.raises(BarValidationError, match="high.*must be >= open"):
        validate_raw_bar({"t": "2026-10-19", "o": 15, "h": 14, "l": 10, "c": 12, "v": 100})

    # low > close
    with pytest.raises(BarValidationError, match=r"low.*must be <= open.*and close"):
        validate_raw_bar({"t": "2026-10-19", "o": 12, "h": 15, "l": 11, "c": 10, "v": 100})


def test_validate_raw_bar_negative_volume() -> None:
    """Verify that negative volume raises BarValidationError."""
    with pytest.raises(BarValidationError, match="non-negative"):
        validate_raw_bar({"t": "2026-10-19", "o": 10, "h": 12, "l": 9, "c": 11, "v": -1})


def test_validate_raw_bar_missing_date() -> None:
    """Verify that missing or malformed timestamp raises BarValidationError."""
    with pytest.raises(BarValidationError, match="missing required timestamp"):
        validate_raw_bar({"o": 10, "h": 12, "l": 9, "c": 11, "v": 100})

    with pytest.raises(BarValidationError, match="Invalid date format"):
        validate_raw_bar({"t": "not-a-date", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100})


def test_resolve_active_security_success(db_session: Session) -> None:
    """Verify active security resolution for historical dates."""
    apple_sec = resolve_active_security(db_session, "AAPL", datetime.date(2026, 10, 20))
    assert apple_sec.ticker == "AAPL"
    assert apple_sec.valid_from <= datetime.date(2026, 10, 20)


def test_resolve_active_security_spacex_valid_and_pre_ipo(db_session: Session) -> None:
    """Verify that SpaceX security is resolved on/after 2026-06-12 and rejected before."""
    # On IPO date
    sec_ipo = resolve_active_security(db_session, "SPCX", datetime.date(2026, 6, 12))
    assert sec_ipo.ticker == "SPCX"
    assert sec_ipo.valid_from == datetime.date(2026, 6, 12)

    # Post IPO date
    sec_post = resolve_active_security(db_session, "SPCX", datetime.date(2026, 10, 20))
    assert sec_post.ticker == "SPCX"

    # Pre IPO date (SpaceX protection)
    with pytest.raises(SecurityResolutionError, match="Pre-IPO or unlisted"):
        resolve_active_security(db_session, "SPCX", datetime.date(2026, 6, 11))


def test_persist_symbol_bars_success_and_idempotency(db_session: Session) -> None:
    """Verify persistence of bars and idempotent duplicate handling."""
    sample_bars = [
        {
            "t": "2026-10-19T04:00:00Z",
            "o": 230.0,
            "h": 235.0,
            "l": 229.0,
            "c": 234.0,
            "v": 5000000,
        },
        {
            "t": "2026-10-20T04:00:00Z",
            "o": 234.0,
            "h": 238.0,
            "l": 233.0,
            "c": 237.5,
            "v": 6000000,
        },
    ]

    # First run: 2 bars inserted
    res1 = persist_symbol_bars(db_session, "AAPL", sample_bars, provider="alpaca", feed="iex")
    assert res1["bars_inserted"] == 2
    assert res1["bars_skipped_duplicate"] == 0

    # Second run: 0 bars inserted, 2 skipped
    res2 = persist_symbol_bars(db_session, "AAPL", sample_bars, provider="alpaca", feed="iex")
    assert res2["bars_inserted"] == 0
    assert res2["bars_skipped_duplicate"] == 2

    # Database count check
    total_bars = db_session.execute(select(func.count(PriceBar.id))).scalar_one()
    assert total_bars == 2


def test_persist_symbol_bars_spacex_pre_ipo_rejected(db_session: Session) -> None:
    """Verify that persist_symbol_bars rejects pre-IPO bars for SpaceX."""
    pre_ipo_bars = [{"t": "2026-06-11", "o": 100.0, "h": 105.0, "l": 98.0, "c": 102.0, "v": 100000}]
    with pytest.raises(SecurityResolutionError, match="Pre-IPO or unlisted"):
        persist_symbol_bars(db_session, "SPCX", pre_ipo_bars, provider="alpaca", feed="iex")


def test_persist_symbol_bars_spacex_post_ipo_accepted(db_session: Session) -> None:
    """Verify that persist_symbol_bars accepts post-IPO bars for SpaceX."""
    post_ipo_bars = [
        {"t": "2026-10-20", "o": 150.0, "h": 160.0, "l": 148.0, "c": 158.0, "v": 1000000}
    ]
    res = persist_symbol_bars(db_session, "SPCX", post_ipo_bars, provider="alpaca", feed="iex")
    assert res["bars_inserted"] == 1
    assert res["bars_skipped_duplicate"] == 0


def test_orm_event_listener_direct_security_id_spacex_rejection(db_session: Session) -> None:
    """Verify that direct insertion with only security_id enforces SpaceX pre-IPO protection."""
    spacex_sec = (
        db_session.execute(select(Security).where(Security.ticker == "SPCX")).scalars().first()
    )
    assert spacex_sec is not None

    # Construct PriceBar setting only security_id (no relationship loaded)
    bad_bar = PriceBar(
        security_id=spacex_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 6, 10),
        open=Decimal("100.00"),
        high=Decimal("110.00"),
        low=Decimal("95.00"),
        close=Decimal("105.00"),
        volume=50000,
    )
    db_session.add(bad_bar)
    with pytest.raises(ValueError, match="SpaceX pre-IPO boundary violation"):
        db_session.commit()
    db_session.rollback()


def test_orm_event_listener_non_spacex_historical_allowed(db_session: Session) -> None:
    """Verify that companies with earlier IPOs (e.g. Apple) accept historical bars."""
    apple_sec = (
        db_session.execute(select(Security).where(Security.ticker == "AAPL")).scalars().first()
    )
    assert apple_sec is not None

    hist_bar = PriceBar(
        security_id=apple_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2020, 1, 15),
        open=Decimal("75.00"),
        high=Decimal("78.00"),
        low=Decimal("74.50"),
        close=Decimal("77.50"),
        volume=1200000,
    )
    db_session.add(hist_bar)
    db_session.commit()

    saved = db_session.get(PriceBar, hist_bar.id)
    assert saved is not None
    assert saved.session_date == datetime.date(2020, 1, 15)


def test_run_persistence_pipeline_mock_all_five_companies(db_session: Session) -> None:
    """Verify full persistence pipeline execution across all 5 companies with mock client."""
    mock_client = MagicMock()

    # Generate 5 mock bars for any queried symbol
    def mock_get_stock_bars(symbols, timeframe, start, end, feed, limit):
        sym = symbols[0]
        base_prices = {"AAPL": 230, "NVDA": 140, "MU": 110, "DELL": 125, "SPCX": 160}
        p = base_prices.get(sym, 100)
        bars = []
        for day in range(19, 24):
            bars.append(
                {
                    "t": f"2026-10-{day}T04:00:00Z",
                    "o": p + day * 0.1,
                    "h": p + day * 0.1 + 2.0,
                    "l": p + day * 0.1 - 1.0,
                    "c": p + day * 0.1 + 1.0,
                    "v": 1000000 * day,
                }
            )
        return {"bars": {sym: bars}}

    mock_client.get_stock_bars.side_effect = mock_get_stock_bars

    # First pipeline execution
    summary1 = run_persistence_pipeline(
        db_session,
        mock_client,
        symbols=("AAPL", "NVDA", "MU", "DELL", "SPCX"),
        start_date="2026-10-19",
        end_date="2026-10-23",
        feed="iex",
    )

    assert len(summary1["symbols_processed"]) == 5
    assert summary1["total_bars_fetched"] == 25
    assert summary1["total_bars_inserted"] == 25
    assert summary1["total_bars_skipped"] == 0

    # Second pipeline execution (idempotency check)
    summary2 = run_persistence_pipeline(
        db_session,
        mock_client,
        symbols=("AAPL", "NVDA", "MU", "DELL", "SPCX"),
        start_date="2026-10-19",
        end_date="2026-10-23",
        feed="iex",
    )

    assert summary2["total_bars_fetched"] == 25
    assert summary2["total_bars_inserted"] == 0
    assert summary2["total_bars_skipped"] == 25

    # Check database total count
    total_in_db = db_session.execute(select(func.count(PriceBar.id))).scalar_one()
    assert total_in_db == 25


def test_fetch_and_backfill_symbol_pagination(db_session: Session) -> None:
    """Verify that fetch_and_backfill_symbol follows next_page_token correctly."""
    mock_client = MagicMock()
    page1 = {
        "bars": {
            "AAPL": [
                {"t": "2026-10-19", "o": 230, "h": 235, "l": 229, "c": 234, "v": 1000},
                {"t": "2026-10-20", "o": 234, "h": 238, "l": 233, "c": 237, "v": 1100},
            ]
        },
        "next_page_token": "token_page_2",
    }
    page2 = {
        "bars": {
            "AAPL": [
                {"t": "2026-10-21", "o": 237, "h": 240, "l": 236, "c": 239, "v": 1200},
            ]
        },
        "next_page_token": None,
    }

    mock_client.get_stock_bars.side_effect = [page1, page2]

    res = fetch_and_backfill_symbol(
        db=db_session,
        client=mock_client,
        ticker="AAPL",
        start_date="2026-10-19",
        end_date="2026-10-21",
        feed="iex",
        page_limit=10,
        max_requests=5,
    )

    assert res["requests_made"] == 2
    assert res["bars_fetched"] == 3
    assert res["bars_inserted"] == 3
    assert res["min_session_date"] == "2026-10-19"
    assert res["max_session_date"] == "2026-10-21"


def test_fetch_and_backfill_symbol_loop_prevention(db_session: Session) -> None:
    """Verify that repeated next_page_token breaks loop and prevents infinite requests."""
    mock_client = MagicMock()
    page_repeated = {
        "bars": {
            "AAPL": [
                {"t": "2026-10-19", "o": 230, "h": 235, "l": 229, "c": 234, "v": 1000},
            ]
        },
        "next_page_token": "duplicate_token_loop",
    }

    mock_client.get_stock_bars.side_effect = [page_repeated, page_repeated]

    res = fetch_and_backfill_symbol(
        db=db_session,
        client=mock_client,
        ticker="AAPL",
        start_date="2026-10-19",
        end_date="2026-10-19",
        feed="iex",
        page_limit=10,
        max_requests=10,
    )

    # Must terminate after second request when token repeats
    assert res["requests_made"] == 2
    assert res["bars_inserted"] == 1
    assert res["bars_skipped_duplicate"] == 1


def test_fetch_and_backfill_spacex_boundary_adjustment(db_session: Session) -> None:
    """Verify that SPCX queries with pre-IPO start_date are clamped to 2026-06-12."""
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = {
        "bars": {
            "SPCX": [
                {"t": "2026-06-12", "o": 120, "h": 130, "l": 118, "c": 128, "v": 500000},
            ]
        },
        "next_page_token": None,
    }

    res = fetch_and_backfill_symbol(
        db=db_session,
        client=mock_client,
        ticker="SPCX",
        start_date="2024-11-01",  # Pre-IPO date
        end_date="2026-10-30",
        feed="iex",
    )

    # Effective start date must be clamped to 2026-06-12
    assert res["start_date"] == "2026-06-12"
    called_start = mock_client.get_stock_bars.call_args.kwargs["start"]
    assert called_start == "2026-06-12"
    assert res["bars_inserted"] == 1


def test_sqlite_cross_process_persistence(tmp_path) -> None:
    """Verify SQLite persistence across independent connection handles."""
    db_file = tmp_path / "persistence_check.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    # Handle 1: Initialize, seed, and insert bars via SQLAlchemy
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    with session_factory() as s1:
        seed_companies(s1)
        aapl_sec = s1.execute(select(Security).where(Security.ticker == "AAPL")).scalar_one()
        bar = PriceBar(
            security_id=aapl_sec.id,
            provider="alpaca",
            feed="iex",
            session_date=datetime.date(2026, 10, 20),
            open=Decimal("230.00"),
            high=Decimal("235.00"),
            low=Decimal("229.00"),
            close=Decimal("234.00"),
            volume=500000,
        )
        s1.add(bar)
        s1.commit()

    engine.dispose()

    # Handle 2: Read directly using standard Python sqlite3 connection
    conn = sqlite3.connect(db_file.as_posix())
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM price_bars")
    total_bars = cursor.fetchone()[0]
    cursor.execute("SELECT session_date, close FROM price_bars WHERE session_date = '2026-10-20'")
    row = cursor.fetchone()
    conn.close()

    assert total_bars == 1
    assert row[0] == "2026-10-20"
    assert float(row[1]) == 234.00


def test_run_historical_backfill_pipeline(db_session: Session) -> None:
    """Verify run_historical_backfill iterates over targets within request limits."""
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = {
        "bars": {
            "AAPL": [
                {
                    "t": "2026-10-20T04:00:00Z",
                    "o": 230.0,
                    "h": 235.0,
                    "l": 229.0,
                    "c": 234.0,
                    "v": 1000,
                }
            ],
            "NVDA": [
                {
                    "t": "2026-10-20T04:00:00Z",
                    "o": 140.0,
                    "h": 145.0,
                    "l": 139.0,
                    "c": 144.0,
                    "v": 2000,
                }
            ],
        },
        "next_page_token": None,
    }

    custom_targets = {
        "AAPL": ("2026-10-15", "2026-10-30"),
        "NVDA": ("2026-10-15", "2026-10-30"),
    }

    summary = run_historical_backfill(
        db=db_session,
        client=mock_client,
        date_ranges=custom_targets,
        max_total_requests=10,
    )

    assert summary["symbols_processed"] == ["AAPL", "NVDA"]
    assert summary["total_requests_made"] == 2
    assert summary["total_bars_fetched"] == 2
    assert summary["total_bars_inserted"] == 2
    assert summary["total_bars_skipped"] == 0
