"""EquityLens-5 Backend Models and Database Tests (EL5-020).

Tests SQLAlchemy 2.x data models, table constraints, relationships,
Alembic migrations, seed idempotency, and SpaceX pre-IPO protection.
Uses an isolated in-memory SQLite database.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import mask_database_url
from backend.app.models import Base, Company, PriceBar, Security
from backend.app.seed import seed_companies


@pytest.fixture
def db_session() -> Session:
    """Provide a clean in-memory SQLite session with all tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Enable SQLite foreign key constraints
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = ON;")

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_mask_database_url() -> None:
    """Verify that credentials in database URLs are cleanly masked."""
    assert mask_database_url(None) == "<NOT_CONFIGURED>"
    assert mask_database_url("") == "<NOT_CONFIGURED>"

    raw_pg = "postgresql+psycopg://myuser:secretpass123@localhost:5432/equitylens"
    masked = mask_database_url(raw_pg)
    assert "secretpass123" not in masked
    assert "myuser:***@localhost:5432/equitylens" in masked


def test_seed_initial_companies_and_idempotency(db_session: Session) -> None:
    """Verify that seed creates the exact 5 companies and running twice is idempotent."""
    # First seed execution
    companies_first = seed_companies(db_session)
    assert len(companies_first) == 5

    companies_in_db = db_session.execute(select(Company)).scalars().all()
    assert len(companies_in_db) == 5

    securities_in_db = db_session.execute(select(Security)).scalars().all()
    assert len(securities_in_db) == 5

    # Check companies by CIK
    ciks = {c.sec_cik: c for c in companies_in_db}
    assert "0000320193" in ciks  # Apple
    assert "0001045810" in ciks  # NVIDIA
    assert "0000723125" in ciks  # Micron
    assert "0001571996" in ciks  # Dell
    assert "0001181412" in ciks  # SpaceX

    # Second seed execution (idempotency check)
    companies_second = seed_companies(db_session)
    assert len(companies_second) == 5

    # Counts must remain exactly 5
    assert len(db_session.execute(select(Company)).scalars().all()) == 5
    assert len(db_session.execute(select(Security)).scalars().all()) == 5


def test_model_relationships(db_session: Session) -> None:
    """Verify Company -> Security -> PriceBar relationship tree and cascade."""
    company = Company(company_name="Test Corp", sec_cik="0009999999")
    db_session.add(company)
    db_session.flush()

    security = Security(
        company_id=company.id,
        ticker="TST",
        exchange="NASDAQ",
        valid_from=datetime.date(2020, 1, 1),
    )
    db_session.add(security)
    db_session.flush()

    bar = PriceBar(
        security_id=security.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 10, 20),
        open=Decimal("150.00"),
        high=Decimal("155.00"),
        low=Decimal("149.50"),
        close=Decimal("153.25"),
        volume=1000000,
    )
    db_session.add(bar)
    db_session.commit()

    # Query back
    queried_company = db_session.get(Company, company.id)
    assert queried_company is not None
    assert len(queried_company.securities) == 1
    assert queried_company.securities[0].ticker == "TST"
    assert len(queried_company.securities[0].price_bars) == 1
    assert queried_company.securities[0].price_bars[0].close == Decimal("153.25")


def test_unique_company_constraints(db_session: Session) -> None:
    """Verify unique constraints on company_name and sec_cik."""
    c1 = Company(company_name="Alpha", sec_cik="0001111111")
    db_session.add(c1)
    db_session.commit()

    # Duplicate name
    c2 = Company(company_name="Alpha", sec_cik="0002222222")
    db_session.add(c2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Duplicate CIK
    c3 = Company(company_name="Beta", sec_cik="0001111111")
    db_session.add(c3)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_unique_price_bar_constraint(db_session: Session) -> None:
    """Verify duplicate bar insertion raises IntegrityError."""
    seed_companies(db_session)
    apple_sec = (
        db_session.execute(select(Security).where(Security.ticker == "AAPL")).scalars().first()
    )
    assert apple_sec is not None

    bar1 = PriceBar(
        security_id=apple_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 10, 20),
        open=Decimal("200.00"),
        high=Decimal("205.00"),
        low=Decimal("199.00"),
        close=Decimal("204.00"),
        volume=500000,
    )
    db_session.add(bar1)
    db_session.commit()

    # Duplicate bar on same security, provider, feed, session_date
    bar2 = PriceBar(
        security_id=apple_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 10, 20),
        open=Decimal("201.00"),
        high=Decimal("206.00"),
        low=Decimal("200.00"),
        close=Decimal("205.00"),
        volume=600000,
    )
    db_session.add(bar2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_check_constraint_negative_prices_and_ohlc(db_session: Session) -> None:
    """Verify check constraints for positive prices, valid high/low, and non-negative volume."""
    seed_companies(db_session)
    nvda_sec = (
        db_session.execute(select(Security).where(Security.ticker == "NVDA")).scalars().first()
    )
    assert nvda_sec is not None

    # Negative open price
    bad_bar1 = PriceBar(
        security_id=nvda_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 10, 20),
        open=Decimal("-10.00"),
        high=Decimal("100.00"),
        low=Decimal("90.00"),
        close=Decimal("95.00"),
        volume=1000,
    )
    db_session.add(bad_bar1)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # High lower than Low
    bad_bar2 = PriceBar(
        security_id=nvda_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 10, 21),
        open=Decimal("100.00"),
        high=Decimal("90.00"),
        low=Decimal("110.00"),
        close=Decimal("95.00"),
        volume=1000,
    )
    db_session.add(bad_bar2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Negative volume
    bad_bar3 = PriceBar(
        security_id=nvda_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 10, 22),
        open=Decimal("100.00"),
        high=Decimal("110.00"),
        low=Decimal("90.00"),
        close=Decimal("95.00"),
        volume=-50,
    )
    db_session.add(bad_bar3)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_spacex_pre_ipo_price_bar_rejected(db_session: Session) -> None:
    """Verify that any price bar prior to 2026-06-12 is rejected for SpaceX."""
    seed_companies(db_session)
    spacex_sec = (
        db_session.execute(select(Security).where(Security.ticker == "SPCX")).scalars().first()
    )
    assert spacex_sec is not None
    assert spacex_sec.valid_from == datetime.date(2026, 6, 12)

    # Attempt to insert pre-IPO price bar (e.g. from former SPAC ETF)
    pre_ipo_bar = PriceBar(
        security_id=spacex_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 6, 11),  # One day prior to IPO
        open=Decimal("100.00"),
        high=Decimal("110.00"),
        low=Decimal("95.00"),
        close=Decimal("105.00"),
        volume=50000,
    )
    db_session.add(pre_ipo_bar)
    with pytest.raises(ValueError, match="SpaceX pre-IPO boundary violation"):
        db_session.commit()
    db_session.rollback()


def test_spacex_post_ipo_price_bar_allowed(db_session: Session) -> None:
    """Verify that price bars on or after 2026-06-12 are accepted for SpaceX."""
    seed_companies(db_session)
    spacex_sec = (
        db_session.execute(select(Security).where(Security.ticker == "SPCX")).scalars().first()
    )
    assert spacex_sec is not None

    post_ipo_bar = PriceBar(
        security_id=spacex_sec.id,
        provider="alpaca",
        feed="iex",
        session_date=datetime.date(2026, 6, 12),  # IPO date
        open=Decimal("120.00"),
        high=Decimal("135.00"),
        low=Decimal("118.00"),
        close=Decimal("130.00"),
        volume=2500000,
    )
    db_session.add(post_ipo_bar)
    db_session.commit()

    saved_bar = db_session.get(PriceBar, post_ipo_bar.id)
    assert saved_bar is not None
    assert saved_bar.session_date == datetime.date(2026, 6, 12)
    assert saved_bar.close == Decimal("130.00")


def test_alembic_migration_upgrade_and_downgrade(tmp_path) -> None:
    """Verify that Alembic migration 0001 applies and downgrades cleanly."""
    db_file = tmp_path / "test_migration.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    alembic_cfg = Config("backend/alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Test upgrade to head
    command.upgrade(alembic_cfg, "head")

    # Verify tables exist via reflection
    engine = create_engine(db_url)
    with engine.connect() as conn:
        tables = conn.dialect.get_table_names(conn)
        assert "companies" in tables
        assert "securities" in tables
        assert "price_bars" in tables

    # Test downgrade to base
    command.downgrade(alembic_cfg, "base")
    with engine.connect() as conn:
        tables = conn.dialect.get_table_names(conn)
        assert "companies" not in tables
        assert "securities" not in tables
        assert "price_bars" not in tables
