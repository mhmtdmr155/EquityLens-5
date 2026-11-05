"""EquityLens-5 SQLAlchemy Database Models (SQLAlchemy 2.x).

Defines core financial entities:
- Company: Legal corporate entity with SEC CIK identifier.
- Security: Tradable financial instrument (ticker + exchange + validity dates).
- PriceBar: Daily OHLCV market price record with provider and feed lineage.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    event,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.orm import Mapper


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.x models."""

    pass


class Company(Base):
    """Corporate entity tracked by EquityLens-5."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    sec_cik: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    securities: Mapped[list[Security]] = relationship(
        "Security",
        back_populates="company",
        cascade="all, delete-orphan",
        order_by="Security.id",
    )

    def __repr__(self) -> str:
        return f"<Company(id={self.id}, name='{self.company_name}', cik='{self.sec_cik}')>"


class Security(Base):
    """Tradable security associated with a company.

    Decoupled from company identity to support historical ticker reuse,
    symbol changes, and strict IPO / listing boundary dates.
    """

    __tablename__ = "securities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False)
    valid_from: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)

    company: Mapped[Company] = relationship("Company", back_populates="securities")
    price_bars: Mapped[list[PriceBar]] = relationship(
        "PriceBar",
        back_populates="security",
        cascade="all, delete-orphan",
        order_by="PriceBar.session_date",
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "exchange",
            "valid_from",
            name="uq_securities_ticker_exchange_valid_from",
        ),
        CheckConstraint(
            "valid_to IS NULL OR valid_to >= valid_from",
            name="ck_securities_valid_date_range",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Security(id={self.id}, ticker='{self.ticker}', "
            f"exchange='{self.exchange}', valid_from={self.valid_from})>"
        )


class PriceBar(Base):
    """Daily OHLCV price bar with provider/feed lineage and strict validation."""

    __tablename__ = "price_bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("securities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    feed: Mapped[str] = mapped_column(String(50), nullable=False)
    session_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    security: Mapped[Security] = relationship("Security", back_populates="price_bars")

    __table_args__ = (
        UniqueConstraint(
            "security_id",
            "provider",
            "feed",
            "session_date",
            name="uq_price_bars_security_provider_feed_date",
        ),
        CheckConstraint("open > 0", name="ck_price_bars_open_positive"),
        CheckConstraint("high > 0", name="ck_price_bars_high_positive"),
        CheckConstraint("low > 0", name="ck_price_bars_low_positive"),
        CheckConstraint("close > 0", name="ck_price_bars_close_positive"),
        CheckConstraint("volume >= 0", name="ck_price_bars_volume_non_negative"),
        CheckConstraint("high >= low", name="ck_price_bars_high_gte_low"),
        CheckConstraint("high >= open", name="ck_price_bars_high_gte_open"),
        CheckConstraint("high >= close", name="ck_price_bars_high_gte_close"),
        CheckConstraint("low <= open", name="ck_price_bars_low_lte_open"),
        CheckConstraint("low <= close", name="ck_price_bars_low_lte_close"),
    )

    def __repr__(self) -> str:
        return (
            f"<PriceBar(security_id={self.security_id}, date={self.session_date}, "
            f"close={self.close}, volume={self.volume})>"
        )


@event.listens_for(PriceBar, "before_insert")
@event.listens_for(PriceBar, "before_update")
def enforce_security_validity_dates(
    mapper: Mapper[PriceBar], connection: Connection, target: PriceBar
) -> None:
    """Enforce security validity boundaries and SpaceX pre-IPO protection.

    Prevents pre-IPO historical SPAC ETF data from contaminating SpaceX records.
    """
    valid_from: datetime.date | None = None
    valid_to: datetime.date | None = None
    ticker_name: str | None = None

    if target.security is not None:
        valid_from = target.security.valid_from
        valid_to = target.security.valid_to
        ticker_name = target.security.ticker
    elif target.security_id is not None:
        stmt = select(Security.valid_from, Security.valid_to, Security.ticker).where(
            Security.id == target.security_id
        )
        row = connection.execute(stmt).first()
        if row is not None:
            valid_from, valid_to, ticker_name = row[0], row[1], row[2]

    if valid_from is not None and target.session_date < valid_from:
        raise ValueError(
            f"Data integrity error: Price bar date {target.session_date} precedes "
            f"security valid_from {valid_from} for ticker '{ticker_name}'. "
            "SpaceX pre-IPO boundary violation."
        )

    if valid_to is not None and target.session_date > valid_to:
        raise ValueError(
            f"Data integrity error: Price bar date {target.session_date} exceeds "
            f"security valid_to {valid_to} for ticker '{ticker_name}'."
        )
