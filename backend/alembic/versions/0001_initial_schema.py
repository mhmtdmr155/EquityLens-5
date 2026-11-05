"""Initial schema for companies, securities, and price_bars

Revision ID: 0001
Revises:
Create Date: 2026-11-05 16:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create companies table
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("sec_cik", sa.String(length=10), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_name", name="uq_companies_company_name"),
        sa.UniqueConstraint("sec_cik", name="uq_companies_sec_cik"),
    )
    op.create_index(op.f("ix_companies_company_name"), "companies", ["company_name"], unique=True)
    op.create_index(op.f("ix_companies_sec_cik"), "companies", ["sec_cik"], unique=True)

    # 2. Create securities table
    op.create_table(
        "securities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("exchange", sa.String(length=20), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "ticker",
            "exchange",
            "valid_from",
            name="uq_securities_ticker_exchange_valid_from",
        ),
        sa.CheckConstraint(
            "valid_to IS NULL OR valid_to >= valid_from",
            name="ck_securities_valid_date_range",
        ),
    )
    op.create_index(op.f("ix_securities_company_id"), "securities", ["company_id"], unique=False)
    op.create_index(op.f("ix_securities_ticker"), "securities", ["ticker"], unique=False)

    # 3. Create price_bars table
    op.create_table(
        "price_bars",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("security_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("feed", sa.String(length=50), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("high", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("low", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("close", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["security_id"], ["securities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "security_id",
            "provider",
            "feed",
            "session_date",
            name="uq_price_bars_security_provider_feed_date",
        ),
        sa.CheckConstraint("open > 0", name="ck_price_bars_open_positive"),
        sa.CheckConstraint("high > 0", name="ck_price_bars_high_positive"),
        sa.CheckConstraint("low > 0", name="ck_price_bars_low_positive"),
        sa.CheckConstraint("close > 0", name="ck_price_bars_close_positive"),
        sa.CheckConstraint("volume >= 0", name="ck_price_bars_volume_non_negative"),
        sa.CheckConstraint("high >= low", name="ck_price_bars_high_gte_low"),
        sa.CheckConstraint("high >= open", name="ck_price_bars_high_gte_open"),
        sa.CheckConstraint("high >= close", name="ck_price_bars_high_gte_close"),
        sa.CheckConstraint("low <= open", name="ck_price_bars_low_lte_open"),
        sa.CheckConstraint("low <= close", name="ck_price_bars_low_lte_close"),
    )
    op.create_index(op.f("ix_price_bars_security_id"), "price_bars", ["security_id"], unique=False)
    op.create_index(
        op.f("ix_price_bars_session_date"), "price_bars", ["session_date"], unique=False
    )


def downgrade() -> None:
    op.drop_table("price_bars")
    op.drop_table("securities")
    op.drop_table("companies")
