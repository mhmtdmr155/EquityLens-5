"""EquityLens-5 Database Session and Engine Management."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import get_settings

logger = logging.getLogger("EquityLens5.Database")


def create_db_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine configured for PostgreSQL or SQLite."""
    url = database_url or get_settings().database_url

    if url.startswith("sqlite"):
        # SQLite configuration for testing
        connect_args: dict[str, Any] = {"check_same_thread": False}
        if ":memory:" in url:
            return create_engine(
                url,
                connect_args=connect_args,
                poolclass=StaticPool,
            )
        return create_engine(url, connect_args=connect_args)

    # PostgreSQL configuration
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_health(db_session: Session) -> dict[str, Any]:
    """Verify database responsiveness using a lightweight probe query."""
    dialect_name = "unknown"
    bind = getattr(db_session, "bind", None)
    if bind is not None and hasattr(bind, "dialect"):
        dialect_name = getattr(bind.dialect, "name", "unknown")

    try:
        db_session.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "connected": True,
            "dialect": dialect_name,
            "error": None,
        }
    except SQLAlchemyError as exc:
        logger.warning("Database health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "connected": False,
            "dialect": dialect_name,
            "error": "Database service is unreachable.",
        }
    except Exception as exc:
        logger.error("Unexpected error during database health check: %s", exc)
        return {
            "status": "unhealthy",
            "connected": False,
            "dialect": dialect_name,
            "error": "Unexpected database error occurred.",
        }
