"""EquityLens-5 Backend Configuration Module."""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def mask_database_url(url: str | None) -> str:
    """Mask credentials in database URL to prevent credential leakage."""
    if not url:
        return "<NOT_CONFIGURED>"
    # Pattern to match postgresql(+driver)?://user:password@host:port/db
    masked = re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", url)
    return masked


class Settings(BaseSettings):
    """Application settings with environment variable precedence."""

    app_name: str = "EquityLens-5 Backend API"
    app_version: str = "0.1.0"
    app_env: str = "development"
    api_prefix: str = "/api/v1"

    # Default to local SQLite development database;
    # overridden by DATABASE_URL when PostgreSQL is active
    database_url: str = "sqlite:///data/equitylens_dev.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def masked_database_url(self) -> str:
        """Return masked database connection string for safe logging."""
        return mask_database_url(self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
