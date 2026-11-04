"""EquityLens-5 | Market Data & SEC Exploration Client Prototype (EL5-011-D02).

This module provides a secure, lightweight, and testable client for querying
Alpaca Market Data v2 and SEC EDGAR endpoints using only the Python standard library.
Enforces strict historical boundaries for SpaceX (SPCX IPO date: 2026-06-12).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("EquityLens5.MarketDataProbe")

# Constants
ALPACA_DATA_V2_URL = "https://data.alpaca.markets/v2"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SPACEX_IPO_DATE_STR = "2026-06-12"
SPACEX_IPO_DATE = datetime.date(2026, 6, 12)
EQUITYLENS_SYMBOLS = ("AAPL", "NVDA", "MU", "DELL", "SPCX")


def mask_credential(val: str | None, visible_chars: int = 4) -> str:
    """Safely mask API key or secret for secure logging and display."""
    if not val:
        return "<NOT_CONFIGURED>"
    if len(val) <= visible_chars * 2:
        return "*" * len(val)
    return f"{val[:visible_chars]}{'*' * (len(val) - visible_chars * 2)}{val[-visible_chars:]}"


def parse_date(date_val: str | datetime.date | datetime.datetime) -> datetime.date:
    """Parse an ISO date string or return date object."""
    if isinstance(date_val, datetime.datetime):
        return date_val.date()
    if isinstance(date_val, datetime.date):
        return date_val
    if not isinstance(date_val, str) or not date_val.strip():
        raise ValueError("Date value cannot be empty.")

    clean_str = date_val.strip()
    if "T" in clean_str:
        clean_str = clean_str.split("T")[0]

    try:
        return datetime.date.fromisoformat(clean_str)
    except ValueError as err:
        raise ValueError(
            f"Invalid date format: '{date_val}'. Expected ISO format YYYY-MM-DD."
        ) from err


def validate_symbol_query_dates(
    symbols: list[str],
    start: str | None,
    end: str | None,
) -> list[str]:
    """Validate query date ranges and enforce SpaceX historical boundary."""
    if not symbols:
        raise ValueError("At least one stock symbol must be specified.")

    clean_symbols = [s.strip().upper() for s in symbols if s and s.strip()]
    if not clean_symbols:
        raise ValueError("No valid, non-empty stock symbols provided.")

    start_date: datetime.date | None = None
    end_date: datetime.date | None = None

    if start is not None:
        start_date = parse_date(start)
    if end is not None:
        end_date = parse_date(end)

    if start_date and end_date and start_date > end_date:
        raise ValueError(
            f"Invalid date range: start date '{start}' cannot be after end date '{end}'."
        )

    if "SPCX" in clean_symbols:
        if start_date is None:
            raise DataIntegrityError(
                "Explicit 'start' date is mandatory when querying SPCX "
                f"(SpaceX IPO was {SPACEX_IPO_DATE_STR})."
            )
        if start_date < SPACEX_IPO_DATE:
            raise DataIntegrityError(
                f"SPCX historical query rejected: SpaceX IPO date is {SPACEX_IPO_DATE_STR}. "
                f"Requested start date '{start}' precedes IPO and belongs to an unrelated entity."
            )

    return clean_symbols


class MarketDataError(Exception):
    """Base exception for market data client errors."""


class AuthenticationError(MarketDataError):
    """Raised when API credentials are missing, invalid, or unauthorized (HTTP 401)."""


class RateLimitError(MarketDataError):
    """Raised when API rate limits are exceeded (HTTP 429)."""


class DataIntegrityError(MarketDataError):
    """Raised when historical boundaries or data integrity constraints are violated."""


class SECConfigurationError(MarketDataError):
    """Raised when SEC Fair Access User-Agent is missing or misconfigured."""


def load_env_file(env_path: Path | str | None = None) -> dict[str, str]:
    """Parse a simple .env file into a dictionary using the standard library.

    Does not override existing variables in os.environ.
    Ignores comments, empty lines, and strips surrounding quotes.
    Never loads .env.example as an active credentials source.
    """
    if env_path is None:
        candidate = Path.cwd() / ".env"
        if not candidate.is_file():
            candidate = Path(__file__).resolve().parent.parent / ".env"
        env_path = candidate

    target_path = Path(env_path)
    if target_path.name == ".env.example":
        logger.warning(".env.example is a template and will not be loaded as credentials source.")
        return {}

    if not target_path.is_file():
        return {}

    loaded_vars: dict[str, str] = {}
    try:
        with open(target_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if key:
                    loaded_vars[key] = val
    except OSError as err:
        logger.warning("Could not read .env file at %s: %s", target_path, err)
        return {}

    return loaded_vars


def get_config_value(key: str, env_vars: dict[str, str] | None = None) -> str:
    """Retrieve configuration value with os.environ having precedence over .env file."""
    val = os.getenv(key)
    if val is not None and val.strip():
        return val.strip()
    if env_vars and key in env_vars:
        file_val = env_vars[key].strip()
        if file_val:
            return file_val
    return ""


@dataclass(frozen=True)
class AlpacaConfig:
    """Configuration container for Alpaca API credentials."""

    api_key_id: str = field(repr=False)
    api_secret_key: str = field(repr=False)
    base_url: str = ALPACA_DATA_V2_URL
    default_feed: str = "iex"

    @property
    def masked_key_id(self) -> str:
        """Return masked key ID for safe logging."""
        return mask_credential(self.api_key_id)

    @classmethod
    def from_env(cls, env_path: Path | str | None = None) -> AlpacaConfig:
        """Load configuration from os.environ or project .env file."""
        env_file_vars = load_env_file(env_path)
        key_id = get_config_value("APCA_API_KEY_ID", env_file_vars)
        secret = get_config_value("APCA_API_SECRET_KEY", env_file_vars)
        if not key_id or not secret:
            raise AuthenticationError(
                "Alpaca API credentials missing. Please set APCA_API_KEY_ID "
                "and APCA_API_SECRET_KEY in your local environment or .env file."
            )
        return cls(api_key_id=key_id, api_secret_key=secret)


class AlpacaDataClient:
    """Lightweight Alpaca Market Data v2 client using urllib."""

    def __init__(self, config: AlpacaConfig) -> None:
        self.config = config

    def build_bars_url(
        self,
        symbols: list[str],
        timeframe: str = "1Day",
        start: str | None = None,
        end: str | None = None,
        feed: str = "iex",
        limit: int = 10,
        page_token: str | None = None,
    ) -> str:
        """Construct secure query URL for stock bars."""
        clean_symbols = validate_symbol_query_dates(symbols=symbols, start=start, end=end)

        if limit <= 0 or limit > 10000:
            raise ValueError(f"Invalid limit {limit}: must be between 1 and 10000.")

        params: dict[str, str | int] = {
            "symbols": ",".join(clean_symbols),
            "timeframe": timeframe,
            "feed": feed.strip().lower(),
            "limit": limit,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if page_token:
            params["page_token"] = page_token

        query_string = urllib.parse.urlencode(params)
        return f"{self.config.base_url}/stocks/bars?{query_string}"

    def get_stock_bars(
        self,
        symbols: list[str],
        timeframe: str = "1Day",
        start: str | None = None,
        end: str | None = None,
        feed: str = "iex",
        limit: int = 10,
        page_token: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Fetch stock bars with safe headers, validation, and robust error handling."""
        url = self.build_bars_url(
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=feed,
            limit=limit,
            page_token=page_token,
        )

        headers = {
            "APCA-API-KEY-ID": self.config.api_key_id,
            "APCA-API-SECRET-KEY": self.config.api_secret_key,
            "User-Agent": "EquityLens5-MarketDataProbe/1.0",
            "Accept": "application/json",
        }

        req = urllib.request.Request(url, headers=headers, method="GET")

        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    raw_bytes = response.read()
                    try:
                        data = json.loads(raw_bytes.decode("utf-8"))
                        if not isinstance(data, dict):
                            raise MarketDataError("Expected JSON object in API response.")
                    except json.JSONDecodeError as err:
                        raise MarketDataError(f"Malformed JSON response: {err}") from err

                    bars = data.get("bars", {})
                    if not isinstance(bars, dict):
                        raise MarketDataError("Expected 'bars' dictionary in API response.")

                    # Enforce data integrity on SPCX bars
                    if "SPCX" in bars and isinstance(bars["SPCX"], list):
                        for bar in bars["SPCX"]:
                            if isinstance(bar, dict) and "t" in bar:
                                bar_date = parse_date(str(bar["t"]))
                                if bar_date < SPACEX_IPO_DATE:
                                    raise DataIntegrityError(
                                        f"Data integrity violation: SPCX bar contains pre-IPO "
                                        f"timestamp '{bar['t']}' prior to {SPACEX_IPO_DATE_STR}. "
                                        "Historical SPAC ETF data cannot be returned as SpaceX."
                                    )

                    return data

            except urllib.error.HTTPError as err:
                status = err.code
                if status == 401:
                    masked = self.config.masked_key_id
                    raise AuthenticationError(
                        f"Authentication failed (HTTP 401) using Key ID {masked}. "
                        "Verify APCA_API_KEY_ID and APCA_API_SECRET_KEY."
                    ) from err
                if status == 403:
                    raise MarketDataError(
                        "Access forbidden (HTTP 403). Your account tier does not have "
                        f"access to feed='{feed}' or specified parameters."
                    ) from err
                if status == 429:
                    if attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise RateLimitError(
                        "Alpaca rate limit exceeded (HTTP 429). Max 200 requests/minute "
                        "on free tier."
                    ) from err
                if 500 <= status < 600:
                    if attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise MarketDataError(f"Alpaca remote server error (HTTP {status}).") from err
                raise MarketDataError(
                    f"Alpaca API request failed (HTTP {status}): {err.reason}"
                ) from err

            except TimeoutError as err:
                raise MarketDataError(f"Request timed out querying Alpaca API: {err}") from err

            except urllib.error.URLError as err:
                if isinstance(err.reason, socket.timeout):
                    raise MarketDataError(
                        f"Request timed out querying Alpaca API: {err.reason}"
                    ) from err
                raise MarketDataError(f"Network connection failed: {err.reason}") from err

        raise MarketDataError("Exhausted retries querying Alpaca Market Data.")


class SECDataClient:
    """SEC EDGAR client strictly adhering to SEC Fair Access policy."""

    def __init__(self, user_agent: str | None = None) -> None:
        self.user_agent = (user_agent or "").strip()

    @classmethod
    def from_env(cls, env_path: Path | str | None = None) -> SECDataClient:
        """Load SECDataClient from os.environ or project .env file."""
        env_file_vars = load_env_file(env_path)
        ua = get_config_value("SEC_USER_AGENT", env_file_vars)
        return cls(user_agent=ua)

    def validate_user_agent(self) -> str:
        """Ensure SEC User-Agent is declared with legitimate contact info."""
        if not self.user_agent:
            raise SECConfigurationError(
                "SEC_USER_AGENT yapılandırılmamış. SEC Fair Access kuralları gereğince "
                "gerçek bir iletişim bilgisi (örn: Name contact@domain.com) sağlanmalıdır."
            )
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        if not re.search(email_pattern, self.user_agent):
            raise SECConfigurationError(
                "SEC Fair Access policy requires a valid contact email in SEC_USER_AGENT. "
                "Current value does not contain a valid email address format."
            )
        if "admin@equitylens5.edu" in self.user_agent.lower():
            raise SECConfigurationError(
                "Geçersiz veya sahte e-posta adresi ('admin@equitylens5.edu') tespit edildi. "
                "SEC Fair Access politikası uyarınca gerçek kurumsal/kişisel e-posta girilmelidir."
            )
        return self.user_agent

    def get_company_tickers(self, timeout: float = 10.0) -> dict[str, Any]:
        """Fetch official SEC company tickers directory."""
        ua = self.validate_user_agent()
        headers = {
            "User-Agent": ua,
            "Accept": "application/json",
        }
        req = urllib.request.Request(SEC_TICKERS_URL, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise MarketDataError("Expected JSON object in SEC tickers response.")
                return data
        except urllib.error.HTTPError as err:
            raise MarketDataError(
                f"SEC EDGAR request failed (HTTP {err.code}): {err.reason}"
            ) from err
        except urllib.error.URLError as err:
            raise MarketDataError(f"Network error querying SEC: {err.reason}") from err


def main() -> None:
    """Run interactive diagnostic probe for market data and SEC credentials."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("EquityLens-5 Market Data Probe (EL5-011-D02)")
    logger.info("Monitored Tickers: %s", ", ".join(EQUITYLENS_SYMBOLS))
    logger.info("SpaceX (SPCX) IPO Date Enforcement: %s", SPACEX_IPO_DATE_STR)

    # Check Alpaca configuration
    try:
        alpaca_cfg = AlpacaConfig.from_env()
        logger.info(
            "Alpaca configuration found. Key ID: %s (Feed: %s)",
            alpaca_cfg.masked_key_id,
            alpaca_cfg.default_feed,
        )
    except AuthenticationError as exc:
        logger.warning("Alpaca configuration status: %s", exc)

    # Check SEC configuration
    sec_client = SECDataClient.from_env()
    if not sec_client.user_agent:
        logger.warning("SEC_USER_AGENT not configured. Set SEC_USER_AGENT in .env.")
    else:
        try:
            sec_client.validate_user_agent()
            logger.info("SEC EDGAR User-Agent biçim doğrulaması başarılı.")
        except SECConfigurationError as exc:
            logger.error("SEC EDGAR User-Agent biçim hatası: %s", exc)


if __name__ == "__main__":
    main()
