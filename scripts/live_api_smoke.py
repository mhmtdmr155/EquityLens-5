"""EquityLens-5 | Live API Smoke Test Helper (EL5-012).

Lightweight, safe verification script for Alpaca Market Data v2 and SEC EDGAR.
Strictly read-only; never writes to database, executes trades, or downloads bulk datasets.
If credentials are not configured, halts cleanly without network requests.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from scripts.market_data_probe import (
        EQUITYLENS_SYMBOLS,
        SPACEX_IPO_DATE_STR,
        AlpacaConfig,
        AlpacaDataClient,
        AuthenticationError,
        MarketDataError,
        SECConfigurationError,
        SECDataClient,
        get_config_value,
        load_env_file,
    )
except ModuleNotFoundError:
    from market_data_probe import (  # type: ignore[no-redef]
        EQUITYLENS_SYMBOLS,
        SPACEX_IPO_DATE_STR,
        AlpacaConfig,
        AlpacaDataClient,
        AuthenticationError,
        MarketDataError,
        SECConfigurationError,
        SECDataClient,
        get_config_value,
        load_env_file,
    )


logger = logging.getLogger("EquityLens5.LiveSmoke")


def check_configuration_status() -> dict[str, Any]:
    """Inspect environment and local .env to determine credentials availability."""
    env_vars = load_env_file()
    key_id = get_config_value("APCA_API_KEY_ID", env_vars)
    secret_key = get_config_value("APCA_API_SECRET_KEY", env_vars)
    sec_ua = get_config_value("SEC_USER_AGENT", env_vars)

    missing = []
    if not key_id:
        missing.append("APCA_API_KEY_ID")
    if not secret_key:
        missing.append("APCA_API_SECRET_KEY")
    if not sec_ua:
        missing.append("SEC_USER_AGENT")

    alpaca_ready = bool(key_id and secret_key)
    sec_ready = bool(sec_ua)

    return {
        "alpaca_ready": alpaca_ready,
        "sec_ready": sec_ready,
        "missing_variables": missing,
        "has_all_credentials": alpaca_ready and sec_ready,
    }


def run_alpaca_smoke_test(
    client: AlpacaDataClient,
    symbols: tuple[str, ...] = EQUITYLENS_SYMBOLS,
    start_date: str = "2026-10-19",
    end_date: str = "2026-10-23",
    feed: str = "iex",
    limit: int = 5,
) -> dict[str, dict[str, Any]]:
    """Execute controlled, single-symbol read-only smoke queries on Alpaca."""
    results: dict[str, dict[str, Any]] = {}

    for sym in symbols:
        # Enforce IPO date for SPCX
        query_start = start_date
        if sym.upper() == "SPCX" and query_start < SPACEX_IPO_DATE_STR:
            query_start = SPACEX_IPO_DATE_STR

        try:
            data = client.get_stock_bars(
                symbols=[sym],
                timeframe="1Day",
                start=query_start,
                end=end_date,
                feed=feed,
                limit=limit,
            )
            bars = data.get("bars", {}).get(sym, [])
            results[sym] = {
                "status": "SUCCESS",
                "feed": feed,
                "bar_count": len(bars),
                "first_date": bars[0].get("t") if bars else None,
                "last_date": bars[-1].get("t") if bars else None,
                "fields": list(bars[0].keys()) if bars else [],
            }
        except (AuthenticationError, MarketDataError) as exc:
            results[sym] = {
                "status": "ERROR",
                "feed": feed,
                "error": str(exc),
            }

    return results


def run_sec_smoke_test(
    client: SECDataClient,
    target_tickers: tuple[str, ...] = EQUITYLENS_SYMBOLS,
) -> dict[str, Any]:
    """Execute single read-only probe on SEC EDGAR company tickers."""
    try:
        data = client.get_company_tickers()
        matched: dict[str, dict[str, Any]] = {}
        for item in data.values():
            if isinstance(item, dict):
                ticker = item.get("ticker", "").upper()
                if ticker in target_tickers:
                    matched[ticker] = {
                        "cik_str": item.get("cik_str"),
                        "title": item.get("title"),
                    }
        return {
            "status": "SUCCESS",
            "matched_companies": matched,
            "total_matched": len(matched),
        }
    except (SECConfigurationError, MarketDataError) as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
        }


def main() -> int:
    """Run diagnostic smoke test or report missing credentials cleanly."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("EquityLens-5 Live API Smoke Test (EL5-012)")

    status = check_configuration_status()

    if not status["has_all_credentials"]:
        logger.warning(
            "Live API execution is BLOCKED. Missing environment variables: %s",
            ", ".join(status["missing_variables"]),
        )
        logger.info(
            "To enable live tests locally: Copy .env.example to .env and configure credentials."
        )
        return 1

    logger.info("Credentials detected. Initiating read-only smoke verification...")

    # Alpaca probe
    alpaca_cfg = AlpacaConfig.from_env()
    alpaca_client = AlpacaDataClient(alpaca_cfg)
    logger.info("Running Alpaca smoke queries (Feed: %s)...", alpaca_cfg.default_feed)
    alpaca_results = run_alpaca_smoke_test(alpaca_client)
    for sym, res in alpaca_results.items():
        logger.info(
            "Symbol: %s -> Status: %s | Bars: %s", sym, res.get("status"), res.get("bar_count", 0)
        )

    # SEC probe
    sec_client = SECDataClient.from_env()
    logger.info("Running SEC EDGAR tickers smoke verification...")
    sec_res = run_sec_smoke_test(sec_client)
    logger.info(
        "SEC Status: %s | Matched Target Companies: %d",
        sec_res.get("status"),
        sec_res.get("total_matched", 0),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
