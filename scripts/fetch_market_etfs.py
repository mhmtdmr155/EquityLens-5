"""Script to fetch and cache market/sector ETF daily OHLCV bars (SPY, XLK, SOXX)."""

# ruff: noqa: E402
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.market_data_probe import AlpacaConfig, AlpacaDataClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EquityLens5.FetchMarketETFs")

TARGET_ETFS = ["SPY", "XLK", "SOXX"]
START_DATE = "2024-11-01"
END_DATE = "2026-10-30"
CACHE_DIR = PROJECT_ROOT / "data" / "market_cache"


def fetch_market_etfs(
    target_symbols: list[str] = TARGET_ETFS,
    start: str = START_DATE,
    end: str = END_DATE,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, Any]:
    """Fetch ETF daily bars from Alpaca and cache locally with safety limits."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = AlpacaConfig.from_env()
    client = AlpacaDataClient(config)

    request_count = 0
    results: dict[str, Any] = {}

    # 1. Pilot query (safety check, 5 bars of SPY)
    logger.info("Executing pilot query for SPY (limit=5) ...")
    pilot_resp = client.get_stock_bars(symbols=["SPY"], limit=5, feed="iex")
    request_count += 1
    pilot_bars = pilot_resp.get("bars", {}).get("SPY", [])
    if not pilot_bars:
        msg = "Pilot query failed: no bars returned for SPY"
        raise RuntimeError(msg)
    logger.info(
        "Pilot query SUCCESS: %d bars returned. First bar: %s",
        len(pilot_bars),
        pilot_bars[0].get("t"),
    )

    # 2. Query target range for each ETF
    for symbol in target_symbols:
        cache_file = cache_dir / f"{symbol}_bars.json"

        # Check if already cached
        if cache_file.is_file() and cache_file.stat().st_size > 500:
            logger.info("Found cached bars for %s at %s", symbol, cache_file.name)
            with open(cache_file, encoding="utf-8") as f:
                cached_data = json.load(f)
            results[symbol] = {
                "status": "CACHED",
                "count": len(cached_data),
                "path": str(cache_file),
            }
            continue

        if request_count >= 15:
            logger.warning("Reached maximum allowed API request limit (15). Halting queries.")
            break

        logger.info(
            "Fetching %s bars (%s to %s) via Alpaca iex feed (Request #%d) ...",
            symbol,
            start,
            end,
            request_count + 1,
        )

        all_symbol_bars: list[dict[str, Any]] = []
        page_token = None

        while True:
            if request_count >= 15:
                logger.warning("Reached request limit during pagination for %s.", symbol)
                break

            resp = client.get_stock_bars(
                symbols=[symbol],
                timeframe="1Day",
                start=start,
                end=end,
                feed="iex",
                limit=1000,
                page_token=page_token,
            )
            request_count += 1

            bars = resp.get("bars", {}).get(symbol, [])
            all_symbol_bars.extend(bars)
            page_token = resp.get("next_page_token")

            if not page_token:
                break
            time.sleep(0.3)

        # Save to JSON
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(all_symbol_bars, f, indent=2)

        logger.info(
            "Successfully fetched and cached %d bars for %s in %s",
            len(all_symbol_bars),
            symbol,
            cache_file.name,
        )

        results[symbol] = {
            "status": "DOWNLOADED",
            "count": len(all_symbol_bars),
            "start": all_symbol_bars[0]["t"] if all_symbol_bars else None,
            "end": all_symbol_bars[-1]["t"] if all_symbol_bars else None,
            "path": str(cache_file),
        }
        time.sleep(0.3)

    logger.info("Market ETF fetching completed. Total API requests made: %d", request_count)
    return results


if __name__ == "__main__":
    res = fetch_market_etfs()
    for sym, r in res.items():
        print(f"{sym}: status={r.get('status')}, count={r.get('count')}")
