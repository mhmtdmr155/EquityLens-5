"""Script to fetch and cache SEC CompanyFacts JSON for tracked symbols."""

# ruff: noqa: E402
from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.market_data_probe import SECDataClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COMPANIES = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "NVDA": {"cik": "0001045810", "name": "NVIDIA Corporation"},
    "MU": {"cik": "0000723125", "name": "Micron Technology, Inc."},
    "DELL": {"cik": "0001571996", "name": "Dell Technologies Inc."},
    "SPCX": {"cik": "0001181412", "name": "Space Exploration Technologies Corp."},
}

CACHE_DIR = PROJECT_ROOT / "data" / "sec_cache"


def fetch_and_cache_facts() -> dict[str, dict]:
    """Fetch CompanyFacts for all 5 companies and cache locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = SECDataClient.from_env()
    ua = client.validate_user_agent()

    results = {}
    for symbol, info in COMPANIES.items():
        cik = info["cik"]
        cache_path = CACHE_DIR / f"{symbol}_facts.json"

        if cache_path.is_file() and cache_path.stat().st_size > 1000:
            logger.info("Found cached facts for %s (%s) at %s", symbol, cik, cache_path.name)
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            results[symbol] = {
                "status": "CACHED",
                "cik": cik,
                "path": str(cache_path),
                "fact_keys": list(data.get("facts", {}).keys()),
                "size_bytes": cache_path.stat().st_size,
            }
            continue

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        logger.info("Fetching SEC facts for %s (%s) from %s ...", symbol, cik, url)

        headers = {
            "User-Agent": ua,
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            logger.info(
                "Successfully cached facts for %s (%d bytes, entity: %s)",
                symbol,
                len(raw_bytes),
                data.get("entityName"),
            )
            results[symbol] = {
                "status": "DOWNLOADED",
                "cik": cik,
                "path": str(cache_path),
                "entity_name": data.get("entityName"),
                "fact_keys": list(data.get("facts", {}).keys()),
                "size_bytes": len(raw_bytes),
            }
        except urllib.error.HTTPError as err:
            logger.error("HTTP error fetching %s: HTTP %d %s", symbol, err.code, err.reason)
            results[symbol] = {
                "status": "HTTP_ERROR",
                "code": err.code,
                "reason": err.reason,
            }
        except Exception as exc:
            logger.error("Error fetching %s: %s", symbol, exc)
            results[symbol] = {
                "status": "ERROR",
                "error": str(exc),
            }

        # SEC Fair Access etiquette
        time.sleep(0.5)

    return results


if __name__ == "__main__":
    res = fetch_and_cache_facts()
    for s, r in res.items():
        print(f"{s}: status={r.get('status')} size={r.get('size_bytes', 0)} bytes")
