"""Unit tests for Live API Smoke Test helper (EL5-012).

All tests are strictly offline and mock network interaction.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from scripts.live_api_smoke import (
    check_configuration_status,
    main,
    run_alpaca_smoke_test,
    run_sec_smoke_test,
)
from scripts.market_data_probe import (
    AlpacaDataClient,
    MarketDataError,
    SECConfigurationError,
    SECDataClient,
)


def test_check_configuration_status_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configuration check detects missing variables."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr("scripts.live_api_smoke.load_env_file", lambda *args, **kwargs: {})

    status = check_configuration_status()
    assert status["has_all_credentials"] is False
    assert status["alpaca_ready"] is False
    assert status["sec_ready"] is False
    assert "APCA_API_KEY_ID" in status["missing_variables"]
    assert "APCA_API_SECRET_KEY" in status["missing_variables"]
    assert "SEC_USER_AGENT" in status["missing_variables"]


def test_check_configuration_status_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify configuration check detects all variables when set."""
    monkeypatch.setenv("APCA_API_KEY_ID", "TEST_KEY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "TEST_SECRET")
    monkeypatch.setenv("SEC_USER_AGENT", "Project user@university.edu")

    status = check_configuration_status()
    assert status["has_all_credentials"] is True
    assert status["alpaca_ready"] is True
    assert status["sec_ready"] is True
    assert status["missing_variables"] == []


def test_run_alpaca_smoke_test_mock_success() -> None:
    """Verify Alpaca smoke test parses mock bar response correctly."""
    mock_client = MagicMock(spec=AlpacaDataClient)
    mock_client.get_stock_bars.return_value = {
        "bars": {
            "AAPL": [
                {
                    "t": "2026-10-19T04:00:00Z",
                    "o": 300.0,
                    "h": 305.0,
                    "l": 298.0,
                    "c": 302.0,
                    "v": 1000000,
                }
            ]
        }
    }

    results = run_alpaca_smoke_test(client=mock_client, symbols=("AAPL",))
    assert "AAPL" in results
    assert results["AAPL"]["status"] == "SUCCESS"
    assert results["AAPL"]["bar_count"] == 1
    assert results["AAPL"]["first_date"] == "2026-10-19T04:00:00Z"
    assert "c" in results["AAPL"]["fields"]


def test_run_alpaca_smoke_test_mock_error() -> None:
    """Verify Alpaca smoke test handles errors gracefully."""
    mock_client = MagicMock(spec=AlpacaDataClient)
    mock_client.get_stock_bars.side_effect = MarketDataError("Network failure")

    results = run_alpaca_smoke_test(client=mock_client, symbols=("NVDA",))
    assert "NVDA" in results
    assert results["NVDA"]["status"] == "ERROR"
    assert "Network failure" in results["NVDA"]["error"]


def test_run_sec_smoke_test_mock_success() -> None:
    """Verify SEC smoke test correctly filters target companies."""
    mock_client = MagicMock(spec=SECDataClient)
    mock_client.get_company_tickers.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "2": {"cik_str": 999999, "ticker": "OTHER", "title": "Other Inc."},
    }

    res = run_sec_smoke_test(client=mock_client, target_tickers=("AAPL", "NVDA"))
    assert res["status"] == "SUCCESS"
    assert res["total_matched"] == 2
    assert "AAPL" in res["matched_companies"]
    assert "NVDA" in res["matched_companies"]
    assert "OTHER" not in res["matched_companies"]


def test_run_sec_smoke_test_mock_error() -> None:
    """Verify SEC smoke test handles errors cleanly."""
    mock_client = MagicMock(spec=SECDataClient)
    mock_client.get_company_tickers.side_effect = SECConfigurationError("Missing UA")

    res = run_sec_smoke_test(client=mock_client)
    assert res["status"] == "ERROR"
    assert "Missing UA" in res["error"]


def test_main_live_smoke_missing_credentials_exits_1(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI main() halts with code 1 and logs BLOCKED message when credentials missing."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr("scripts.live_api_smoke.load_env_file", lambda *args, **kwargs: {})

    with caplog.at_level(logging.INFO):
        exit_code = main()

    assert exit_code == 1
    assert "Live API execution is BLOCKED" in caplog.text


def test_main_live_smoke_configured_exits_0(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI main() executes smoke probes and returns 0 when credentials are configured."""
    monkeypatch.setenv("APCA_API_KEY_ID", "PK1234567890ABCDEF")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "SK1234567890FEDCBA")
    monkeypatch.setenv("SEC_USER_AGENT", "EquityLensResearch user@university.edu")

    with patch("scripts.live_api_smoke.run_alpaca_smoke_test") as mock_alpaca:
        mock_alpaca.return_value = {
            "AAPL": {"status": "SUCCESS", "bar_count": 5},
            "NVDA": {"status": "SUCCESS", "bar_count": 5},
        }
        with patch("scripts.live_api_smoke.run_sec_smoke_test") as mock_sec:
            mock_sec.return_value = {"status": "SUCCESS", "total_matched": 5}

            with caplog.at_level(logging.INFO):
                exit_code = main()

    assert exit_code == 0
    assert "Initiating read-only smoke verification" in caplog.text
    assert "SEC Status: SUCCESS" in caplog.text
