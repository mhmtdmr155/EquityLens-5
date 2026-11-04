"""Unit tests for Market Data & SEC probe prototype (EL5-011-D01).

All network calls are strictly mocked; no real external calls or API keys are required.
Validates security, SpaceX historical integrity, and configuration management.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.market_data_probe import (
    AlpacaConfig,
    AlpacaDataClient,
    AuthenticationError,
    DataIntegrityError,
    MarketDataError,
    RateLimitError,
    SECConfigurationError,
    SECDataClient,
    get_config_value,
    load_env_file,
    main,
    mask_credential,
    parse_date,
)

# --- 1. Credential Security & Masking Tests ---

# --- 1. Credential Security & Masking Tests ---


def test_mask_credential() -> None:
    """Verify safe credential masking for display and logs."""
    assert mask_credential(None) == "<NOT_CONFIGURED>"
    assert mask_credential("") == "<NOT_CONFIGURED>"
    assert mask_credential("PK1234567890ABCDEF") == "PK12**********CDEF"
    assert mask_credential("SHORT") == "*****"


def test_alpaca_config_credentials_repr_hidden() -> None:
    """Verify that neither api_key_id nor api_secret_key appears in repr() or str()."""
    key_id = "PK1234567890ABCDEF"
    secret = "SUPER_SECRET_ALPACA_KEY_9999"
    cfg = AlpacaConfig(api_key_id=key_id, api_secret_key=secret)

    repr_str = repr(cfg)
    str_val = str(cfg)

    # Neither raw key ID nor raw secret key should appear in repr or str
    assert key_id not in repr_str
    assert secret not in repr_str
    assert key_id not in str_val
    assert secret not in str_val

    # Masked key ID should still work properly via property
    assert cfg.masked_key_id == "PK12**********CDEF"


def test_alpaca_config_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that missing environment variables raise AuthenticationError."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    with pytest.raises(AuthenticationError, match="credentials missing"):
        AlpacaConfig.from_env()


def test_alpaca_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify loading credentials correctly from environment variables."""
    monkeypatch.setenv("APCA_API_KEY_ID", "TEST_KEY_ID")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "TEST_SECRET_KEY")

    cfg = AlpacaConfig.from_env()
    assert cfg.api_key_id == "TEST_KEY_ID"
    assert cfg.api_secret_key == "TEST_SECRET_KEY"


def test_headers_contain_credentials_and_url_does_not() -> None:
    """Verify credentials are sent in HTTP headers, never in the URL query string."""
    key_id = "PK_TEST_HEADER_KEY_123"
    secret = "SK_TEST_HEADER_SECRET_456"
    cfg = AlpacaConfig(api_key_id=key_id, api_secret_key=secret)
    client = AlpacaDataClient(cfg)

    url = client.build_bars_url(symbols=["AAPL"])
    assert key_id not in url
    assert secret not in url

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"bars": {"AAPL": []}}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        client.get_stock_bars(symbols=["AAPL"])

        call_args = mock_urlopen.call_args[0]
        req = call_args[0]
        assert req.headers["Apca-api-key-id"] == key_id
        assert req.headers["Apca-api-secret-key"] == secret
        assert key_id not in req.full_url
        assert secret not in req.full_url


# --- 2. Environment File (.env) Parsing Tests ---


def test_load_env_file_custom_path(tmp_path: Path) -> None:
    """Verify standard library parsing of .env file with quotes and comments."""
    test_env = tmp_path / ".env.test"
    test_env.write_text(
        "# Test configuration file\n"
        "APCA_API_KEY_ID=TEST_ID_FROM_FILE\n"
        'APCA_API_SECRET_KEY="QUOTED_SECRET_VAL"\n'
        "SEC_USER_AGENT='TestAgent user@domain.com'\n"
        "\n"
        "EMPTY_VAL=\n",
        encoding="utf-8",
    )

    vars_dict = load_env_file(test_env)
    assert vars_dict["APCA_API_KEY_ID"] == "TEST_ID_FROM_FILE"
    assert vars_dict["APCA_API_SECRET_KEY"] == "QUOTED_SECRET_VAL"
    assert vars_dict["SEC_USER_AGENT"] == "TestAgent user@domain.com"
    assert vars_dict["EMPTY_VAL"] == ""


def test_load_env_file_ignores_env_example(tmp_path: Path) -> None:
    """Verify that .env.example is never loaded as credentials."""
    example_file = tmp_path / ".env.example"
    example_file.write_text("APCA_API_KEY_ID=SHOULD_NOT_BE_READ\n", encoding="utf-8")

    vars_dict = load_env_file(example_file)
    assert vars_dict == {}


def test_env_precedence_os_environ_over_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that real os.environ takes strict precedence over .env file."""
    test_env = tmp_path / ".env.override"
    test_env.write_text("APCA_API_KEY_ID=FILE_KEY_ID\n", encoding="utf-8")

    # Set in OS environment
    monkeypatch.setenv("APCA_API_KEY_ID", "OS_ENV_KEY_ID")

    file_vars = load_env_file(test_env)
    val = get_config_value("APCA_API_KEY_ID", file_vars)

    assert val == "OS_ENV_KEY_ID"


# --- 3. Parameter and Date Range Validation Tests ---


def test_parse_date_valid_and_invalid() -> None:
    """Verify ISO date parser with various formats."""
    import datetime

    assert parse_date("2026-06-12") == datetime.date(2026, 6, 12)
    assert parse_date("2026-06-12T04:00:00Z") == datetime.date(2026, 6, 12)

    with pytest.raises(ValueError, match="Invalid date format"):
        parse_date("not-a-date")

    with pytest.raises(ValueError, match="Date value cannot be empty"):
        parse_date("")


def test_build_bars_url_valid() -> None:
    """Verify proper URL and query parameter encoding."""
    cfg = AlpacaConfig(api_key_id="k", api_secret_key="s")
    client = AlpacaDataClient(cfg)

    url = client.build_bars_url(
        symbols=["AAPL", "nvda"],
        timeframe="1Day",
        start="2026-10-01",
        end="2026-10-05",
        feed="iex",
        limit=5,
    )

    assert "https://data.alpaca.markets/v2/stocks/bars?" in url
    assert "symbols=AAPL%2CNVDA" in url
    assert "timeframe=1Day" in url
    assert "feed=iex" in url
    assert "limit=5" in url
    assert "start=2026-10-01" in url
    assert "end=2026-10-05" in url


def test_build_bars_url_empty_or_whitespace_symbols() -> None:
    """Verify error on empty or whitespace-only symbol list."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    with pytest.raises(ValueError, match="At least one stock symbol"):
        client.build_bars_url(symbols=[])

    with pytest.raises(ValueError, match="No valid, non-empty stock symbols"):
        client.build_bars_url(symbols=["   ", ""])


def test_build_bars_url_invalid_limit() -> None:
    """Verify error on non-positive or excessive limit values."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    with pytest.raises(ValueError, match="Invalid limit"):
        client.build_bars_url(symbols=["AAPL"], limit=0)

    with pytest.raises(ValueError, match="Invalid limit"):
        client.build_bars_url(symbols=["AAPL"], limit=10001)


def test_build_bars_url_start_after_end() -> None:
    """Verify error when start date is after end date."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    with pytest.raises(ValueError, match="start date '2026-10-10' cannot be after end date"):
        client.build_bars_url(symbols=["AAPL"], start="2026-10-10", end="2026-10-01")


# --- 4. SpaceX (SPCX) Historical Boundary & Data Integrity Tests ---


def test_spacex_historical_boundary_rejects_pre_ipo() -> None:
    """Verify rejection of pre-2026-06-12 queries for SPCX without making network calls."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    mock_urlopen = MagicMock()
    with patch("urllib.request.urlopen", mock_urlopen):
        # 1. 2025 date rejected
        with pytest.raises(DataIntegrityError, match="SPCX historical query rejected"):
            client.get_stock_bars(symbols=["SPCX"], start="2025-01-01")

        # 2. 2026-06-11 (day before IPO) rejected
        with pytest.raises(DataIntegrityError, match="SPCX historical query rejected"):
            client.get_stock_bars(symbols=["SPCX"], start="2026-06-11")

        # 3. Missing explicit start date rejected
        with pytest.raises(DataIntegrityError, match="Explicit 'start' date is mandatory"):
            client.get_stock_bars(symbols=["SPCX"], start=None)

    # CRITICAL: Verify NO network request was ever dispatched
    assert mock_urlopen.call_count == 0


def test_spacex_historical_boundary_allows_ipo_and_future() -> None:
    """Verify acceptance of SPCX queries on or after 2026-06-12."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"bars": {"SPCX": []}}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Exactly on IPO date: allowed
        res1 = client.get_stock_bars(symbols=["SPCX"], start="2026-06-12")
        assert "bars" in res1

        # Post-IPO date: allowed
        res2 = client.get_stock_bars(symbols=["SPCX"], start="2026-07-01")
        assert "bars" in res2


def test_non_spacex_symbols_allow_historical_dates() -> None:
    """Verify that AAPL, NVDA, MU, DELL queries for historical data are NOT blocked."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"bars": {"AAPL": []}}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Historical queries for standard stocks must succeed
        res = client.get_stock_bars(symbols=["AAPL"], start="2018-01-01")
        assert "bars" in res

        res_multi = client.get_stock_bars(symbols=["NVDA", "MU", "DELL"], start="2020-05-01")
        assert "bars" in res_multi


def test_mixed_symbols_with_spacex_protection() -> None:
    """Verify that batch queries containing SPCX along with others enforce the IPO boundary."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    # Batch with AAPL and SPCX with pre-IPO date: MUST REJECT
    with pytest.raises(DataIntegrityError, match="SPCX historical query rejected"):
        client.get_stock_bars(symbols=["AAPL", "SPCX"], start="2020-01-01")

    # Batch with AAPL and SPCX on or after IPO: MUST SUCCEED
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"bars": {"AAPL": [], "SPCX": []}}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.get_stock_bars(symbols=["AAPL", "SPCX"], start="2026-06-12")
        assert "bars" in res


def test_spacex_response_pre_ipo_contamination_detected() -> None:
    """Verify DataIntegrityError if API response contains pre-IPO SPCX bars."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    # Contaminated payload returned by API (e.g. from former SPAC ETF)
    contaminated_payload = {
        "bars": {
            "SPCX": [
                {
                    "t": "2025-05-10T04:00:00Z",
                    "o": 25.0,
                    "h": 26.0,
                    "l": 24.5,
                    "c": 25.5,
                    "v": 10000,
                }
            ]
        }
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(contaminated_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(DataIntegrityError, match="pre-IPO timestamp '2025-05-10T04:00:00Z'"):
            client.get_stock_bars(symbols=["SPCX"], start="2026-06-12")


# --- 5. Mock HTTP and Error Handling Tests ---


def test_get_stock_bars_success_mock() -> None:
    """Verify successful parsing of bars response using mocked connection."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))
    mock_payload = {
        "bars": {
            "AAPL": [
                {
                    "t": "2026-10-01T04:00:00Z",
                    "o": 330.0,
                    "h": 335.0,
                    "l": 328.0,
                    "c": 332.0,
                    "v": 45000000,
                }
            ]
        },
        "next_page_token": None,
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.get_stock_bars(symbols=["AAPL"])

    assert "bars" in res
    assert "AAPL" in res["bars"]
    assert len(res["bars"]["AAPL"]) == 1
    assert res["bars"]["AAPL"][0]["c"] == 332.0


def test_get_stock_bars_http_401_mock() -> None:
    """Verify HTTP 401 raises AuthenticationError with masked Key ID and no secret leak."""
    raw_key = "PK9876543210FEDCBA"
    raw_secret = "SUPER_SECRET_VALUE_XYZ"
    client = AlpacaDataClient(AlpacaConfig(api_key_id=raw_key, api_secret_key=raw_secret))
    http_error = urllib.error.HTTPError(
        url="https://data.alpaca.markets/v2/stocks/bars",
        code=401,
        msg="Unauthorized",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"message":"forbidden"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(AuthenticationError) as exc_info:
            client.get_stock_bars(symbols=["AAPL"])

    err_msg = str(exc_info.value)
    assert "HTTP 401" in err_msg
    assert "PK98**********DCBA" in err_msg
    assert raw_key not in err_msg
    assert raw_secret not in err_msg


def test_get_stock_bars_http_403_mock() -> None:
    """Verify HTTP 403 raises MarketDataError indicating tier or parameter restriction."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))
    http_error = urllib.error.HTTPError(
        url="https://data.alpaca.markets/v2/stocks/bars",
        code=403,
        msg="Forbidden",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"message":"forbidden feed"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(MarketDataError, match="HTTP 403"):
            client.get_stock_bars(symbols=["AAPL"], feed="sip")


def test_get_stock_bars_http_429_mock() -> None:
    """Verify HTTP 429 raises RateLimitError after retries."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))
    http_error = urllib.error.HTTPError(
        url="https://data.alpaca.markets/v2/stocks/bars",
        code=429,
        msg="Too Many Requests",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"message":"rate limit exceeded"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with patch("time.sleep"):  # Avoid real delays in test
            with pytest.raises(RateLimitError, match="HTTP 429"):
                client.get_stock_bars(symbols=["AAPL"], max_retries=1)


def test_get_stock_bars_http_500_mock() -> None:
    """Verify HTTP 500 raises MarketDataError after retries."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))
    http_error = urllib.error.HTTPError(
        url="https://data.alpaca.markets/v2/stocks/bars",
        code=500,
        msg="Internal Server Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b"server error"),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with patch("time.sleep"):
            with pytest.raises(MarketDataError, match="HTTP 500"):
                client.get_stock_bars(symbols=["AAPL"], max_retries=1)


def test_get_stock_bars_timeout_mock() -> None:
    """Verify TimeoutError raises MarketDataError with descriptive message."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        with pytest.raises(MarketDataError, match="Request timed out"):
            client.get_stock_bars(symbols=["AAPL"])


def test_get_stock_bars_malformed_json_mock() -> None:
    """Verify handling of invalid JSON response."""
    client = AlpacaDataClient(AlpacaConfig(api_key_id="k", api_secret_key="s"))
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<html>Not JSON</html>"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(MarketDataError, match="Malformed JSON"):
            client.get_stock_bars(symbols=["AAPL"])


# --- 6. SEC EDGAR Client & User-Agent Tests ---


def test_sec_client_missing_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SECConfigurationError when SEC_USER_AGENT is not provided."""
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    client = SECDataClient(user_agent="")

    with pytest.raises(SECConfigurationError, match="SEC_USER_AGENT yapılandırılmamış"):
        client.validate_user_agent()


def test_sec_client_invalid_user_agent_format() -> None:
    """Verify SECConfigurationError when User-Agent lacks an email address."""
    client = SECDataClient(user_agent="MyAcademicProjectNoEmail")

    with pytest.raises(SECConfigurationError, match="valid contact email"):
        client.validate_user_agent()


def test_sec_client_rejects_fake_email() -> None:
    """Verify explicit rejection of known fabricated domain/email."""
    client = SECDataClient(user_agent="EquityLens5 admin@equitylens5.edu")

    with pytest.raises(SECConfigurationError, match="sahte e-posta adresi"):
        client.validate_user_agent()


def test_sec_client_success_mock() -> None:
    """Verify SEC tickers fetch using mocked response."""
    client = SECDataClient(user_agent="EquityLens5Project user@university.edu")
    mock_payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1181412, "ticker": "SPCX", "title": "SPACE EXPLORATION TECHNOLOGIES CORP"},
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        data = client.get_company_tickers()

    assert "0" in data
    assert data["0"]["ticker"] == "AAPL"
    assert data["1"]["cik_str"] == 1181412


# --- 7. CLI Main Execution & Diagnostic Tests ---


def test_main_cli_unconfigured(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI main() runs without crashing when environment variables are not set."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with caplog.at_level(logging.INFO):
        main()

    assert "EquityLens-5 Market Data Probe" in caplog.text
    assert "Alpaca configuration status: Alpaca API credentials missing" in caplog.text
    assert "SEC_USER_AGENT not configured" in caplog.text


def test_main_cli_configured_valid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI main() reports properly and contains NO unmasked credentials."""
    raw_key = "PK1234567890ABCDEF"
    raw_secret = "SK1234567890FEDCBA"
    monkeypatch.setenv("APCA_API_KEY_ID", raw_key)
    monkeypatch.setenv("APCA_API_SECRET_KEY", raw_secret)
    monkeypatch.setenv("SEC_USER_AGENT", "EquityLensResearch research@university.edu")

    with caplog.at_level(logging.INFO):
        main()

    assert "Alpaca configuration found. Key ID: PK12**********CDEF" in caplog.text
    assert "SEC EDGAR User-Agent biçim doğrulaması başarılı" in caplog.text
    # Verify neither unmasked key ID nor unmasked secret key appears in logs
    assert raw_key not in caplog.text
    assert raw_secret not in caplog.text


def test_main_cli_configured_invalid_sec_ua(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify CLI main() catches and reports invalid SEC UA format properly."""
    monkeypatch.setenv("APCA_API_KEY_ID", "PK1234567890ABCDEF")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "SK1234567890FEDCBA")
    monkeypatch.setenv("SEC_USER_AGENT", "InvalidUserAgentWithoutEmail")

    with caplog.at_level(logging.INFO):
        main()

    assert "SEC EDGAR User-Agent biçim hatası" in caplog.text
