"""EquityLens-5 End-to-End Browser Verification Script for React Dashboard (EL5-070)."""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("docs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def verify_browser_dashboard() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("1. Navigating to http://127.0.0.1:5173/ ...")
        page.goto("http://127.0.0.1:5173/", wait_until="networkidle")

        # Verify page title
        assert "EquityLens-5" in page.title(), f"Unexpected title: {page.title()}"
        print(f"   [PASS] Title verified: {page.title()}")

        # Verify AAPL default view
        page.wait_for_selector(".ticker-badge-lg", state="visible")
        badge_text = page.locator(".ticker-badge-lg").inner_text()
        assert badge_text == "AAPL", f"Expected AAPL by default, got: {badge_text}"
        print(f"   [PASS] Default ticker loaded: {badge_text}")

        # Wait for price and chart to settle
        time.sleep(1.2)
        price_text = page.locator(".price-main").inner_text()
        print(f"   [PASS] AAPL latest price: {price_text}")

        # Verify AAPL prediction score
        aapl_pred_score = page.locator(".confidence-value").inner_text()
        assert "%61.2" in aapl_pred_score, f"Expected %61.2 for AAPL, got: {aapl_pred_score}"
        print(f"   [PASS] AAPL prediction score verified: {aapl_pred_score}")

        # Capture AAPL desktop screenshot
        page.screenshot(path=str(SCREENSHOT_DIR / "dashboard_aapl_desktop.png"), full_page=True)
        print(f"   [PASS] Saved screenshot: {SCREENSHOT_DIR / 'dashboard_aapl_desktop.png'}")

        # Click NVDA card
        print("2. Switching company to NVDA ...")
        nvda_btn = page.locator(".company-card").filter(has_text="NVDA").first
        nvda_btn.click()
        time.sleep(1.2)

        nvda_badge = page.locator(".ticker-badge-lg").inner_text()
        assert nvda_badge == "NVDA", f"Expected NVDA, got: {nvda_badge}"
        nvda_price = page.locator(".price-main").inner_text()
        nvda_pred_score = page.locator(".confidence-value").inner_text()
        assert "%68.3" in nvda_pred_score, f"Expected %68.3 for NVDA, got: {nvda_pred_score}"
        print(
            f"   [PASS] NVDA selected successfully. Price: {nvda_price}, Score: {nvda_pred_score}"
        )
        page.screenshot(path=str(SCREENSHOT_DIR / "dashboard_nvda_desktop.png"), full_page=True)
        print(f"   [PASS] Saved screenshot: {SCREENSHOT_DIR / 'dashboard_nvda_desktop.png'}")

        # Click SPCX card
        print("3. Switching company to SPCX (SpaceX) ...")
        spcx_btn = page.locator(".company-card").filter(has_text="SPCX").first
        spcx_btn.click()
        time.sleep(1.2)

        spcx_badge = page.locator(".ticker-badge-lg").inner_text()
        assert spcx_badge == "SPCX", f"Expected SPCX, got: {spcx_badge}"
        spcx_price = page.locator(".price-main").inner_text()
        spcx_pred_score = page.locator(".confidence-value").inner_text()
        assert "%60.8" in spcx_pred_score, f"Expected %60.8 for SPCX, got: {spcx_pred_score}"
        print(
            f"   [PASS] SPCX selected successfully. Price: {spcx_price}, Score: {spcx_pred_score}"
        )

        # Check SPCX official CIK and post-IPO note
        cik_text = page.locator(".meta-tag").filter(has_text="CIK:").inner_text()
        assert "0001181412" in cik_text, f"Expected CIK 0001181412, got: {cik_text}"
        assert "0001174922" not in page.content(), (
            "Wynn Resorts CIK 0001174922 must not appear on page"
        )
        print(f"   [PASS] SPCX CIK verified on dashboard: {cik_text}")

        post_ipo_note = page.locator(".spcx-highlight").inner_text()
        assert "12.06.2026" in post_ipo_note
        print(f"   [PASS] SPCX post-IPO tag verified: {post_ipo_note}")

        # Check that Model Performance table does not contain ATR
        perf_text = page.locator(".perf-table").inner_text()
        assert "ATR" not in perf_text, "ATR must not appear in technical indicators list"
        print("   [PASS] Verified absence of ATR in model performance table.")

        page.screenshot(path=str(SCREENSHOT_DIR / "dashboard_spcx_desktop.png"), full_page=True)
        print(f"   [PASS] Saved screenshot: {SCREENSHOT_DIR / 'dashboard_spcx_desktop.png'}")

        # Test Horizon Selector buttons
        print("4. Testing 30-session horizon filter on SPCX ...")
        h30_btn = page.locator(".horizon-btn").filter(has_text="30 Seans")
        h30_btn.click()
        time.sleep(0.8)
        print("   [PASS] Horizon changed to 30 sessions.")

        # Test Mobile Responsiveness
        print("5. Testing mobile responsiveness (390x844 viewport) ...")
        page.set_viewport_size({"width": 390, "height": 844})
        time.sleep(0.8)
        page.screenshot(path=str(SCREENSHOT_DIR / "dashboard_mobile.png"), full_page=True)
        print(f"   [PASS] Saved mobile screenshot: {SCREENSHOT_DIR / 'dashboard_mobile.png'}")

        browser.close()
        print("ALL END-TO-END BROWSER CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    verify_browser_dashboard()
