"""
Scraper / form-submission module.

Uses Playwright (async API) to navigate to the Israeli police traffic-violation
appeal portal and submit the user's appeal automatically.

The exact selectors depend on the live site structure; they are centralised here
so that updates only require changes in one place.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from config.settings import APPEAL_URL, TEMP_DIR, TWOCAPTCHA_API_KEY
from ai.analyzer import ViolationData

logger = logging.getLogger(__name__)

# CSS selectors for the appeal form fields.
# Update these if the site changes its structure.
SELECTORS = {
    "case_number": 'input[name="caseNumber"], input[id*="case"], input[placeholder*="תיק"]',
    "owner_name":  'input[name="fullName"], input[id*="name"], input[placeholder*="שם"]',
    "owner_id":    'input[name="idNumber"], input[id*="id"], input[placeholder*="זהות"]',
    "vehicle_number": 'input[name="vehicleNumber"], input[id*="vehicle"], input[placeholder*="רכב"]',
    "appeal_text": 'textarea[name="appealText"], textarea[id*="appeal"], textarea[placeholder*="ערר"]',
    "file_input":  'input[type="file"]',
    "submit":      'button[type="submit"], input[type="submit"]',
}


async def _solve_captcha_if_needed(page) -> None:
    """
    Attempt to solve a reCAPTCHA using the 2captcha service.
    Skipped silently when TWOCAPTCHA_API_KEY is not set or no CAPTCHA found.
    """
    if not TWOCAPTCHA_API_KEY:
        return
    try:
        frame = page.frame_locator('iframe[src*="recaptcha"]')
        # Check whether a CAPTCHA iframe is present
        if await frame.locator(".recaptcha-checkbox").count() == 0:
            return
        from twocaptcha import TwoCaptcha  # type: ignore
        solver = TwoCaptcha(TWOCAPTCHA_API_KEY)
        site_key_el = page.locator('[data-sitekey]')
        site_key = await site_key_el.get_attribute("data-sitekey")
        result = solver.recaptcha(sitekey=site_key, url=page.url)
        token = result["code"]
        await page.evaluate(
            f'document.getElementById("g-recaptcha-response").value = "{token}";'
        )
        logger.info("CAPTCHA solved via 2captcha")
    except Exception as exc:
        logger.warning("CAPTCHA solving skipped or failed: %s", exc)


async def submit_appeal(
    violation: ViolationData,
    appeal_text: str,
    attachment_path: str | None = None,
) -> tuple[bool, str]:
    """
    Fill in and submit the appeal form.

    Returns (success: bool, screenshot_path: str).
    *screenshot_path* points to a PNG file with the confirmation screen
    (or the error state when success is False).
    """
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

    os.makedirs(TEMP_DIR, exist_ok=True)
    screenshot_path = str(Path(TEMP_DIR) / "confirmation.png")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="he-IL",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            logger.info("Navigating to appeal portal: %s", APPEAL_URL)
            await page.goto(APPEAL_URL, wait_until="networkidle", timeout=30_000)

            # ── Fill fields ───────────────────────────────────────────────
            async def fill(selector: str, value: str) -> None:
                if not value:
                    return
                try:
                    locator = page.locator(selector).first
                    await locator.wait_for(state="visible", timeout=5_000)
                    await locator.fill(value)
                except PlaywrightTimeout:
                    logger.warning("Field not found for selector: %s", selector)

            await fill(SELECTORS["case_number"],    violation.get("case_number", ""))
            await fill(SELECTORS["owner_name"],     violation.get("owner_name", ""))
            await fill(SELECTORS["owner_id"],       violation.get("owner_id", ""))
            await fill(SELECTORS["vehicle_number"], violation.get("vehicle_number", ""))
            await fill(SELECTORS["appeal_text"],    appeal_text)

            # ── Attach file (if provided) ─────────────────────────────────
            if attachment_path and Path(attachment_path).exists():
                try:
                    file_input = page.locator(SELECTORS["file_input"]).first
                    await file_input.set_input_files(attachment_path)
                    logger.info("Attached file: %s", attachment_path)
                except PlaywrightTimeout:
                    logger.warning("File input not found; skipping attachment")

            # ── CAPTCHA ───────────────────────────────────────────────────
            await _solve_captcha_if_needed(page)

            # ── Submit ────────────────────────────────────────────────────
            submit_btn = page.locator(SELECTORS["submit"]).first
            await submit_btn.wait_for(state="visible", timeout=5_000)
            await submit_btn.click()
            await page.wait_for_load_state("networkidle", timeout=15_000)

            await page.screenshot(path=screenshot_path, full_page=True)
            logger.info("Appeal submitted; screenshot saved to %s", screenshot_path)
            return True, screenshot_path

        except Exception as exc:
            logger.error("Appeal submission failed: %s", exc)
            try:
                await page.screenshot(path=screenshot_path, full_page=True)
            except Exception:
                pass
            return False, screenshot_path
        finally:
            await browser.close()
