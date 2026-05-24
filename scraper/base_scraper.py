from __future__ import annotations

import asyncio
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import async_playwright, BrowserContext, Page

from utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Canonical product schema
# ---------------------------------------------------------------------------

@dataclass
class ProductItem:
    """Canonical product item schema for the fashion scraper pipeline."""

    item_id: str = ""
    source: str = ""
    name: str = ""
    price: float = 0.0
    currency: str = ""
    category: str = ""
    gender: str = ""
    color: str = ""
    material: str = ""
    description: str = ""
    url: str = ""
    image_url: str = ""


# ---------------------------------------------------------------------------
# BaseScraper
# ---------------------------------------------------------------------------

# Pool of ≥ 3 distinct desktop browser User-Agent strings (Req 3.1)
_USER_AGENTS: list[str] = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4.1 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
    ),
]

# Regex for price extraction — matches integers and decimals (Req 4.4)
_PRICE_RE = re.compile(r"\d+[.,]?\d*")

# Init script to mask navigator.webdriver (Req 3.2)
_WEBDRIVER_MASK_SCRIPT = (
    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
)


class BaseScraper(ABC):
    """
    Abstract base class providing shared Playwright infrastructure for all
    fashion scrapers.  Concrete subclasses must define SOURCE, CATEGORY_URLS,
    and implement _extract_items().
    """

    # --- Abstract class attributes (must be overridden by subclasses) -------

    SOURCE: str  # e.g. "hm"
    CATEGORY_URLS: dict[str, str]  # {"tops": url, "bottoms": url}

    def __init__(self) -> None:
        self._log = setup_logger(f"scraper.{self.__class__.__name__}")

    # --- Browser context creation -------------------------------------------

    async def _create_context(self, playwright) -> BrowserContext:
        """
        Launch a Chromium browser with anti-bot settings and return a new
        browser context.

        Anti-bot measures applied (Req 3.1, 3.2, 3.5, 3.7):
        - Random User-Agent from pool of ≥ 3 strings
        - navigator.webdriver masked via init script
        - --disable-blink-features=AutomationControlled launch arg
        - Viewport 1366×768, locale en-US
        """
        user_agent = random.choice(_USER_AGENTS)
        browser = await playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
        )
        await context.add_init_script(_WEBDRIVER_MASK_SCRIPT)
        return context

    # --- Page helpers --------------------------------------------------------

    async def _scroll_page(self, page: Page) -> None:
        """
        Scroll to the bottom of the page 5 times with 1.5-second pauses to
        trigger lazy-loaded product cards (Req 4.3).
        """
        for _ in range(5):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1.5)

    async def _navigate(self, page: Page, url: str) -> bool:
        """
        Navigate to *url* with a 30-second timeout.

        Retry logic (Req 3.6):
        - HTTP 429 or 503 → wait 30 s and retry, up to 3 attempts.
        - Returns True on success, False after all retries are exhausted.
        """
        for attempt in range(1, 4):
            try:
                response = await page.goto(url, timeout=30_000)
                if response is not None and response.status in (429, 503):
                    self._log.warning(
                        "Rate limited (HTTP %d) on %s — waiting 30s "
                        "(attempt %d/3)",
                        response.status,
                        url,
                        attempt,
                    )
                    await asyncio.sleep(30)
                    continue
                return True
            except Exception as exc:  # noqa: BLE001
                self._log.error(
                    "Failed to load %s (attempt %d/3): %s",
                    url,
                    attempt,
                    exc,
                )
                if attempt < 3:
                    await asyncio.sleep(30)

        self._log.error("Giving up on %s after 3 attempts", url)
        return False

    # --- Data helpers --------------------------------------------------------

    def _parse_price(self, text: str) -> float | None:
        """
        Strip currency symbols and thousands separators, return a float.

        - Returns 0.0 if no numeric value is found (Req 4.4).
        - Logs a warning and returns None if the result would be negative
          (caller should discard the item) (Req 4.4).
        """
        # Normalise European decimal comma to period
        normalised = text.replace(",", ".")
        match = _PRICE_RE.search(normalised)
        if not match:
            return 0.0
        value = float(match.group().replace(",", "."))
        if value < 0:
            self._log.warning(
                "Negative price parsed from %r — discarding item", text
            )
            return None
        return value

    def _resolve_url(self, url: str, base: str) -> str:
        """
        Resolve a URL to a fully qualified HTTPS URL (Req 2.7, 2.8, 4.5).

        - ``//``-prefixed  → ``https:`` + url
        - ``/``-relative   → base.rstrip("/") + url
        - Absolute URLs    → returned unchanged
        """
        if url.startswith("//"):
            return "https:" + url
        if url.startswith("/"):
            return base.rstrip("/") + url
        return url

    def _make_item_id(self, unique_part: str) -> str:
        """Return ``"{SOURCE}_{unique_part}"`` (Req 2.5)."""
        return f"{self.SOURCE}_{unique_part}"

    def _write_output(self, items: list[dict]) -> None:
        """
        Write ``{"source": SOURCE, "count": N, "items": [...]}`` to
        ``data/{SOURCE}_products.json``, creating the ``data/`` directory
        if it does not exist (Req 7.1, 7.2, 7.3).
        """
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / f"{self.SOURCE}_products.json"
        payload = {
            "source": self.SOURCE,
            "count": len(items),
            "items": items,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        self._log.info(
            "Wrote %d items to %s", len(items), output_path
        )

    # --- Abstract hook -------------------------------------------------------

    @abstractmethod
    async def _extract_items(self, page: Page, category: str) -> list[dict]:
        """
        Extract all product items from a loaded category page.

        Subclasses implement site-specific DOM parsing here.
        """

    # --- Public entry point --------------------------------------------------

    async def run(self) -> list[dict]:
        """
        Scrape all categories defined in CATEGORY_URLS, write the per-source
        output file, and return the full list of collected items.

        Behaviour (Req 8.1, 8.2, 8.3, 4.6):
        - Logs INFO at the start of each category (source, label, URL).
        - Navigates, scrolls, then extracts items.
        - Discards items with an empty ``name`` field.
        - Logs INFO for each successfully extracted item (name, price, color).
        - Logs INFO at the end of each category (source, label, count).
        - Applies a random inter-page delay of 2–5 s between categories.
        - Calls _write_output with all collected items.
        """
        all_items: list[dict] = []

        async with async_playwright() as playwright:
            context = await self._create_context(playwright)
            page = await context.new_page()

            categories = list(self.CATEGORY_URLS.items())
            for idx, (category, url) in enumerate(categories):
                # Req 8.1 — log start of category
                self._log.info(
                    "[%s] Starting category '%s' → %s",
                    self.SOURCE,
                    category,
                    url,
                )

                success = await self._navigate(page, url)
                if not success:
                    self._log.error(
                        "[%s] Skipping category '%s' — navigation failed",
                        self.SOURCE,
                        category,
                    )
                    # Still log end-of-category with 0 items
                    self._log.info(
                        "[%s] Finished category '%s' — extracted 0 items",
                        self.SOURCE,
                        category,
                    )
                    # Apply inter-page delay before next category
                    if idx < len(categories) - 1:
                        await asyncio.sleep(random.uniform(2, 5))
                    continue

                await self._scroll_page(page)

                raw_items = await self._extract_items(page, category)

                # Req 4.6 — discard items with empty name
                valid_items: list[dict] = []
                for item in raw_items:
                    if not item.get("name", ""):
                        continue
                    valid_items.append(item)
                    # Req 8.2 — log each successfully extracted item
                    self._log.info(
                        "[%s] Extracted item: name=%r price=%s color=%r",
                        self.SOURCE,
                        item.get("name", ""),
                        item.get("price", ""),
                        item.get("color", ""),
                    )

                all_items.extend(valid_items)

                # Req 8.3 — log end of category
                self._log.info(
                    "[%s] Finished category '%s' — extracted %d items",
                    self.SOURCE,
                    category,
                    len(valid_items),
                )

                # Inter-page delay between categories (Req 3.3)
                if idx < len(categories) - 1:
                    await asyncio.sleep(random.uniform(2, 5))

            await context.close()

        self._write_output(all_items)
        return all_items
