"""
H&M product scraper.

Implements HmScraper(BaseScraper) for scraping men's tops and bottoms
from the H&M US website.

Requirements: 1.1, 1.4, 2.1–2.8, 3.3, 3.4, 4.1, 4.4, 4.5, 4.6, 7.1
"""

from __future__ import annotations

import asyncio
import random
import re
from urllib.parse import urlparse

from playwright.async_api import Page

from scraper.base_scraper import BaseScraper
from utils.logger import setup_logger

# Base URL used for resolving relative/protocol-relative URLs
_HM_BASE = "https://www2.hm.com"

# Regex to extract a product article code from an H&M URL path.
# H&M product URLs typically contain a numeric article code, e.g.:
#   /en_us/productpage.1234567.html
_ARTICLE_CODE_RE = re.compile(r"\.(\d{7,})\.")


class HmScraper(BaseScraper):
    """
    Concrete scraper for H&M men's products.

    Scrapes tops and bottoms category pages, parses product cards, and
    returns a list of canonical product item dicts.
    """

    SOURCE = "hm"

    CATEGORY_URLS: dict[str, str] = {
        "tops":    "https://www2.hm.com/en_us/men/products/tops.html",
        "bottoms": "https://www2.hm.com/en_us/men/products/pants.html",
    }

    def __init__(self) -> None:
        super().__init__()
        # Use the shared logger from BaseScraper (self._log is already set),
        # but we can also reference it directly for clarity.

    # -------------------------------------------------------------------------
    # Abstract method implementation
    # -------------------------------------------------------------------------

    async def _extract_items(self, page: Page, category: str) -> list[dict]:
        """
        Parse H&M product cards from the loaded category page.

        Selects all ``li.product-item`` cards and extracts:
        - name      : ``.item-heading a`` text
        - price     : ``.item-price`` text → float via ``_parse_price``
        - color     : ``.item-details`` text or ``data-color`` attribute
        - url       : ``a[href]`` on card → resolved to absolute HTTPS
        - image_url : ``img[src]`` or ``img[data-src]`` → resolved to HTTPS
        - item_id   : article code from URL path or ``data-articlecode``

        Cards with missing critical selectors are skipped with a DEBUG log.
        Items with empty ``name`` or negative price are discarded.

        An inter-product delay of ``random.uniform(0.5, 2)`` seconds is
        applied between card processing steps (Req 3.4).
        """
        items: list[dict] = []

        # Locate all product cards on the page
        cards = await page.query_selector_all("li.product-item")
        self._log.debug(
            "[hm] Found %d product cards for category '%s'",
            len(cards),
            category,
        )

        for card in cards:
            try:
                # --- Name ---------------------------------------------------
                name_el = await card.query_selector(".item-heading a")
                if name_el is None:
                    self._log.debug(
                        "[hm] Skipping card — missing '.item-heading a'"
                    )
                    continue
                name = (await name_el.inner_text()).strip()

                # --- Price --------------------------------------------------
                price_el = await card.query_selector(".item-price")
                price_text = ""
                if price_el is not None:
                    price_text = (await price_el.inner_text()).strip()
                price = self._parse_price(price_text)
                if price is None:
                    # Negative price — discard item (Req 4.4)
                    self._log.debug(
                        "[hm] Discarding item %r — negative price", name
                    )
                    continue

                # --- Color --------------------------------------------------
                color = ""
                color_el = await card.query_selector(".item-details")
                if color_el is not None:
                    color = (await color_el.inner_text()).strip()
                if not color:
                    # Fallback: data-color attribute on the card itself
                    color = (await card.get_attribute("data-color")) or ""

                # --- URL ----------------------------------------------------
                url = ""
                link_el = await card.query_selector("a[href]")
                if link_el is not None:
                    href = (await link_el.get_attribute("href")) or ""
                    url = self._resolve_url(href, _HM_BASE)
                else:
                    self._log.debug(
                        "[hm] Skipping card — missing 'a[href]'"
                    )
                    continue

                # --- Image URL ----------------------------------------------
                image_url = ""
                img_el = await card.query_selector("img")
                if img_el is not None:
                    # Prefer src; fall back to data-src (lazy-loaded images)
                    src = (await img_el.get_attribute("src")) or ""
                    if not src or src.startswith("data:"):
                        src = (await img_el.get_attribute("data-src")) or ""
                    if src:
                        image_url = self._resolve_url(src, _HM_BASE)

                # --- Item ID ------------------------------------------------
                # Try to extract the article code from the URL path first.
                item_id_part = ""
                if url:
                    path = urlparse(url).path
                    match = _ARTICLE_CODE_RE.search(path)
                    if match:
                        item_id_part = match.group(1)

                # Fallback: data-articlecode attribute on the card
                if not item_id_part:
                    item_id_part = (
                        await card.get_attribute("data-articlecode")
                    ) or ""

                # Last resort: use the URL path itself (slugified)
                if not item_id_part and url:
                    item_id_part = urlparse(url).path.strip("/").replace("/", "_")

                item_id = self._make_item_id(item_id_part) if item_id_part else ""

                # --- Build canonical item dict ------------------------------
                item: dict = {
                    "item_id":     item_id,
                    "source":      self.SOURCE,
                    "name":        name,
                    "price":       price,
                    "currency":    "USD",
                    "category":    category,
                    "gender":      "men",
                    "color":       color,
                    "material":    "",
                    "description": "",
                    "url":         url,
                    "image_url":   image_url,
                }

                items.append(item)

            except Exception as exc:  # noqa: BLE001
                # Any unexpected error on a single card must not abort the run
                self._log.debug(
                    "[hm] Skipping card due to unexpected error: %s", exc
                )
                continue

            # Inter-product delay (Req 3.4)
            await asyncio.sleep(random.uniform(0.5, 2))

        return items
