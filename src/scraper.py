"""Scraper for Sportpacks.de Shopify store.

Fetches collection pages to discover all product URLs,
then fetches each product's .js endpoint for clean JSON data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.models import ShopifyProduct

logger = logging.getLogger(__name__)


BRAND_NAME = "Sport Packs"


def _convert_image_url(url: str) -> str:
    """Convert protocol-relative URL to full HTTPS URL."""
    if url.startswith("//"):
        return f"https:{url}"
    if not url.startswith("http"):
        return f"https://{url}"
    return url


def _clean_html(html: str | None) -> str:
    """Strip HTML tags from a string."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)


def _parse_category(category: str | None) -> str | None:
    """Parse and clean the category string.
    
    Splits combined categories like "Sweaters & Hoodies" → "Sweaters, Hoodies"
    """
    if not category:
        return None
    # Split on common separators
    parts = re.split(r"\s*[&/]\s*", category)
    cleaned = ", ".join(p.strip() for p in parts if p.strip())
    return cleaned if cleaned else None


def _format_price_cents(price_cents: int, currency: str = "EUR") -> str:
    """Format a price in cents to a display string."""
    euros = price_cents / 100.0
    return f"{euros:.2f}{currency}"


def _extract_product_handles(page_text: str) -> list[str]:
    """Extract product handles from collection page HTML."""
    soup = BeautifulSoup(page_text, "lxml")
    handles: set[str] = set()

    # Look for product links in the grid
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Match /products/product-handle patterns
        match = re.search(r"/products/([^/?]+)", href)
        if match:
            handles.add(match.group(1))

    logger.debug("Found %d product handles on collection page", len(handles))
    return sorted(handles)


def _is_empty_collection_page(page_text: str) -> bool:
    """Check if a collection page has no products."""
    return "no products found" in page_text.lower() or "keine produkte gefunden" in page_text.lower()


async def fetch_product_json(
    client: httpx.AsyncClient,
    handle: str,
    base_url: str = "https://sportpacks.de",
    retries: int = 3,
) -> dict[str, Any] | None:
    """Fetch product data from Shopify's .js endpoint."""
    url = f"{base_url}/en/products/{handle}.js"
    for attempt in range(retries):
        try:
            resp = await client.get(url, timeout=15.0)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(
                "Failed to fetch %s (attempt %d/%d): HTTP %d",
                url,
                attempt + 1,
                retries,
                resp.status_code,
            )
        except (httpx.TimeoutException, httpx.RequestError) as e:
            logger.warning(
                "Request error for %s (attempt %d/%d): %s",
                url,
                attempt + 1,
                retries,
                e,
            )
        if attempt < retries - 1:
            await _exponential_backoff(attempt)
    return None


async def _exponential_backoff(attempt: int) -> None:
    """Sleep with exponential backoff."""
    delay = min(2 ** (attempt + 1), 30)
    await asyncio.sleep(delay)


async def fetch_collection_page(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 3,
) -> str | None:
    """Fetch a collection page HTML."""
    for attempt in range(retries):
        try:
            resp = await client.get(url, timeout=30.0)
            if resp.status_code == 200:
                return resp.text
            logger.warning(
                "Failed to fetch collection %s (attempt %d/%d): HTTP %d",
                url,
                attempt + 1,
                retries,
                resp.status_code,
            )
        except (httpx.TimeoutException, httpx.RequestError) as e:
            logger.warning(
                "Request error for %s (attempt %d/%d): %s",
                url,
                attempt + 1,
                retries,
                e,
            )
        if attempt < retries - 1:
            await _exponential_backoff(attempt)
    return None


async def scrape_all_product_handles(
    client: httpx.AsyncClient,
    base_collection_url: str,
) -> list[str]:
    """Scrape all product handles from all collection pages."""
    all_handles: list[str] = []
    page = 1

    while True:
        url = f"{base_collection_url}?page={page}"
        logger.info("Scraping collection page %d...", page)

        html = await fetch_collection_page(client, url)
        if html is None:
            logger.error("Failed to fetch collection page %d, stopping.", page)
            break

        if _is_empty_collection_page(html):
            logger.info("Page %d is empty (no products). Done scraping collection.", page)
            break

        handles = _extract_product_handles(html)
        if not handles:
            logger.info("No handles found on page %d. Done scraping collection.", page)
            break

        logger.info("Found %d products on page %d.", len(handles), page)
        all_handles.extend(handles)

        # Rate limit between pages
        await asyncio.sleep(0.5)
        page += 1

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for h in all_handles:
        if h not in seen:
            seen.add(h)
            deduped.append(h)

    logger.info("Total unique product handles found: %d", len(deduped))
    return deduped


async def scrape_product(
    client: httpx.AsyncClient,
    handle: str,
    store_domain: str = "https://sportpacks.de",
    source_name: str = "scraper-sportpacks",
) -> dict[str, Any] | None:
    """Scrape a single product and return a dict ready for DB insertion."""
    raw = await fetch_product_json(client, handle, base_url=store_domain)
    if raw is None:
        return None

    try:
        # Only pass fields that the dataclass expects
        field_names = set(ShopifyProduct.__dataclass_fields__)
        filtered = {k: v for k, v in raw.items() if k in field_names}
        product = ShopifyProduct(**filtered)
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        logger.error("Failed to parse product %s: %s", handle, e)
        return None

    # Process images
    images = [_convert_image_url(img) for img in product.images if img]
    main_image = images[0] if images else ""
    # Format: "url1 , url2" (space before and after comma per user request)
    additional_images = " , ".join(images[1:]) if len(images) > 1 else None

    # Process pricing
    price_str: str | None = None
    sale_str: str | None = None

    if product.compare_at_price and product.compare_at_price > 0:
        # On sale: compare_at_price is the original, price is the sale price
        price_str = _format_price_cents(product.compare_at_price)
        sale_str = _format_price_cents(product.price)
    else:
        price_str = _format_price_cents(product.price) if product.price else None

    # Clean description
    description = _clean_html(product.description) if product.description else ""

    # Process category
    category = _parse_category(product.type)

    # Build metadata
    metadata = {
        "shopify_id": product.id,
        "handle": product.handle,
        "vendor": product.vendor,
        "type": product.type,
        "tags": product.tags,
        "options": product.options,
        "variants": product.variants,
        "available": product.available,
        "created_at": product.created_at,
    }
    metadata_str = json.dumps(metadata, ensure_ascii=False)

    # Build info text for embedding
    info_parts = [
        f"Title: {product.title}",
        f"Brand: {product.vendor}",
        f"Category: {category}" if category else None,
        f"Description: {description}" if description else None,
        f"Price: {price_str}" if price_str else None,
        f"Sale: {sale_str}" if sale_str else None,
    ]
    info_text = ". ".join(p for p in info_parts if p)

    # Extract size from title if present (e.g., "Vintage Tee | S" → "S")
    size = None
    if "|" in product.title:
        parts = product.title.split("|")
        if len(parts) >= 2:
            size = parts[-1].strip()

    scraped = {
        "id": str(product.id),
        "source": source_name,
        "product_url": f"{store_domain}/en/products/{product.handle}",
        "image_url": main_image,
        "brand": BRAND_NAME,
        "title": product.title,
        "description": description,
        "category": category,
        "gender": None,  # No gender info available from the data
        "price": price_str,
        "sale": sale_str,
        "additional_images": additional_images,
        "tags": product.tags if product.tags else None,
        "metadata": metadata_str,
        "second_hand": False,
        "other": info_text,  # Combined info text used for text embedding
        "size": size,
        "country": "DE",
        "image_embedding": None,
        "info_embedding": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return scraped
