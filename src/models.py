"""Data models for the Sportpacks scraper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScraperConfig:
    """Configuration for the scraper."""

    supabase_url: str
    supabase_key: str
    collection_url: str
    source_name: str = "scraper-sportpacks"
    store_domain: str = "https://sportpacks.de"
    batch_size: int = 20
    max_workers: int = 5
    embedding_model: str = "google/siglip-base-patch16-384"


@dataclass
class ScrapedProduct:
    """Represents a fully scraped and processed product ready for DB insertion."""

    id: str
    source: str
    product_url: str
    image_url: str
    brand: str
    title: str
    description: str
    category: str | None
    gender: str | None
    price: str | None
    sale: str | None
    additional_images: str | None
    tags: list[str] | None
    metadata: str
    second_hand: bool = False
    image_embedding: list[float] | None = None
    info_embedding: list[float] | None = None
    other: str | None = None
    size: str | None = None
    country: str | None = None
    affiliate_url: str | None = None
    compressed_image_url: str | None = None


@dataclass
class ShopifyProduct:
    """Raw Shopify product JSON data."""

    id: int
    title: str
    handle: str
    description: str | None
    vendor: str
    type: str
    tags: list[str]
    price: int
    compare_at_price: int | None
    images: list[str]
    variants: list[dict[str, Any]]
    options: list[dict[str, Any]]
    available: bool
    created_at: str | None = None
