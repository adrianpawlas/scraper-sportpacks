"""Supabase database client for upserting scraped product data."""

from __future__ import annotations

import logging
from typing import Any

from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseDB:
    """Manages Supabase database operations."""

    def __init__(self, url: str, key: str) -> None:
        self.client: Client = create_client(url, key)
        self.table_name = "products"

    def upsert_products(
        self,
        products: list[dict[str, Any]],
        batch_size: int = 50,
    ) -> tuple[int, int]:
        """Upsert products into the database in batches.

        Returns:
            Tuple of (success_count, error_count).
        """
        success = 0
        errors = 0

        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            try:
                response = (
                    self.client.table(self.table_name)
                    .upsert(batch, on_conflict="source,product_url")
                    .execute()
                )
                # Count successful inserts/updates
                if response.data:
                    success += len(response.data)
                else:
                    success += len(batch)
                logger.info(
                    "Upserted batch %d-%d/%d: %d products",
                    i,
                    min(i + batch_size, len(products)),
                    len(products),
                    len(batch),
                )
            except Exception as e:
                errors += len(batch)
                logger.error("Failed to upsert batch %d-%d: %s", i, i + len(batch), e)

        return success, errors

    def get_existing_product_urls(self) -> set[str]:
        """Get all existing product URLs from the database."""
        try:
            response = (
                self.client.table(self.table_name)
                .select("product_url")
                .eq("source", "scraper-sportpacks")
                .execute()
            )
            if response.data:
                return {row["product_url"] for row in response.data}
            return set()
        except Exception as e:
            logger.error("Failed to fetch existing products: %s", e)
            return set()
