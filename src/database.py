"""Supabase database client for upserting scraped product data."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseDB:
    """Manages Supabase database operations."""

    def __init__(self, url: str, key: str) -> None:
        self.client: Client = create_client(url, key)
        self.table_name = "products"

    def get_products_by_source(self, source: str) -> dict[str, dict[str, Any]]:
        """Fetch ALL existing products for a given source.

        Returns a dict mapping product_url -> full product record.
        Useful for comparing scraped data against existing records.
        """
        try:
            response = (
                self.client.table(self.table_name)
                .select("*")
                .eq("source", source)
                .execute()
            )
            if response.data:
                return {row["product_url"]: row for row in response.data}
            return {}
        except Exception as e:
            logger.error("Failed to fetch products for source '%s': %s", source, e)
            return {}

    def get_existing_product_urls(self, source: str = "scraper-sportpacks") -> set[str]:
        """Get all existing product URLs from the database for a given source."""
        try:
            response = (
                self.client.table(self.table_name)
                .select("product_url")
                .eq("source", source)
                .execute()
            )
            if response.data:
                return {row["product_url"] for row in response.data}
            return set()
        except Exception as e:
            logger.error("Failed to fetch existing products: %s", e)
            return set()

    def upsert_products(
        self,
        products: list[dict[str, Any]],
        batch_size: int = 50,
    ) -> tuple[int, int]:
        """Upsert products into the database in batches with retry logic.

        Each batch is retried up to 3 times before being logged to a local
        failure file and skipped.

        Args:
            products: List of product dicts to upsert.
            batch_size: Number of products per batch (default 50).

        Returns:
            Tuple of (success_count, error_count).
        """
        success = 0
        errors = 0

        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            last_error: Exception | None = None

            for attempt in range(3):
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
                        "Upserted batch %d-%d/%d: %d products (attempt %d)",
                        i + 1,
                        min(i + batch_size, len(products)),
                        len(products),
                        len(batch),
                        attempt + 1,
                    )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        delay = min(2 ** (attempt + 1), 10)
                        logger.warning(
                            "Retry %d/3 for batch %d-%d in %ds: %s",
                            attempt + 1,
                            i + 1,
                            min(i + batch_size, len(products)),
                            delay,
                            e,
                        )
                        time.sleep(delay)
                    else:
                        errors += len(batch)
                        logger.error(
                            "Failed to upsert batch %d-%d after 3 attempts: %s",
                            i + 1,
                            min(i + batch_size, len(products)),
                            e,
                        )
                        # Log failed batch to local file
                        self._log_failed_batch(batch, last_error)

        return success, errors

    def _log_failed_batch(self, batch: list[dict[str, Any]], error: Exception) -> None:
        """Log a failed batch to the local failure log file."""
        try:
            with open("failed_batches.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now(timezone.utc).isoformat()}] Failed batch: {error}\n")
                json.dump(batch, f, ensure_ascii=False, indent=2, default=str)
                f.write("\n---\n")
        except OSError as e:
            logger.error("Failed to write to failed_batches.log: %s", e)

    def cleanup_stale_products(
        self,
        source: str,
        seen_urls: set[str],
    ) -> dict[str, int]:
        """Handle stale products after a scrape run.

        For each product from this source that was NOT seen in the current run:
            - If it has been unseen for 2+ runs (unseen_count >= 1), delete it.
            - Otherwise, increment unseen_count to mark first missed run.

        For each product that WAS seen, reset unseen_count to 0.

        Args:
            source: The source name to clean up.
            seen_urls: Set of product_url values that were seen in this run.

        Returns:
            Dict with keys "deleted", "marked_stale", "errors".
        """
        result: dict[str, int] = {"deleted": 0, "marked_stale": 0, "errors": 0}

        # 1. Reset unseen_count for all seen products
        if seen_urls:
            try:
                (
                    self.client.table(self.table_name)
                    .update({"unseen_count": 0})
                    .eq("source", source)
                    .in_("product_url", list(seen_urls))
                    .execute()
                )
                logger.info(
                    "Reset unseen_count for %d seen products.",
                    len(seen_urls),
                )
            except Exception as e:
                logger.warning(
                    "Could not batch-reset unseen_count for seen products "
                    "(column may not exist yet): %s",
                    e,
                )
                # If column doesn't exist, log a one-time hint
                if "column" in str(e).lower() and "unseen" in str(e).lower():
                    logger.warning(
                        "Run: ALTER TABLE products ADD COLUMN unseen_count INTEGER DEFAULT 0;"
                    )
                result["errors"] += 1

        # 2. Fetch all products for this source and handle unseen ones
        all_products = self.get_products_by_source(source)
        for product_url, existing in all_products.items():
            if product_url in seen_urls:
                continue

            current_unseen = existing.get("unseen_count", 0)
            if current_unseen is None:
                current_unseen = 0

            if current_unseen >= 1:
                # Second consecutive miss — delete
                try:
                    (
                        self.client.table(self.table_name)
                        .delete()
                        .eq("source", source)
                        .eq("product_url", product_url)
                        .execute()
                    )
                    result["deleted"] += 1
                    logger.info(
                        "Deleted stale product (unseen for 2 runs): %s",
                        product_url,
                    )
                except Exception as e:
                    result["errors"] += 1
                    logger.error(
                        "Failed to delete stale product %s: %s",
                        product_url,
                        e,
                    )
            else:
                # First missed run — mark as unseen
                try:
                    (
                        self.client.table(self.table_name)
                        .update({"unseen_count": 1})
                        .eq("source", source)
                        .eq("product_url", product_url)
                        .execute()
                    )
                    result["marked_stale"] += 1
                    logger.info(
                        "Marked product as unseen (1st miss): %s",
                        product_url,
                    )
                except Exception as e:
                    result["errors"] += 1
                    logger.error(
                        "Failed to mark product %s as unseen: %s",
                        product_url,
                        e,
                    )

        return result

    def reset_country_to_null(self, source: str | None = None) -> int:
        """Set the country column to NULL for all products.

        Args:
            source: If provided, only update products for this source.
                    If None, update all products across all sources.

        Returns:
            Number of rows updated.
        """
        try:
            query = self.client.table(self.table_name).update({"country": None})
            if source:
                query = query.eq("source", source)
            response = query.execute()
            updated = len(response.data) if response.data else 0
            logger.info(
                "Reset country to NULL for %d products%s.",
                updated,
                f" (source: {source})" if source else "",
            )
            return updated
        except Exception as e:
            logger.error("Failed to reset country to NULL: %s", e)
            return 0
