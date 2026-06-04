"""Main entry point for the Sportpacks.de scraper.

Orchestrates the full pipeline:
1. Discover all product URLs from collection pages
2. Fetch product data from Shopify .js endpoints
3. Compare scraped data against existing database records
4. Only generate embeddings for new products or those with changed images
5. Batch upsert new/updated products (50 per batch with retry)
6. Clean up stale products (unseen for 2 consecutive runs)
7. Print a summary of the run
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from src.database import SupabaseDB
from src.embeddings import generate_image_embedding, generate_text_embedding, load_model
from src.scraper import scrape_all_product_handles, scrape_product

logger = logging.getLogger(__name__)

# Fields that determine whether a product has changed
_COMPARISON_FIELDS = [
    "title",
    "description",
    "price",
    "sale",
    "image_url",
    "additional_images",
    "category",
    "tags",
    "metadata",
    "other",
    "size",
    "brand",
]


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _product_has_changed(scraped: dict[str, Any], existing: dict[str, Any]) -> bool:
    """Compare scraped product against existing database record.

    Returns True if any meaningful field has changed, False if identical.
    """
    for field in _COMPARISON_FIELDS:
        scraped_val = scraped.get(field)
        existing_val = existing.get(field)
        if scraped_val != existing_val:
            logger.debug(
                "Field '%s' changed for %s",
                field,
                scraped.get("product_url", "?"),
            )
            return True
    return False


async def process_products_batch(
    client: httpx.AsyncClient,
    handles: list[str],
    store_domain: str,
    source_name: str,
    db: SupabaseDB,
    existing_products: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], set[str]]:
    """Process a batch of product handles with smart upsert logic.

    For each product:
      - Scrape the Shopify .js endpoint
      - Compare against the existing DB record (if any)
      - If unchanged → skip entirely (no DB write, no embedding regeneration)
      - If changed → update DB, regenerate embeddings ONLY if image URL changed
      - If new → insert with fresh embeddings

    Batches are collected (up to 50) and upserted in a single request.

    Returns:
        Tuple of (stats dict, set of seen product URLs).
    """
    stats: dict[str, int] = {
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
    }
    # Pre-populate seen_urls with ALL discovered handles so that a transient
    # scrape failure doesn't cause a product to be incorrectly cleaned up
    seen_urls: set[str] = {
        f"{store_domain}/en/products/{handle}" for handle in handles
    }
    db_batch: list[dict[str, Any]] = []

    for handle in handles:
        try:
            # --- Step 1: Scrape ---
            product_data = await scrape_product(
                client, handle, store_domain=store_domain, source_name=source_name
            )
            if product_data is None:
                stats["errors"] += 1
                logger.warning("Failed to scrape product %s, skipping.", handle)
                continue

            product_url = product_data["product_url"]
            seen_urls.add(product_url)
            existing = existing_products.get(product_url)

            # --- Step 2: Decide if product is new, changed, or unchanged ---
            if existing is None:
                # *** NEW PRODUCT ***
                stats["new"] += 1
                product_data["unseen_count"] = 0

                # Generate image embedding
                if product_data.get("image_url"):
                    img_emb = await generate_image_embedding(
                        product_data["image_url"]
                    )
                    if img_emb:
                        product_data["image_embedding"] = img_emb
                    else:
                        logger.warning(
                            "Failed to generate image embedding for %s", handle
                        )
                    # Stagger: 0.5s delay between HF API calls
                    await asyncio.sleep(0.5)

                # Generate text embedding
                info_text = product_data.get("other", product_data["title"])
                text_emb = generate_text_embedding(info_text)
                if text_emb:
                    product_data["info_embedding"] = text_emb
                else:
                    logger.warning(
                        "Failed to generate text embedding for %s", handle
                    )

                db_batch.append(product_data)
                logger.debug("New product: %s — %s", handle, product_data["title"])

            elif _product_has_changed(product_data, existing):
                # *** CHANGED PRODUCT ***
                stats["updated"] += 1
                product_data["unseen_count"] = 0

                # Only regenerate embeddings if the image URL changed
                image_changed = product_data.get("image_url") != existing.get(
                    "image_url"
                )
                if image_changed:
                    logger.debug(
                        "Image URL changed for %s — regenerating embeddings", handle
                    )
                    if product_data.get("image_url"):
                        img_emb = await generate_image_embedding(
                            product_data["image_url"]
                        )
                        if img_emb:
                            product_data["image_embedding"] = img_emb
                        await asyncio.sleep(0.5)

                    info_text = product_data.get("other", product_data["title"])
                    text_emb = generate_text_embedding(info_text)
                    if text_emb:
                        product_data["info_embedding"] = text_emb
                else:
                    # Preserve existing embeddings — image hasn't changed
                    product_data["image_embedding"] = existing.get("image_embedding")
                    product_data["info_embedding"] = existing.get("info_embedding")

                db_batch.append(product_data)
                logger.debug("Updated product: %s — %s", handle, product_data["title"])

            else:
                # *** UNCHANGED PRODUCT — skip completely ***
                stats["unchanged"] += 1
                logger.debug("Unchanged product (skipped): %s", handle)

            # --- Step 3: Flush batch when we hit 50 ---
            if len(db_batch) >= 50:
                _, err = db.upsert_products(db_batch, batch_size=50)
                stats["errors"] += err
                db_batch = []

            # Small delay between products to be respectful
            await asyncio.sleep(0.2)

        except Exception as e:
            stats["errors"] += 1
            logger.error(
                "Error processing product %s: %s", handle, e, exc_info=True
            )

    # Flush remaining batch
    if db_batch:
        _, err = db.upsert_products(db_batch, batch_size=50)
        stats["errors"] += err

    return stats, seen_urls


async def run_scraper(
    supabase_url: str,
    supabase_key: str,
    collection_url: str,
    store_domain: str = "https://sportpacks.de",
    source_name: str = "scraper-sportpacks",
    max_workers: int = 5,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full smart scraper pipeline.

    Steps:
      1. Discover all product handles from collection pages
      2. Fetch all existing products from the database (for comparison)
      3. For each product: scrape, compare, conditionally embed, batch-upsert
      4. Clean up stale products (unseen for 2 consecutive runs)
      5. Print a run summary
    """
    setup_logging(verbose)
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Sportpacks.de Smart Scraper")
    logger.info("=" * 60)

    # Initialize database
    logger.info("Connecting to Supabase...")
    db = SupabaseDB(supabase_url, supabase_key)

    # Load the embedding model
    logger.info("Loading SigLIP embedding model...")
    load_model()

    # Fetch all existing products for comparison
    logger.info("Fetching existing products from database...")
    existing_products = db.get_products_by_source(source_name)
    logger.info("Found %d existing products in database.", len(existing_products))

    # Step 1: Discover all products
    logger.info("Discovering product URLs from collection pages...")
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    ) as client:
        all_handles = await scrape_all_product_handles(client, collection_url)

    if not all_handles:
        logger.error("No products found. Exiting.")
        return {
            "total": 0,
            "new": 0,
            "updated": 0,
            "unchanged": 0,
            "deleted": 0,
            "errors": 0,
        }

    logger.info("Discovered %d unique products.", len(all_handles))

    # Step 2: Process all products
    logger.info("Starting smart product processing pipeline...")

    total_stats: dict[str, int] = {
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
    }
    all_seen_urls: set[str] = set()

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
    ) as client:
        # Process in chunks to avoid memory issues
        chunk_size = max_workers
        for i in range(0, len(all_handles), chunk_size):
            chunk = all_handles[i : i + chunk_size]
            logger.info(
                "Processing chunk %d/%d (products %d-%d)...",
                i // chunk_size + 1,
                (len(all_handles) + chunk_size - 1) // chunk_size,
                i + 1,
                min(i + chunk_size, len(all_handles)),
            )

            chunk_stats, chunk_seen = await process_products_batch(
                client,
                chunk,
                store_domain=store_domain,
                source_name=source_name,
                db=db,
                existing_products=existing_products,
            )

            for key in ("new", "updated", "unchanged", "errors"):
                total_stats[key] += chunk_stats.get(key, 0)
            all_seen_urls.update(chunk_seen)

    # Step 3: Clean up stale products
    logger.info("Cleaning up stale products...")
    cleanup_result = db.cleanup_stale_products(source_name, all_seen_urls)

    # Step 4: Print run summary
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    logger.info("=" * 60)
    logger.info("Scraping complete!")
    logger.info("  Total products discovered:  %d", len(all_handles))
    logger.info("  New products added:         %d", total_stats["new"])
    logger.info("  Products updated:           %d", total_stats["updated"])
    logger.info("  Products unchanged (skip):  %d", total_stats["unchanged"])
    logger.info("  Stale products deleted:     %d", cleanup_result.get("deleted", 0))
    logger.info("  First-miss marked:          %d", cleanup_result.get("marked_stale", 0))
    logger.info("  Errors:                     %d", total_stats["errors"])
    logger.info("  Time elapsed:               %dm %ds", minutes, seconds)
    logger.info("=" * 60)

    return {
        "total": len(all_handles),
        "new": total_stats["new"],
        "updated": total_stats["updated"],
        "unchanged": total_stats["unchanged"],
        "deleted": cleanup_result.get("deleted", 0),
        "marked_stale": cleanup_result.get("marked_stale", 0),
        "errors": total_stats["errors"],
    }


def main() -> None:
    """CLI entry point."""
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Sportpacks.de Smart Scraper")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_KEY"))
    parser.add_argument(
        "--collection-url",
        default=os.getenv(
            "COLLECTION_URL",
            "https://sportpacks.de/en/collections/alle-artikel-1",
        ),
    )
    parser.add_argument(
        "--store-domain",
        default=os.getenv("STORE_DOMAIN", "https://sportpacks.de"),
    )
    parser.add_argument(
        "--source",
        default=os.getenv("SOURCE_NAME", "scraper-sportpacks"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("MAX_WORKERS", "5")),
        help="Number of products to process per chunk",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if not args.supabase_url or not args.supabase_key:
        parser.error(
            "Supabase URL and key are required. "
            "Set SUPABASE_URL and SUPABASE_KEY in .env or pass them."
        )

    asyncio.run(
        run_scraper(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            collection_url=args.collection_url,
            store_domain=args.store_domain,
            source_name=args.source,
            max_workers=args.workers,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
