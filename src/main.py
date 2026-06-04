"""Main entry point for the Sportpacks.de scraper.

Orchestrates the full pipeline:
1. Discover all product URLs from collection pages
2. Fetch product data from Shopify .js endpoints
3. Generate image and text embeddings using SigLIP
4. Upsert everything to Supabase
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


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def process_products_batch(
    client: httpx.AsyncClient,
    handles: list[str],
    store_domain: str,
    source_name: str,
    db: SupabaseDB,
    batch_size: int,
) -> dict[str, Any]:
    """Process a batch of product handles: scrape, embed, and upload."""
    stats = {"scraped": 0, "embedded": 0, "uploaded": 0, "errors": 0}
    db_batch: list[dict[str, Any]] = []

    for i, handle in enumerate(handles):
        try:
            # Step 1: Scrape product data
            product_data = await scrape_product(
                client, handle, store_domain=store_domain, source_name=source_name
            )
            if product_data is None:
                stats["errors"] += 1
                logger.warning("Failed to scrape product %s, skipping.", handle)
                continue

            stats["scraped"] += 1

            # Step 2: Generate image embedding
            if product_data["image_url"]:
                img_emb = await generate_image_embedding(product_data["image_url"])
                if img_emb:
                    product_data["image_embedding"] = img_emb
                else:
                    logger.warning(
                        "Failed to generate image embedding for %s", handle
                    )

            # Step 3: Generate text embedding
            info_text = product_data.get("other", product_data["title"])
            text_emb = generate_text_embedding(info_text)
            if text_emb:
                product_data["info_embedding"] = text_emb
            else:
                logger.warning(
                    "Failed to generate text embedding for %s", handle
                )

            if product_data.get("image_embedding") or product_data.get("info_embedding"):
                stats["embedded"] += 1

            # Step 4: Add to DB batch
            db_batch.append(product_data)

            # Upload in batches
            if len(db_batch) >= batch_size:
                suc, err = db.upsert_products(db_batch, batch_size=batch_size)
                stats["uploaded"] += suc
                stats["errors"] += err
                db_batch = []

            # Small delay between products to be respectful
            await asyncio.sleep(0.2)

        except Exception as e:
            stats["errors"] += 1
            logger.error("Error processing product %s: %s", handle, e, exc_info=True)

    # Upload remaining batch
    if db_batch:
        suc, err = db.upsert_products(db_batch, batch_size=batch_size)
        stats["uploaded"] += suc
        stats["errors"] += err

    return stats


async def run_scraper(
    supabase_url: str,
    supabase_key: str,
    collection_url: str,
    store_domain: str = "https://sportpacks.de",
    source_name: str = "scraper-sportpacks",
    batch_size: int = 20,
    max_workers: int = 5,
    verbose: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Run the full scraper pipeline."""
    setup_logging(verbose)
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Sportpacks.de Scraper")
    logger.info("=" * 60)

    # Initialize database
    logger.info("Connecting to Supabase...")
    db = SupabaseDB(supabase_url, supabase_key)

    # Load the embedding model
    logger.info("Loading SigLIP embedding model...")
    load_model()

    # Get existing products if skipping
    existing_urls: set[str] = set()
    if skip_existing:
        logger.info("Fetching existing product URLs from database...")
        existing_urls = db.get_existing_product_urls()
        logger.info("Found %d existing products in database.", len(existing_urls))

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
        return {"total": 0, "scraped": 0, "embedded": 0, "uploaded": 0, "errors": 0, "skipped": 0}

    logger.info("Discovered %d unique products.", len(all_handles))

    # Filter out existing products if needed
    if skip_existing and existing_urls:
        handles_to_scrape = []
        skipped = 0
        for handle in all_handles:
            url = f"{store_domain}/en/products/{handle}"
            if url in existing_urls:
                skipped += 1
            else:
                handles_to_scrape.append(handle)
        logger.info("Skipping %d already-imported products.", skipped)
        logger.info("Products to scrape: %d", len(handles_to_scrape))
    else:
        handles_to_scrape = all_handles
        skipped = 0

    if not handles_to_scrape:
        logger.info("All products already in database. Nothing to do.")
        return {
            "total": len(all_handles),
            "scraped": 0,
            "embedded": 0,
            "uploaded": 0,
            "errors": 0,
            "skipped": skipped,
        }

    # Step 2: Process products
    logger.info("Starting product processing pipeline...")

    total_stats = {"scraped": 0, "embedded": 0, "uploaded": 0, "errors": 0}

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
    ) as client:
        # Process in manageable chunks to avoid memory issues
        chunk_size = max_workers  # Process N products per chunk
        for i in range(0, len(handles_to_scrape), chunk_size):
            chunk = handles_to_scrape[i : i + chunk_size]
            logger.info(
                "Processing chunk %d/%d (products %d-%d)...",
                i // chunk_size + 1,
                (len(handles_to_scrape) + chunk_size - 1) // chunk_size,
                i + 1,
                min(i + chunk_size, len(handles_to_scrape)),
            )

            chunk_stats = await process_products_batch(
                client,
                chunk,
                store_domain=store_domain,
                source_name=source_name,
                db=db,
                batch_size=batch_size,
            )

            for key in total_stats:
                total_stats[key] += chunk_stats.get(key, 0)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    logger.info("=" * 60)
    logger.info("Scraping complete!")
    logger.info("  Total products discovered: %d", len(all_handles))
    logger.info("  Skipped (already in DB):   %d", skipped)
    logger.info("  New products scraped:      %d", total_stats["scraped"])
    logger.info("  Embeddings generated:      %d", total_stats["embedded"])
    logger.info("  Uploaded to Supabase:      %d", total_stats["uploaded"])
    logger.info("  Errors:                    %d", total_stats["errors"])
    logger.info("  Time elapsed:              %dm %ds", minutes, seconds)
    logger.info("=" * 60)

    return {
        "total": len(all_handles),
        "skipped": skipped,
        **total_stats,
    }


def main() -> None:
    """CLI entry point."""
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Sportpacks.de Scraper")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_KEY"))
    parser.add_argument(
        "--collection-url",
        default=os.getenv("COLLECTION_URL", "https://sportpacks.de/en/collections/alle-artikel-1"),
    )
    parser.add_argument("--store-domain", default=os.getenv("STORE_DOMAIN", "https://sportpacks.de"))
    parser.add_argument("--source", default=os.getenv("SOURCE_NAME", "scraper-sportpacks"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "20")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("MAX_WORKERS", "5")))
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-scrape and re-upload all products even if already in DB",
    )

    args = parser.parse_args()

    if not args.supabase_url or not args.supabase_key:
        parser.error("Supabase URL and key are required. Set SUPABASE_URL and SUPABASE_KEY in .env or pass them.")

    asyncio.run(
        run_scraper(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            collection_url=args.collection_url,
            store_domain=args.store_domain,
            source_name=args.source,
            batch_size=args.batch_size,
            max_workers=args.workers,
            verbose=args.verbose,
            skip_existing=not args.no_skip_existing,
        )
    )


if __name__ == "__main__":
    main()
