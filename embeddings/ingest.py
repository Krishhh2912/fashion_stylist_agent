"""
embeddings/ingest.py

Reads the scraped catalog JSON, generates embeddings for each item
using sentence-transformers, and upserts them into Qdrant.

Usage:
    python -m embeddings.ingest
    python -m embeddings.ingest --file data/processed/catalog_latest.json
    python -m embeddings.ingest --force        # re-embed everything
"""

import argparse
import json
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct

from config import settings
from utils.logger import setup_logger
from db.db_client import get_sync_client, init_collection

log = setup_logger(__name__)

# How many items to upload to Qdrant in one API call
BATCH_SIZE = 64


# ─── Embedding model ─────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    """Load model once, reuse across all batches."""
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        log.info("Embedding model loaded")
    return _model


# ─── Text builder ────────────────────────────────────────────────────────────

def build_embedding_text(item: dict) -> str:
    """
    Build the text string that gets embedded for each catalog item.
    Combine all semantically useful fields into one string.
    The richer the text, the better the semantic search results.

    Example output:
        "Slim Fit Linen T-Shirt | tops | white | 100% linen |
         perfect for warm weather occasions | $24.99"
    """
    parts = [
        item.get("name", ""),
        item.get("category", ""),
        item.get("color", ""),
        item.get("material", ""),
        item.get("description", ""),
        f"${item.get('price', 0)}",
    ]
    # Filter empty strings and join
    return " | ".join(p.strip() for p in parts if p.strip())


# ─── Payload builder ─────────────────────────────────────────────────────────

def build_payload(item: dict) -> dict:
    """
    Build the metadata payload stored alongside the vector in Qdrant.
    These fields are used for pre-filtering before vector search.
    All fields that exist in payload indexes must be present here.
    """
    return {
        # ── Filterable fields (indexed) ───────────────────────────────
        "category":  item.get("category", ""),       # tops | bottoms | shoes
        "source":    item.get("source", ""),          # hm | zara
        "gender":    item.get("gender", "unisex"),    # men | women | unisex
        "color":     item.get("color", "").lower(),   # normalise to lowercase
        "price":     float(item.get("price", 0)),

        # ── Display fields (returned in search results) ────────────────
        "item_id":   item.get("item_id", ""),
        "name":      item.get("name", ""),
        "currency":  item.get("currency", "USD"),
        "material":  item.get("material", ""),
        "description": item.get("description", ""),
        "url":       item.get("url", ""),
        "image_url": item.get("image_url", ""),
    }


# ─── Main ingestion pipeline ─────────────────────────────────────────────────

def run_ingestion(catalog_path: str, force: bool = False):
    """
    Full ingestion pipeline:
    1. Load catalog JSON
    2. Validate items
    3. Skip already-ingested items (unless --force)
    4. Embed in batches
    5. Upsert to Qdrant
    """
    path = Path(catalog_path)
    if not path.exists():
        log.error(f"Catalog file not found: {path}")
        return

    # ── Load catalog ──────────────────────────────────────────────────────
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Support both { "items": [...] } and plain list formats
    items = data.get("items", data) if isinstance(data, dict) else data
    log.info(f"Loaded {len(items)} items from {path}")

    # ── Validate ──────────────────────────────────────────────────────────
    valid_items = [
        i for i in items
        if i.get("name") and i.get("category") and float(i.get("price", 0)) > 0
    ]
    skipped = len(items) - len(valid_items)
    if skipped:
        log.warning(f"Skipped {skipped} items with missing name/category/price")
    log.info(f"Valid items: {len(valid_items)}")

    if not valid_items:
        log.error("No valid items to ingest")
        return

    # ── Check existing points ─────────────────────────────────────────────
    client     = get_sync_client()
    collection = settings.qdrant_collection

    # Ensure collection exists before upserting
    init_collection()

    if not force:
        # Get all existing item_ids already in Qdrant
        existing_ids = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=collection,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=["item_id"],
                with_vectors=False,
            )
            for point in results:
                existing_ids.add(point.payload.get("item_id", ""))
            if offset is None:
                break

        before = len(valid_items)
        valid_items = [
            i for i in valid_items
            if i.get("item_id", "") not in existing_ids
        ]
        log.info(
            f"Skipping {before - len(valid_items)} already-ingested items "
            f"| {len(valid_items)} new items to ingest"
        )

    if not valid_items:
        log.info("Nothing new to ingest — catalog is up to date")
        return

    # ── Embed + upsert in batches ─────────────────────────────────────────
    model        = get_embedding_model()
    total        = len(valid_items)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    ingested     = 0

    log.info(f"Starting ingestion | items={total} batch_size={BATCH_SIZE} batches={total_batches}")

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)
        batch = valid_items[start:end]

        # Build embedding texts for this batch
        texts = [build_embedding_text(item) for item in batch]

        # Embed all texts in one call (faster than one by one)
        log.debug(f"Embedding batch {batch_num + 1}/{total_batches} | size={len(batch)}")
        vectors = model.encode(texts, show_progress_bar=False).tolist()

        # Build Qdrant points
        points = [
            PointStruct(
                id=str(uuid.uuid4()),     # unique UUID per point
                vector=vector,
                payload=build_payload(item),
            )
            for item, vector in zip(batch, vectors)
        ]

        # Upsert to Qdrant
        client.upsert(
            collection_name=collection,
            points=points,
        )

        ingested += len(batch)
        log.info(f"Batch {batch_num + 1}/{total_batches} upserted | progress={ingested}/{total}")

    # ── Summary ───────────────────────────────────────────────────────────
    collection_info = client.get_collection(collection)
    total_in_db     = collection_info.points_count

    log.info("=" * 50)
    log.info("Ingestion complete")
    log.info(f"  Items ingested  : {ingested}")
    log.info(f"  Total in Qdrant : {total_in_db}")
    log.info(f"  Collection      : {collection}")
    log.info("=" * 50)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest fashion catalog into Qdrant")
    parser.add_argument(
        "--file",
        default="data/processed/catalog_latest.json",
        help="Path to catalog JSON file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest all items even if already in Qdrant",
    )
    args = parser.parse_args()

    run_ingestion(args.file, force=args.force)