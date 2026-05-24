"""
db/qdrant_client.py

Qdrant connection and collection setup.

Responsibilities:
- Sync client  → ingestion, collection setup, health checks
- Async client → retrieval inside FastAPI async endpoints (non-blocking)
- Create the fashion_catalog collection if it does not exist
- Define the vector config and payload indexes
"""

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)

from config import settings
from utils.logger import setup_logger

log = setup_logger(__name__)

_sync_client: QdrantClient = None 
_async_client: AsyncQdrantClient = None


def get_sync_client() -> QdrantClient:
    """
    Singleton sync client.
    Used for: ingestion, collection setup, health checks.
    """
    global _sync_client
    if _sync_client is None:
        log.info(f"Connecting Qdrant sync client | host={settings.qdrant_host} port={settings.qdrant_port}")
        _sync_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        log.info("Qdrant sync client connected")
    return _sync_client


def get_async_client() -> AsyncQdrantClient:
    """
    Singleton async client.
    Used for: retrieval inside FastAPI async endpoints (non-blocking).
    """
    global _async_client
    if _async_client is None:
        log.info(f"Connecting Qdrant async client | host={settings.qdrant_host} port={settings.qdrant_port}")
        _async_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        log.info("Qdrant async client connected")
    return _async_client


def init_collection():
    """
    Create the fashion_catalog collection if it does not already exist.
    Also creates payload indexes for fast metadata pre-filtering.
    Safe to call on every startup — skips if collection already exists.
    """
    client     = get_sync_client()
    collection = settings.qdrant_collection

    existing = [c.name for c in client.get_collections().collections]

    if collection in existing:
        log.info(f"Collection '{collection}' already exists — skipping creation")
        return

    log.info(f"Creating collection '{collection}' | vector_size={settings.vector_size}")

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=settings.vector_size,
            distance=Distance.COSINE,
        ),
    )

    # ── Payload indexes for pre-filtering ─────────────────────────────────
    # Allow Qdrant to filter by metadata BEFORE vector search —
    # faster and more relevant than post-filtering the full catalog.
    indexes = {
        "category": PayloadSchemaType.KEYWORD,   # "tops" | "bottoms" | "shoes"
        "source":   PayloadSchemaType.KEYWORD,   # "hm"  | "zara"
        "gender":   PayloadSchemaType.KEYWORD,   # "men" | "women" | "unisex"
        "color":    PayloadSchemaType.KEYWORD,   # "white" | "navy" etc
        "price":    PayloadSchemaType.FLOAT,     # for range filters e.g. price < 100
    }

    for field, schema_type in indexes.items():
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema_type,
        )
        log.debug(f"Payload index created | field={field} type={schema_type}")

    log.info(f"Collection '{collection}' created with {len(indexes)} payload indexes")


def health_check() -> bool:
    """Return True if Qdrant is reachable."""
    try:
        get_sync_client().get_collections()
        return True
    except Exception as e:
        log.error(f"Qdrant health check failed: {e}")
        return False