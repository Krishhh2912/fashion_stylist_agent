"""
embeddings/retriever.py

Semantic search over the Qdrant fashion catalog.

Two-phase retrieval:
    1. Metadata pre-filter  — narrow by category / gender / price / color
                              (uses Qdrant payload indexes — fast)
    2. Vector search        — cosine similarity on the filtered subset
                              (returns top-K most relevant items)

Uses the async Qdrant client so retrieval is non-blocking inside FastAPI.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, MatchAny

from config import settings
from utils.logger import setup_logger
from db.db_client import get_async_client

log = setup_logger(__name__)


# ─── Embedding model (shared with ingest.py) ─────────────────────────────────

_model: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        log.info("Embedding model loaded")
    return _model


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Returns a list of floats."""
    model = get_embedding_model()
    vector = model.encode(text, show_progress_bar=False).tolist()
    log.debug(f"Query embedded | text='{text[:60]}...' dim={len(vector)}")
    return vector


# ─── Retrieved item dataclass ─────────────────────────────────────────────────

@dataclass
class RetrievedItem:
    """A single search result returned from Qdrant."""
    item_id:     str
    name:        str
    category:    str
    source:      str
    price:       float
    currency:    str
    color:       str
    material:    str
    description: str
    url:         str
    image_url:   str
    gender:      str
    score:       float      # cosine similarity score (0-1, higher = more relevant)

    def to_dict(self) -> dict:
        return {
            "item_id":     self.item_id,
            "name":        self.name,
            "category":    self.category,
            "source":      self.source,
            "price":       self.price,
            "currency":    self.currency,
            "color":       self.color,
            "material":    self.material,
            "description": self.description,
            "url":         self.url,
            "image_url":   self.image_url,
            "gender":      self.gender,
            "score":       round(self.score, 4),
        }


# ─── Filter builder ───────────────────────────────────────────────────────────

def build_filter(
    category:  Optional[str]        = None,
    gender:    Optional[str]        = None,
    max_price: Optional[float]      = None,
    min_price: Optional[float]      = None,
    colors:    Optional[list[str]]  = None,
    sources:   Optional[list[str]]  = None,
) -> Optional[Filter]:
    """
    Build a Qdrant Filter from optional metadata constraints.
    Only adds conditions for fields that are actually provided.
    Returns None if no filters specified (search full catalog).
    """
    conditions = []

    if category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )

    if gender and gender != "unisex":
        conditions.append(
            FieldCondition(key="gender", match=MatchAny(any=[gender, "unisex"]))
        )

    if max_price is not None or min_price is not None:
        conditions.append(
            FieldCondition(
                key="price",
                range=Range(
                    gte=min_price if min_price is not None else None,
                    lte=max_price if max_price is not None else None,
                ),
            )
        )

    if colors:
        # Match any of the provided colors (case-insensitive stored at ingest)
        conditions.append(
            FieldCondition(
                key="color",
                match=MatchAny(any=[c.lower() for c in colors]),
            )
        )

    if sources:
        conditions.append(
            FieldCondition(key="source", match=MatchAny(any=sources))
        )

    if not conditions:
        return None

    return Filter(must=conditions)


# ─── Core search ─────────────────────────────────────────────────────────────

async def search(
    query:     str,
    category:  Optional[str]       = None,
    gender:    Optional[str]       = None,
    max_price: Optional[float]     = None,
    min_price: Optional[float]     = None,
    colors:    Optional[list[str]] = None,
    sources:   Optional[list[str]] = None,
    top_k:     Optional[int]       = None,
) -> list[RetrievedItem]:
    """
    Search the Qdrant catalog for items matching the query.

    Args:
        query     : natural language search string (embedded internally)
        category  : "tops" | "bottoms" | "shoes"
        gender    : "men" | "women" | "unisex"
        max_price : upper price bound
        min_price : lower price bound
        colors    : list of color strings to match
        sources   : ["hm"] | ["zara"] | ["hm", "zara"]
        top_k     : number of results (defaults to settings.rag_top_k)

    Returns:
        List of RetrievedItem sorted by similarity score descending.
    """
    top_k  = top_k or settings.rag_top_k
    vector = embed_query(query)
    qfilter = build_filter(
        category=category,
        gender=gender,
        max_price=max_price,
        min_price=min_price,
        colors=colors,
        sources=sources,
    )

    log.info(
        f"[search] query='{query[:60]}' | category={category} | "
        f"gender={gender} | max_price={max_price} | top_k={top_k}"
    )

    client = get_async_client()

    results = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        query_filter=qfilter,
        limit=top_k,
        score_threshold=settings.rag_score_threshold,
        with_payload=True,
    )

    items = []

    for hit in results.points:
        p = hit.payload

        items.append(RetrievedItem(
            item_id=p.get("item_id", ""),
            name=p.get("name", ""),
            category=p.get("category", ""),
            source=p.get("source", ""),
            price=float(p.get("price", 0)),
            currency=p.get("currency", "USD"),
            color=p.get("color", ""),
            material=p.get("material", ""),
            description=p.get("description", ""),
            url=p.get("url", ""),
            image_url=p.get("image_url", ""),
            gender=p.get("gender", "unisex"),
            score=hit.score,
        ))

    top_score = round(items[0].score, 4) if items else 0
    log.info(f"[search] returned {len(items)} results | top_score={top_score}")
    return items


# ─── Outfit candidates ────────────────────────────────────────────────────────

async def get_outfit_candidates(
    query:     str,
    gender:    Optional[str]       = None,
    max_price: Optional[float]     = None,
    colors:    Optional[list[str]] = None,
) -> dict[str, list[RetrievedItem]]:
    """
    Retrieve candidates for all three outfit slots in parallel.
    Runs tops / bottoms / shoes searches concurrently using asyncio.gather.

    Returns:
        {
            "tops":    [RetrievedItem, ...],
            "bottoms": [RetrievedItem, ...],
            "shoes":   [RetrievedItem, ...],
        }
    """
    log.info(f"[get_outfit_candidates] query='{query[:60]}' | gender={gender} | max_price={max_price}")

    # Run all 3 category searches concurrently — 3x faster than sequential
    tops_task, bottoms_task, shoes_task = await asyncio.gather(
        search(query=query, category="tops",    gender=gender, max_price=max_price, colors=colors),
        search(query=query, category="bottoms", gender=gender, max_price=max_price),
        search(query=query, category="shoes",   gender=gender, max_price=max_price),
    )

    candidates = {
        "tops":    tops_task,
        "bottoms": bottoms_task,
        "shoes":   shoes_task,
    }

    for cat, results in candidates.items():
        log.info(f"[get_outfit_candidates] {cat}: {len(results)} candidates")

    return candidates

# ─── Standalone runner ────────────────────────────────────────────────────────

async def _run_interactive():
    print("\n" + "=" * 55)
    print("  Retriever Test — Standalone Search")
    print("  Type 'quit' to exit")
    print("=" * 55 + "\n")

    while True:
        query = input("Search query: ").strip()
        if not query or query.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        category = input("Category (tops / bottoms / shoes / leave empty for all): ").strip() or None
        
        gender_input = input("Gender (men / women / leave empty to search all): ").strip()
        gender = gender_input if gender_input else None

        max_price_input = input("Max price (leave empty for no limit): ").strip()
        max_price = float(max_price_input) if max_price_input else None

        print("\nSearching...\n")

        try:
            if category:
                results = await search(
                    query=query,
                    category=category,
                    gender=gender,
                    max_price=max_price,
                )
                _print_results(category, results)
            else:
                candidates = await get_outfit_candidates(
                    query=query,
                    gender=gender,
                    max_price=max_price,
                )
                for cat, items in candidates.items():
                    _print_results(cat, items)

        except Exception as e:
            print(f"[Error] {e}\n")

        print()


def _print_results(category: str, items: list):
    print(f"── {category.upper()} ({len(items)} results) ──")
    if not items:
        print("  No results found.\n")
        return
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item.name}")
        print(f"     Price    : ${item.price} | Color: {item.color} | Source: {item.source}")
        print(f"     Score    : {round(item.score, 4)}")
        print(f"     Desc     : {item.description[:100] if item.description else 'N/A'}")
        print()


if __name__ == "__main__":
    asyncio.run(_run_interactive())