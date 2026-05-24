"""
agent/tool.py

LangChain tool that Quinn can call to search the fashion catalog.
The LLM decides when to call this — only when the user asks for outfit recommendations.
"""

import asyncio
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from utils.logger import setup_logger
from embeddings.retriever import get_outfit_candidates

log = setup_logger(__name__)

# Category display order used when formatting results for the LLM
_CATEGORIES = ("tops", "bottoms", "shoes")


# ─── Input schema ─────────────────────────────────────────────────────────────
# Explicit Pydantic schema gives LangChain full control over what JSON schema
# is sent to Groq. Optional fields with no default force the model to omit them
# rather than passing the string "null".

class SearchCatalogInput(BaseModel):
    query: str = Field(
        description='Describe the style or occasion e.g. "casual summer yacht party outfit"'
    )
    gender: str = Field(
        default="men",
        description='"men" or "women"',
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Maximum price per item in USD. Omit if no budget constraint.",
    )
    colors: Optional[list[str]] = Field(
        default=None,
        description='Preferred colors e.g. ["white", "navy"]. Omit if no color preference.',
    )


# ─── Formatting ───────────────────────────────────────────────────────────────

def _format_candidates(candidates: dict) -> str:
    """Format outfit candidates as readable text for the LLM to reason over."""
    lines = []
    for category in _CATEGORIES:
        items = candidates.get(category, [])
        lines.append(f"\n── {category.upper()} ({len(items)} options) ──")

        if not items:
            lines.append("  No items found for this category.")
            continue

        for i, item in enumerate(items, 1):
            lines.append(
                f"  {i}. [{item.item_id}] {item.name}\n"
                f"     Price    : ${item.price} {item.currency}\n"
                f"     Category : {item.category} | Gender: {item.gender}\n"
                f"     Color    : {item.color} | Material: {item.material}\n"
                f"     Source   : {item.source}\n"
                f"     Desc     : {item.description[:150] if item.description else 'N/A'}\n"
                f"     URL      : {item.url}\n"
                f"     Image    : {item.image_url}\n"
                f"     Score    : {round(item.score, 4)}"
            )

    return "\n".join(lines)


# ─── Tool implementation ──────────────────────────────────────────────────────

def _search_catalog(
    query: str,
    gender: str = "men",
    max_price: Optional[float] = None,
    colors: Optional[list[str]] = None,
) -> str:
    log.info(
        f"[search_catalog] query='{query}' | gender={gender} "
        f"| max_price={max_price} | colors={colors}"
    )

    # get_outfit_candidates is async; run it in a new event loop.
    # LangGraph's ToolNode calls tools synchronously, so asyncio.run is correct here.
    candidates = asyncio.run(
        get_outfit_candidates(
            query=query,
            gender=gender,
            max_price=max_price,
            colors=colors,
        )
    )

    result = _format_candidates(candidates)

    log.info(
        f"[search_catalog] returning candidates | "
        f"tops={len(candidates.get('tops', []))} "
        f"bottoms={len(candidates.get('bottoms', []))} "
        f"shoes={len(candidates.get('shoes', []))}"
    )
    return result


# ─── Exported tool ────────────────────────────────────────────────────────────
# StructuredTool with an explicit args_schema gives LangChain full control over
# the JSON schema sent to Groq, preventing the model from passing "null" strings
# for optional fields.

search_catalog = StructuredTool.from_function(
    func=_search_catalog,
    name="search_catalog",
    description=(
        "Search the fashion catalog for outfit candidates. "
        "Call this whenever the user asks for outfit recommendations, "
        "clothing suggestions, or what to wear for any occasion."
    ),
    args_schema=SearchCatalogInput,
)
