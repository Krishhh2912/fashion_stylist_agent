"""
config.py

Central settings — all loaded from .env via pydantic-settings.
Every module imports `settings` from here; nothing reads os.environ directly.

Usage:
    from config import settings
    print(settings.groq_model_name)
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── LLM ──────────────────────────────────────────────────────────────
    groq_api_key:      Optional[str]   = None
    groq_model_name:   Optional[str]   = None
    groq_temperature:  Optional[float] = None

    # ─── Qdrant ───────────────────────────────────────────────────────────
    qdrant_host:       Optional[str]   = None
    qdrant_port:       Optional[int]   = None
    qdrant_collection: Optional[str]   = None

    # ─── Embeddings ───────────────────────────────────────────────────────
    embedding_model:   Optional[str]   = None
    vector_size:       Optional[int]   = None

    # ─── RAG ──────────────────────────────────────────────────────────────
    rag_top_k:              Optional[int]   = None
    rag_score_threshold:    Optional[float] = None

    # ─── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: Optional[str] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Global singleton — import this everywhere
settings = get_settings()
