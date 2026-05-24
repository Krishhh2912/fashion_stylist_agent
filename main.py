"""
main.py

Application entry point. Registers routes and starts the server.

Run:
    python main.py

Swagger UI:
    http://localhost:8002/docs
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.db_client import init_collection, health_check

from config import settings
from utils.logger import setup_logger
from agent.styling_agent import get_graph
from api.route import router

log = setup_logger(__name__)

APP_VERSION = "1.0.0"


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    log.info("=" * 50)
    log.info("Starting Luxury Stylist API")
    log.info("=" * 50)
    get_graph()

    qdrant_ok = health_check()
    if qdrant_ok:
        log.info("Qdrant connected")
        init_collection()
    else:
        log.warning("Qdrant unavailable — RAG features will not work")
    
    log.info("LangGraph graph warmed up and ready")
    
    yield
    log.info("Shutting down Luxury Stylist API")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Luxury Stylist Concierge",
    description=(
        "AI-powered fashion concierge. "
        "Chat with Quinn, your personal stylist, "
        "to get outfit recommendations tailored to your occasion and wardrobe."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register routes ──────────────────────────────────────────────────────────

app.include_router(router)


@app.get("/", tags=["System"])
def root():
    return {
        "service": "Luxury Stylist Concierge",
        "version": APP_VERSION,
        "docs":    "http://localhost:8002/docs",
        "health":  "http://localhost:8002/health",
        "chat":    "POST http://localhost:8002/api/v1/chat",
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)