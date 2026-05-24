# ARCHITECTURE

## Quickeee Luxury Stylist Concierge — Technical Design Document

---

## 1. Project Overview

Quickeee is an ultra-premium AI fashion concierge. Users interact with Quinn — an AI stylist — via natural conversation. Quinn understands the user's occasion, style preferences and budget, searches a real fashion catalog using semantic search, and recommends a complete outfit with product links.

---

## 2. Folder Structure

```
fashion/
├── main.py                          FastAPI entry point — runs on 0.0.0.0:8002
├── config.py                        Central settings loaded from .env
├── .env                             API keys and configuration
│
├── agent/
│   ├── styling_agent.py             LangGraph graph + Groq LLM + token/cost tracking
│   └── tools.py                     search_catalog tool — called by LLM when needed
│
├── prompt/
│   └── stylist_prompts.py           System prompt defining Quinn's persona and rules
│
├── api/
│   └── route.py                     FastAPI route handlers and request/response schemas
│
├── db/
│   ├── schemas.py                   Pydantic schemas shared across the application
│   └── db_client.py                 Qdrant sync + async client singletons
│
├── embeddings/
│   ├── ingest.py                    Embed catalog JSON → upsert to Qdrant
│   └── retriever.py                 Async semantic search with metadata pre-filtering
│
├── utils/
│   └── logger.py                    Centralised logging (console + rotating file)
│
└── data/
    └── catalog.json                 Fashion catalog (mock data — see note below)
```

---

## 3. System Architecture

### 3.1 Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| LLM | Groq — llama-3.3-70b-versatile | Fast inference, good tool calling support |
| Agent framework | LangGraph | Explicit state management, conditional routing |
| Vector DB | Qdrant | Purpose-built for vector search, payload filtering, async client |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, runs locally, 384-dim vectors |
| API | FastAPI | Async, auto Swagger docs, Pydantic validation |
| Config | pydantic-settings | Type-safe settings loaded from .env |
| Logging | Python stdlib logging | Structured console + file output |

---

### 3.2 Agent Design (LangGraph)

The agent uses a **tool-calling graph** with conditional routing:

```
[START]
   │
   ▼
chat_node  ──── Groq decides: plain reply OR call search_catalog?
   │
should_continue()
   │                    │
   ▼ tool call          ▼ plain reply
tools_node           [END]
   │
   ▼
chat_node  ──── Groq reads catalog results, writes outfit recommendation
   │
[END]
```

**Why this design:**
- Quinn only searches the catalog when the user actually asks for outfit recommendations
- For chitchat ("hello", "how are you") the tool is never called — saves tokens
- The graph loop allows multi-step reasoning (tool → read results → reply)

**State:** `AgentState` holds the full `messages` list using LangGraph's `add_messages` reducer. Every turn appends to the list — the LLM always sees the full conversation history.

---

### 3.3 RAG Pipeline

Two-phase retrieval for speed and relevance:

**Phase 1 — Metadata pre-filter (Qdrant payload indexes)**
```
Filter by: category (tops/bottoms/shoes) + gender + price range + color
```
Narrows the search space before any vector computation.

**Phase 2 — Vector similarity search (cosine)**
```
Query text → embed (384-dim) → cosine similarity → top-K results
```

**Parallel search:** `get_outfit_candidates()` runs tops, bottoms and shoes searches concurrently using `asyncio.gather` — 3x faster than sequential.

```python
tops, bottoms, shoes = await asyncio.gather(
    search(category="tops", ...),
    search(category="bottoms", ...),
    search(category="shoes", ...),
)
```

---

### 3.4 Qdrant Schema

**Collection:** `fashion_catalog`
**Vector size:** 384 (all-MiniLM-L6-v2)
**Distance:** Cosine

**Payload fields:**

| Field | Type | Indexed | Purpose |
|-------|------|---------|---------|
| item_id | string | No | Deduplication |
| name | string | No | Display |
| category | keyword | ✅ Yes | Pre-filter |
| source | keyword | ✅ Yes | Pre-filter |
| gender | keyword | ✅ Yes | Pre-filter |
| color | keyword | ✅ Yes | Pre-filter |
| price | float | ✅ Yes | Range filter |
| material | string | No | Display |
| description | string | No | Display |
| url | string | No | Product link |
| image_url | string | No | Product image |

---

### 3.5 Embedding Strategy

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Runs locally — no API cost
- 384-dimensional vectors
- Fast inference (~5ms per item)

**Embedding text format:**
```
"{name} | {category} | {color} | {material} | {description} | ${price}"
```

All semantically relevant fields combined into one string before embedding. Consistent between ingestion (`ingest.py`) and retrieval (`retriever.py`).

---

### 3.6 Token Economics

**Model pricing (llama-3.3-70b-versatile):**
- Input:  $0.59 per 1M tokens
- Output: $0.79 per 1M tokens

**Cost tracking:** Every LLM call logs per-turn and cumulative session costs:
```
[tokens:turn]    input=950 | output=141 | total=1091 | cost=$0.00067279
[tokens:session] input=950 | output=141 | total=1091 | cost=$0.00067279 | turns=1
```

**Token optimisation:**
- `temperature=0` for tool calling — prevents malformed tool call generation
- Tool output capped at 150 chars per item description
- RAG retrieves top 15 candidates per category — enough variety, small prompt

---

### 3.7 Conversation Memory

The LLM is stateless — it remembers nothing between API calls. Memory is maintained by:

**CLI:** A `history` list kept in the terminal loop, passed to every `chat()` call.

**API:** The client sends `history` in every request body and receives the updated `history` in every response. The server is fully stateless.

```
Request:  { "message": "...", "history": [...previous turns...] }
Response: { "reply": "...", "history": [...updated turns...] }
```

---

## 4. API Design

**Base URL:** `http://localhost:8002`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check — model + status |
| `/api/v1/chat` | POST | Chat with Quinn |
| `/docs` | GET | Swagger UI |

**Chat request:**
```json
{
  "message": "I need an outfit for a yacht party",
  "history": []
}
```

**Chat response:**
```json
{
  "reply": "For your yacht party I'd suggest...",
  "history": [...],
  "token_usage": { "input_tokens": 950, "output_tokens": 141, "total_tokens": 1091 },
  "session_tokens": { "input_tokens": 950, "output_tokens": 141, "total_tokens": 1091 }
}
```

---

## 5. Note on Scraping

> **Important:** Scraping fashion websites without explicit permission may violate their Terms of Service. Major retailers including H&M and Zara actively block automated scraping using bot protection services (Akamai, Cloudflare).
>
> **Legal alternatives for production use:**
> - Use official retailer APIs where available (e.g. ASOS Partner API, Zalando Partner Program)
> - License product data from fashion data providers (e.g. Lyst, Edited, Stylitics)
> - Build partnerships directly with brands for catalog access
>
> For this project, a realistic mock catalog (`data/catalog.json`) has been used to demonstrate the full RAG pipeline. The scraper architecture is designed to work with any data source that produces the standard catalog JSON format.

---

## 6. Data Flow Summary

```
User prompt
    │
    ▼
POST /api/v1/chat  (FastAPI)
    │
    ▼
chat_node  →  Groq (llama-3.3-70b-versatile)
    │
    ├── Plain reply → return to user
    │
    └── Tool call → search_catalog()
                        │
                        ▼
                  embed query (MiniLM)
                        │
                        ▼
                  Qdrant search
                  (metadata filter + cosine similarity)
                  tops / bottoms / shoes in parallel
                        │
                        ▼
                  formatted catalog results
                        │
                        ▼
                  chat_node → Groq reads results
                        │
                        ▼
                  outfit recommendation + URLs
                        │
                        ▼
                  return to user
```

---

## 7. Local Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add GROQ_API_KEY to .env

# 3. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 4. Ingest catalog
python -m embeddings.ingest --file data/catalog.json

# 5. Start API
python main.py
# Swagger UI → http://localhost:8002/docs

# 6. Terminal chat (test without API)
python -m agent.styling_agent
```
