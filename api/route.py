"""
api/route.py

All API endpoints for the Quickeee Stylist Concierge.
Schemas are imported from db/schemas.py.

Endpoints:
    GET  /health       -- health check
    POST /api/v1/chat  -- chat with Quinn the stylist
"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from config import settings
from utils.logger import setup_logger
from agent.styling_agent import chat, get_session_tokens
from db.schemas import MessageItem, TokenUsage, ChatRequest, ChatResponse, HealthResponse

log = setup_logger(__name__)

router = APIRouter()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def history_to_messages(history: list[MessageItem]) -> list[BaseMessage]:
    """Convert API MessageItem list to LangChain message objects."""
    messages: list[BaseMessage] = []
    for item in history:
        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            messages.append(AIMessage(content=item.content))
    return messages


def messages_to_history(messages: list[BaseMessage]) -> list[MessageItem]:
    """Convert LangChain message objects back to serialisable MessageItem list."""
    items: list[MessageItem] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            items.append(MessageItem(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            items.append(MessageItem(role="assistant", content=msg.content))
    return items


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    log.info("[GET /health] called")
    return HealthResponse(status="ok", model=settings.groq_model_name)


@router.post("/api/v1/chat", response_model=ChatResponse, tags=["Styling"], summary="Chat with Quinn the stylist",
    description="""
        Send a message to Quinn, your AI fashion stylist.
        **Multi-turn conversation:**
        1. First request: `message` = your text, `history` = `[]`
        2. Each next request: pass back the `history` array from the previous response

        **Response includes:**
        - `reply` — Quinn's response
        - `history` — updated conversation (pass back next time)
        - `token_usage` — input/output/total tokens **for this turn**
        - `session_tokens` — cumulative tokens **across all turns**
    """,
)

async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    log.info(
        f"[POST /api/v1/chat] "
        f"message='{request.message[:60]}{'...' if len(request.message) > 60 else ''}' "
        f"| history_turns={len(request.history)}"
    )

    history_messages = history_to_messages(request.history)

    try:
        reply, updated_messages, usage = chat(request.message, history_messages)
    except Exception as e:
        log.error(f"[POST /api/v1/chat] Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    updated_history = messages_to_history(updated_messages)
    session = get_session_tokens()

    log.info(
        f"[POST /api/v1/chat] done | "
        f"turn tokens={usage.get('total_tokens', 0)} | "
        f"session total={session['total']}"
    )

    return ChatResponse(
        reply=reply,
        history=updated_history,
        token_usage=TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ),
        session_tokens=TokenUsage(
            input_tokens=session["input"],
            output_tokens=session["output"],
            total_tokens=session["total"],
        ),
    )