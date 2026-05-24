"""
db/schemas.py
All Pydantic schemas used across the application.
Import from here everywhere — never define schemas inline in route files.
"""

from pydantic import BaseModel, Field

# ─── Shared ───────────────────────────────────────────────────────────────────
class MessageItem(BaseModel):
    """A single conversation turn."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message text")


class TokenUsage(BaseModel):
    """Token counts returned with every chat response."""
    input_tokens:  int = Field(default=0, description="Prompt tokens sent to the LLM")
    output_tokens: int = Field(default=0, description="Completion tokens returned by the LLM")
    total_tokens:  int = Field(default=0, description="input_tokens + output_tokens")


# ─── Chat ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Your message to the stylist",
        examples=["I have dark navy chinos -- what should I wear for a yacht party?"],
    )
    history: list[MessageItem] = Field(
        default=[],
        description=(
            "Previous conversation turns. "
            "First request: send []. "
            "Each subsequent request: pass back the history from the previous response."
        ),
    )

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Quinn's response")
    history: list[MessageItem] = Field(
        ...,
        description="Updated history -- pass this back on the next request to maintain context",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token counts for this turn only",
    )
    session_tokens: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Cumulative token counts across all turns in this session",
    )


# ─── Health ───────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    model: str = Field(..., description="Active LLM model name")