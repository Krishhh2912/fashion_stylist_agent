"""
agent/styling_agent.py

LangGraph conversational fashion stylist agent with tool-calling support.

Run terminal chat:
    python -m agent.styling_agent
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import settings
from utils.logger import setup_logger
from prompt.stylist_prompts import SYSTEM_PROMPT
from agent.tool import search_catalog

log = setup_logger(__name__)

# ─── Tools ───────────────────────────────────────────────────────────────────

TOOLS = [search_catalog]

# ─── Pricing (llama-3.3-70b-versatile) ───────────────────────────────────────

_INPUT_COST_PER_M  = 0.59   # USD per 1M input tokens
_OUTPUT_COST_PER_M = 0.79   # USD per 1M output tokens


def _calc_cost(input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    """Returns (input_cost, output_cost, total_cost) in USD."""
    input_cost  = (input_tokens  / 1_000_000) * _INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * _OUTPUT_COST_PER_M
    return round(input_cost, 8), round(output_cost, 8), round(input_cost + output_cost, 8)


# ─── Session tracker ─────────────────────────────────────────────────────────

_session: dict = {
    "input_tokens":  0,
    "output_tokens": 0,
    "total_tokens":  0,
    "turns":         0,
    "total_cost":    0.0,
}


# ─── LLM (cached singleton) ──────────────────────────────────────────────────

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Return a cached LLM instance with tools bound. Initialised once."""
    global _llm
    if _llm is None:
        log.debug(
            f"Initialising ChatGroq | model={settings.groq_model_name} "
            f"| temperature={settings.groq_temperature}"
        )
        _llm = ChatGroq(
            model_name=settings.groq_model_name,
            temperature=settings.groq_temperature,
            groq_api_key=settings.groq_api_key,
        ).bind_tools(TOOLS)
        log.info("ChatGroq LLM initialised and tools bound")
    return _llm


# ─── State ───────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ─── Nodes ───────────────────────────────────────────────────────────────────

def chat_node(state: AgentState) -> AgentState:
    """Call Groq. May return a plain reply or a tool-call request."""
    log.info(f"[chat_node] history_length={len(state['messages'])}")

    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = get_llm().invoke(messages_with_system)

    _track_tokens(response)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            log.info(f"[chat_node] Tool call requested: {tc['name']} | args={tc['args']}")

    return {"messages": [response]}


def _track_tokens(response) -> None:
    """Extract token usage from the Groq response and accumulate session totals."""
    usage         = response.response_metadata.get("token_usage", {})
    input_tokens  = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens  = usage.get("total_tokens", input_tokens + output_tokens)

    input_cost, output_cost, turn_cost = _calc_cost(input_tokens, output_tokens)

    _session["input_tokens"]  += input_tokens
    _session["output_tokens"] += output_tokens
    _session["total_tokens"]  += total_tokens
    _session["turns"]         += 1
    _session["total_cost"]    += turn_cost

    log.info(
        f"[tokens:turn]    input={input_tokens} | output={output_tokens} | "
        f"total={total_tokens} | cost=${turn_cost:.8f} "
        f"(in=${input_cost:.8f} out=${output_cost:.8f})"
    )
    log.info(
        f"[tokens:session] input={_session['input_tokens']} | "
        f"output={_session['output_tokens']} | total={_session['total_tokens']} | "
        f"cost=${_session['total_cost']:.8f} | turns={_session['turns']}"
    )


def should_continue(state: AgentState) -> str:
    """
    Router: if the last message contains tool calls, go to the tools node.
    Otherwise the reply is ready — end the graph.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        log.info("[router] Tool call detected → routing to tools node")
        return "tools"
    log.info("[router] No tool call → ending")
    return "end"


# ─── Graph ───────────────────────────────────────────────────────────────────

def build_graph():
    log.debug("Building LangGraph state graph...")

    builder = StateGraph(AgentState)
    builder.add_node("chat",  chat_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.set_entry_point("chat")

    builder.add_conditional_edges(
        "chat",
        should_continue,
        {"tools": "tools", "end": "__end__"},
    )
    # After a tool executes, return to chat so Groq can read the result
    builder.add_edge("tools", "chat")

    graph = builder.compile()
    log.info("LangGraph graph compiled successfully")
    return graph


_graph = None


def get_graph():
    """Return the compiled LangGraph graph, building it once on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─── Public API ──────────────────────────────────────────────────────────────

def get_session_tokens() -> dict:
    """Return cumulative token usage for the current server session."""
    return {
        "input":  _session["input_tokens"],
        "output": _session["output_tokens"],
        "total":  _session["total_tokens"],
        "turns":  _session["turns"],
    }


def chat(user_message: str, history: list[BaseMessage]) -> tuple[str, list[BaseMessage], dict]:
    """
    Send one user message and return the agent reply.

    Args:
        user_message : latest user input string
        history      : prior conversation messages (HumanMessage + AIMessage)

    Returns:
        (reply_text, updated_history, turn_usage)
        turn_usage = { "input_tokens", "output_tokens", "total_tokens" }
    """
    log.info(f"[chat] User: '{user_message[:80]}{'...' if len(user_message) > 80 else ''}'")

    state_in = {"messages": history + [HumanMessage(content=user_message)]}
    result   = get_graph().invoke(state_in)

    updated_history: list[BaseMessage] = result["messages"]
    reply: str = updated_history[-1].content

    # Pull token counts from the last AI message in the updated history
    last_ai = next(
        (m for m in reversed(updated_history) if isinstance(m, AIMessage)),
        None,
    )
    raw = getattr(last_ai, "response_metadata", {}).get("token_usage", {}) if last_ai else {}
    turn_usage = {
        "input_tokens":  raw.get("prompt_tokens", 0),
        "output_tokens": raw.get("completion_tokens", 0),
        "total_tokens":  raw.get("total_tokens", 0),
    }

    log.info(f"[chat] Done | total_messages={len(updated_history)}")
    return reply, updated_history, turn_usage


# ─── CLI helpers ─────────────────────────────────────────────────────────────

def _print_session_summary() -> None:
    s = _session
    print(f"\n[Session Summary]")
    print(f"  Turns         : {s['turns']}")
    print(f"  Input  tokens : {s['input_tokens']}")
    print(f"  Output tokens : {s['output_tokens']}")
    print(f"  Total  tokens : {s['total_tokens']}")
    print(f"  Total  cost   : ${s['total_cost']:.8f}")


def _print_history(history: list[BaseMessage]) -> None:
    print("\n--- Conversation History ---")
    for msg in history:
        if isinstance(msg, HumanMessage):
            print(f"You  : {msg.content}\n")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"Quinn: {msg.content}\n")
    print("----------------------------\n")


# ─── CLI conversation loop ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Stylist Agent -- Terminal Chat")
    print("  Commands: 'history' | 'tokens' | 'clear' | 'quit'")
    print("=" * 60 + "\n")

    history: list[BaseMessage] = []

    print("Quinn: Hello! I'm Quinn, your personal stylist.")
    print("       Tell me what you're dressing for and I'll build the perfect look.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nQuinn: Stay stylish. Goodbye!")
            _print_session_summary()
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Quinn: Stay stylish. Goodbye!")
            _print_session_summary()
            break

        if user_input.lower() == "clear":
            history = []
            print("\n[Conversation cleared -- starting fresh]\n")
            continue

        if user_input.lower() == "tokens":
            _print_session_summary()
            print()
            continue

        if user_input.lower() == "history":
            _print_history(history)
            continue

        try:
            reply, history, _ = chat(user_input, history)
            print(f"\nQuinn: {reply}\n")
        except Exception as e:
            log.error(f"Agent error: {e}")
            print(f"\n[Error] {e}")
            print("Check GROQ_API_KEY in your .env file\n")
