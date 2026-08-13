"""
agent.py
---------
DEVFORGE Student Support AI Agent — LangGraph workflow.

Workflow nodes:
    1. classify_question   -> decides if the question is DEVFORGE/technical related
    2. faq_lookup           -> (bonus) answers common DEVFORGE internship FAQs directly
    3. ai_support_agent      -> sends the question to an Ollama Cloud model (e.g. Qwen)
    4. safe_response          -> polite refusal for unrelated questions
    5. format_response      -> (bonus) normalizes/cleans the final answer shape

State is kept per-session using LangGraph's MemorySaver checkpointer, so a
conversation has short-term memory across turns (bonus requirement).
"""

import os
from typing import List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------
# NOTE: variable names are fixed by the project spec.
OLLAMA_API_KEY = os.environ.get("OLLAMAAPIKEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5")
OLLAMA_CLOUD_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")

MISSING_KEY_MSG = (
    "⚠️ The AI Support Agent is not fully configured: the OLLAMAAPIKEY "
    "environment variable is missing. Please set it (see .env.example) and "
    "restart the service."
)

SYSTEM_PROMPT = """You are the DEVFORGE Student Support AI Agent.
You help DEVFORGE internship students with:
- AI Engineering
- Web Development
- Python
- FastAPI
- LangChain
- LangGraph
- GitHub
- Render deployment
- Student assignments
- Project guidance
- General DEVFORGE internship learning support

Answer clearly, concisely, and helpfully, with practical steps or short code
snippets where useful. Keep a friendly, mentor-like tone suitable for
students learning to build software."""

# Simple keyword-based classifier. Kept dependency-free and fast; the LLM
# itself is only invoked once the question is already known to be on-topic,
# which also protects the Ollama Cloud quota from irrelevant chatter.
DEVFORGE_KEYWORDS = [
    "ai", "artificial intelligence", "engineer", "web dev", "website",
    "html", "css", "javascript", "react", "node", "python", "fastapi",
    "flask", "django", "langchain", "langgraph", "llm", "ollama", "qwen",
    "github", "git", "repo", "repository", "render", "deploy", "deployment",
    "assignment", "project", "internship", "devforge", "task", "bug",
    "error", "debug", "code", "coding", "api", "endpoint", "database",
    "docker", "server", "backend", "frontend", "model", "prompt", "agent",
    "workflow", "rag", "vector", "embedding", "package", "library",
    "install", "environment variable", "venv", "requirements.txt",
]

# Bonus: a small FAQ knowledge base answered directly, without calling the LLM.
FAQ_DB = {
    "what is devforge": (
        "DEVFORGE is the internship program this agent supports — it helps "
        "students learn AI engineering, web development, and modern "
        "developer tooling (Python, FastAPI, LangChain, LangGraph, GitHub, "
        "and Render) through hands-on projects."
    ),
    "how do i submit my assignment": (
        "Push your completed project to your GitHub repository, make sure "
        "it includes a clear README, deploy it (e.g. on Render), and submit "
        "your GitHub repo link plus your live deployment link."
    ),
    "how do i deploy on render": (
        "Push your code to GitHub, create a new Web Service on Render "
        "linked to that repo, set your start command to "
        "`uvicorn main:app --host 0.0.0.0 --port $PORT`, add your "
        "environment variables in the Render dashboard, and deploy."
    ),
    "what is langgraph": (
        "LangGraph is a library for building stateful, multi-step agent "
        "workflows as a graph of nodes (steps) and edges (transitions), "
        "built to work with LangChain."
    ),
}


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    messages: List[dict]          # running conversation, bonus memory
    question: str                 # latest user question
    category: Literal["related", "unrelated", "faq"]
    answer: str                   # raw answer before formatting
    response: str                 # final formatted answer
    error: str                    # populated on configuration/runtime errors


# ---------------------------------------------------------------------------
# Node 1: Question Classification
# ---------------------------------------------------------------------------
def classify_question(state: AgentState) -> AgentState:
    question = state.get("question", "").strip()
    lowered = question.lower()

    if not question:
        state["category"] = "unrelated"
        return state

    # Bonus FAQ short-circuit: check for a close FAQ match first.
    for faq_key in FAQ_DB:
        if faq_key in lowered or lowered in faq_key:
            state["category"] = "faq"
            return state

    is_related = any(keyword in lowered for keyword in DEVFORGE_KEYWORDS)
    state["category"] = "related" if is_related else "unrelated"
    return state


def route_after_classification(state: AgentState) -> str:
    category = state.get("category", "unrelated")
    if category == "faq":
        return "faq_lookup"
    if category == "related":
        return "ai_support_agent"
    return "safe_response"


# ---------------------------------------------------------------------------
# Node 2 (bonus): FAQ lookup
# ---------------------------------------------------------------------------
def faq_lookup(state: AgentState) -> AgentState:
    lowered = state.get("question", "").lower().strip()
    for faq_key, faq_answer in FAQ_DB.items():
        if faq_key in lowered or lowered in faq_key:
            state["answer"] = faq_answer
            return state
    # Fallback safety net; should not normally trigger.
    state["answer"] = "I couldn't find that in the DEVFORGE FAQ."
    return state


# ---------------------------------------------------------------------------
# Node 3: AI Support Agent (Ollama Cloud)
# ---------------------------------------------------------------------------
def ai_support_agent(state: AgentState) -> AgentState:
    if not OLLAMA_API_KEY:
        state["error"] = MISSING_KEY_MSG
        state["answer"] = MISSING_KEY_MSG
        return state

    try:
        # Imported lazily so the app can still start (and /health can still
        # respond) even if this dependency has an issue.
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_CLOUD_HOST,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
            },
        )

        history = state.get("messages", [])
        lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history[-10:]:  # keep last 10 turns of memory
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        lc_messages.append(HumanMessage(content=state["question"]))

        result = llm.invoke(lc_messages)
        state["answer"] = result.content.strip()
    except Exception as exc:  # noqa: BLE001 - surface any provider error safely
        state["error"] = str(exc)
        state["answer"] = (
            "Sorry — I ran into an error contacting the Ollama Cloud model. "
            f"Details: {exc}"
        )

    return state


# ---------------------------------------------------------------------------
# Node 4: Safe Response
# ---------------------------------------------------------------------------
def safe_response(state: AgentState) -> AgentState:
    state["answer"] = (
        "I'm the DEVFORGE Student Support AI Agent 🤖 — I can only help with "
        "DEVFORGE internship learning, AI engineering, web development, "
        "Python, FastAPI, LangChain, LangGraph, GitHub, Render deployment, "
        "assignments, and project guidance. Could you rephrase your "
        "question so it relates to one of those topics?"
    )
    return state


# ---------------------------------------------------------------------------
# Node 5 (bonus): Format Response
# ---------------------------------------------------------------------------
def format_response(state: AgentState) -> AgentState:
    answer = state.get("answer", "").strip()
    category = state.get("category", "unrelated")

    prefix = {
        "related": "🛠️ **DEVFORGE Support**",
        "faq": "📘 **DEVFORGE FAQ**",
        "unrelated": "ℹ️ **Notice**",
    }.get(category, "")

    formatted = f"{prefix}\n\n{answer}" if prefix else answer
    state["response"] = formatted

    # Append to conversation memory (bonus: LangGraph state memory).
    messages = state.get("messages", [])
    messages.append({"role": "user", "content": state.get("question", "")})
    messages.append({"role": "assistant", "content": formatted})
    state["messages"] = messages
    return state


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("faq_lookup", faq_lookup)
    graph.add_node("ai_support_agent", ai_support_agent)
    graph.add_node("safe_response", safe_response)
    graph.add_node("format_response", format_response)

    graph.set_entry_point("classify_question")

    graph.add_conditional_edges(
        "classify_question",
        route_after_classification,
        {
            "faq_lookup": "faq_lookup",
            "ai_support_agent": "ai_support_agent",
            "safe_response": "safe_response",
        },
    )

    graph.add_edge("faq_lookup", "format_response")
    graph.add_edge("ai_support_agent", "format_response")
    graph.add_edge("safe_response", "format_response")
    graph.add_edge("format_response", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Compiled once at import time and reused across requests.
agent_graph = build_graph()


def run_agent(question: str, session_id: str = "default") -> dict:
    """Runs the LangGraph workflow for a single student question.

    session_id enables per-conversation memory via the MemorySaver
    checkpointer (bonus requirement).
    """
    config = {"configurable": {"thread_id": session_id}}
    result = agent_graph.invoke({"question": question}, config=config)
    return {
        "response": result.get("response", ""),
        "category": result.get("category", "unrelated"),
        "error": result.get("error"),
    }
