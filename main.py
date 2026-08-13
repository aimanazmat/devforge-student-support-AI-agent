"""
main.py
--------
FastAPI application for the DEVFORGE Student Support AI Agent.

Endpoints:
    GET  /        -> welcome message + docs path
    GET  /health  -> health check
    POST /chat    -> runs the LangGraph agent workflow on a student message
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()  # loads .env locally; on Render, real env vars are used instead

from agent import run_agent  # noqa: E402  (import after load_dotenv on purpose)

app = FastAPI(
    title="DEVFORGE Student Support AI Agent",
    description=(
        "A LangGraph-powered AI agent that helps DEVFORGE internship "
        "students with AI engineering, web development, Python, FastAPI, "
        "LangChain, LangGraph, GitHub, Render deployment, assignments, "
        "and project guidance."
    ),
    version="1.0.0",
)

# Allow a frontend (e.g. deployed on Vercel) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Student's question")
    session_id: str = Field(
        default="default",
        description="Optional conversation/session id for memory continuity",
    )


class ChatResponse(BaseModel):
    response: str
    category: str
    session_id: str


@app.get("/")
def root():
    """Welcome message and API documentation path."""
    return {
        "message": "👋 Welcome to the DEVFORGE Student Support AI Agent API",
        "docs": "/docs",
        "health": "/health",
        "chat_endpoint": "POST /chat",
    }


@app.get("/health")
def health():
    """Application health status, including basic config checks."""
    api_key_configured = bool(os.environ.get("OLLAMAAPIKEY"))
    return {
        "status": "ok",
        "service": "devforge-student-support-agent",
        "ollama_api_key_configured": api_key_configured,
        "ollama_model": os.environ.get("OLLAMA_MODEL", "qwen2.5"),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Receives a student message, runs it through the LangGraph workflow,
    and returns the AI agent's response."""
    try:
        result = run_agent(request.message, session_id=request.session_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Agent workflow failed: {exc}"
        ) from exc

    return ChatResponse(
        response=result["response"],
        category=result["category"],
        session_id=request.session_id,
    )
