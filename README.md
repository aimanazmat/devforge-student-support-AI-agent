# DEVFORGE Student Support AI Agent

A Python AI agent that helps DEVFORGE internship students with AI
engineering, web development, Python, FastAPI, LangChain, LangGraph,
GitHub, Render deployment, assignments, and general project guidance.

Built with **FastAPI**, **LangChain**, **LangGraph**, and an **Ollama Cloud**
model (e.g. Qwen). Deployed for free on **Render**.

## Architecture

The agent is a LangGraph workflow with five nodes:

```
                 ┌────────────────────┐
                 │ classify_question  │
                 └─────────┬──────────┘
              ┌────────────┼────────────┐
              ▼             ▼             ▼
      ┌──────────────┐ ┌──────────────────┐ ┌───────────────┐
      │  faq_lookup   │ │ ai_support_agent  │ │ safe_response  │
      │ (DEVFORGE FAQ)│ │ (Ollama Cloud LLM)│ │  (off-topic)   │
      └───────┬───────┘ └─────────┬─────────┘ └───────┬────────┘
              └────────────┬──────┴──────────────────┘
                            ▼
                  ┌───────────────────┐
                  │  format_response   │
                  │ (+ saves memory)   │
                  └───────────────────┘
```

1. **classify_question** — decides whether a question is a known FAQ,
   DEVFORGE/technical ("related"), or off-topic ("unrelated").
2. **faq_lookup** *(bonus)* — answers common DEVFORGE internship questions
   instantly, without calling the LLM.
3. **ai_support_agent** — sends related questions to an Ollama Cloud model
   (e.g. Qwen) via `langchain-ollama`.
4. **safe_response** — politely explains the agent's scope for unrelated
   questions.
5. **format_response** *(bonus)* — normalizes the final answer and appends
   the turn to LangGraph's conversation memory (`MemorySaver`, keyed by
   `session_id`) so follow-up questions retain context.

## Project Files

| File              | Purpose                                             |
|-------------------|------------------------------------------------------|
| `main.py`         | FastAPI app and endpoints (`/`, `/health`, `/chat`) |
| `agent.py`        | LangGraph workflow, nodes, and Ollama Cloud client  |
| `requirements.txt`| Python dependencies                                 |
| `.env.example`    | Placeholder environment variables                   |
| `.gitignore`      | Excludes `.env` and other local files from Git      |
| `render.yaml`     | Render Blueprint for one-click deployment           |

## Environment Variables

| Variable       | Description                              |
|----------------|-------------------------------------------|
| `OLLAMAAPIKEY` | Your Ollama Cloud API key                 |
| `OLLAMA_MODEL` | Ollama Cloud model name, e.g. `qwen2.5`   |
| `OLLAMA_HOST`  | (Optional) Ollama Cloud host, defaults to `https://ollama.com` |

Copy `.env.example` to `.env` and fill in your real key for local development:

```bash
cp .env.example .env
```

**Never commit your real `.env` file** — it's already excluded via
`.gitignore`.

## Run Locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env with your real API key
uvicorn main:app --reload
```

Visit:
- `http://127.0.0.1:8000/` — welcome message
- `http://127.0.0.1:8000/health` — health check
- `http://127.0.0.1:8000/docs` — interactive Swagger UI

## API Reference

### `GET /`
Returns a welcome message and the docs path.

### `GET /health`
Returns service health status and whether the Ollama API key is configured.

### `POST /chat`
Request body:
```json
{
  "message": "How can I deploy my Python AI agent on Render?",
  "session_id": "student-123"
}
```
Response:
```json
{
  "response": "🛠️ **DEVFORGE Support**\n\n...",
  "category": "related",
  "session_id": "student-123"
}
```

`session_id` is optional (defaults to `"default"`) and is used to keep
short conversation memory per student/session.

## Testing

Related question:
```bash
curl -X POST https://YOUR-RENDER-URL/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How can I deploy my Python AI agent on Render?"}'
```

Unrelated question:
```bash
curl -X POST https://YOUR-RENDER-URL/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who will win the next cricket match?"}'
```

## Deploying to Render

1. Push this repository to GitHub.
2. In Render, choose **New + → Blueprint** and point it at your repo (it
   will read `render.yaml`), or create a **New Web Service** manually with:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables in the Render dashboard:
   - `OLLAMAAPIKEY` = your real Ollama Cloud API key
   - `OLLAMA_MODEL` = `qwen2.5` (or another Ollama Cloud model)
4. Deploy, then verify `/`, `/health`, `/docs`, and `POST /chat` on the
   live URL.

## Error Handling

- If `OLLAMAAPIKEY` is missing, `/health` reports
  `"ollama_api_key_configured": false`, and `/chat` returns a clear
  configuration-error message instead of crashing.
- Any Ollama Cloud API failure (network, auth, rate limit) is caught and
  returned as a readable error message in the chat response rather than a
  500 crash.

## Bonus Features Implemented

- ✅ Conversation memory via LangGraph `MemorySaver` state, keyed by `session_id`
- ✅ FAQ node for common DEVFORGE internship questions
- ✅ Fourth LangGraph node (`format_response`) to format/normalize output
- ✅ Error handling for missing/invalid API keys and provider errors

## Security

- `.env` is git-ignored — only `.env.example` (placeholders) is committed.
- The real API key is set as an environment variable in Render, never in code.
