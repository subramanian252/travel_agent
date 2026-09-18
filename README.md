# LazyPlan — Multi-Agent Travel Planner

LazyPlan is a full-stack, asynchronous travel-planning application built with LangGraph and FastAPI. It researches travel options through specialist agents, streams live workflow progress to the browser, and uses Human-in-the-Loop approval before generating a final itinerary.

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-1C3C3C)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-checkpoints-4169E1?logo=postgresql&logoColor=white)

## Highlights

- Guardrail, supervisor, flight, hotel, weather, budget, itinerary, approval, and final-response graph nodes.
- Supervisor selects only the agents needed for the current trip.
- MCP-powered hotel, flight/airport, and weather research.
- Real Server-Sent Event (SSE) updates for graph-node progress and supervisor-selected agents.
- Durable Human-in-the-Loop review with LangGraph `interrupt()` and PostgreSQL checkpoints.
- Approve or revise a draft using the same persistent trip thread.
- New-trip action permanently removes the prior checkpointed thread only after confirmation.
- Portfolio-style responsive frontend with Markdown itinerary and table rendering.

## Architecture

```text
Browser UI
  │  SSE updates / HITL review
  ▼
FastAPI
  │
  ▼
LangGraph workflow ── AsyncPostgresSaver
  │
  ├─ Guardrail → Supervisor
  ├─ Flight / Hotel / Weather / Budget agents
  ├─ Itinerary draft
  ├─ Human approval interrupt
  └─ Final response
```

Read the detailed [interview guide](docs/INTERVIEW_GUIDE.md) for implementation snippets, architecture explanations, and a demo script.

## State flow

```mermaid
flowchart TD
    A[User trip brief] --> B[Guardrail]
    B -->|safe + destination present| C[Supervisor]
    B -->|unsafe or incomplete| Z[Return a clear reason]
    C --> D{Selected agents}
    D --> E[Flight research]
    D --> F[Hotel research]
    D --> G[Weather research]
    D --> H[Budget estimates]
    E --> I[Itinerary draft]
    F --> I
    G --> I
    H --> I
    I --> J{{HITL interrupt}}
    J -->|Approve| K[Final travel response]
    J -->|Revise with feedback| C
```

```mermaid
sequenceDiagram
    participant U as User / Browser
    participant API as FastAPI
    participant G as LangGraph
    participant DB as PostgreSQL

    U->>API: Create trip thread
    API-->>U: UUID thread ID
    U->>API: Start plan + thread ID
    API->>G: astream(new request)
    G->>DB: Save checkpoints
    G-->>API: Node updates and agent selection
    API-->>U: SSE progress events
    G-->>API: HITL interrupt + draft
    API-->>U: Review request
    U->>API: Approve or revise feedback
    API->>G: Command(resume=review)
    G->>DB: Resume same checkpoint
    G-->>API: Final plan
    API-->>U: SSE final response
```

## Quick start

### 1. Install dependencies

This project uses Python 3.13+ and `uv`.

```bash
uv sync
```

### 2. Configure environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_key
EXTERNAL_END_POINT=postgresql://user:password@host/database
TAVILY_API_KEY=your_tavily_key
AVIATION_STACK_API_KEY=your_aviationstack_key
OPENWEATHER_API_KEY=your_openweather_key
```

`EXTERNAL_END_POINT` must point to a PostgreSQL database accessible by the application. SSL is added automatically when absent.

### 3. Run the app

```bash
uv run uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## User flow

1. The guardrail validates the request.
2. The supervisor extracts constraints and selects appropriate agents.
3. Selected flight, hotel, weather, and/or budget agents gather context.
4. The itinerary agent produces a reviewable draft.
5. The graph pauses for Human-in-the-Loop approval.
6. Approval creates the final response; feedback returns the workflow to the supervisor for a focused revision.

## Feature walkthrough

### 1. Guardrail validation

The graph first checks whether a request is safe and contains a clear destination. If not, it ends early with a concise explanation instead of calling research tools unnecessarily.

### 2. Supervisor-based orchestration

The supervisor extracts trip constraints such as origin, destination, duration, budget, travel style, and special requests. It chooses the relevant specialist agents, and the UI displays the exact selection in the live graph panel.

### 3. Specialist research agents

- **Flight agent:** resolves flight information through airport and aviation tools.
- **Hotel agent:** uses Tavily MCP search for relevant accommodation research.
- **Weather agent:** calls the local Weather MCP server for travel-aware conditions and forecasts.
- **Budget agent:** creates estimated spending guidance and labels it as an estimate.

### 4. Streamed graph activity

FastAPI converts LangGraph `astream` updates to Server-Sent Events. The browser sees actual completed nodes and selected agents, rather than relying on a fake loading timer.

### 5. Human-in-the-Loop review

The itinerary agent creates a draft. LangGraph `interrupt()` saves the graph state in PostgreSQL and pauses. The user can approve the draft or send revision feedback; feedback resumes the same thread and lets the supervisor rerun only the required work.

### 6. Thread lifecycle

The browser uses a UUID trip thread. The **New Trip** button asks for confirmation, permanently deletes the current checkpointed thread, and starts a new isolated trip. This makes the destructive action explicit.

## Technology stack

| Layer | Technology | Why it is used |
| --- | --- | --- |
| Language | Python 3.13+ | Main backend and agent implementation. |
| Agent orchestration | LangGraph + LangChain | Stateful nodes, conditional routing, tool loops, streaming, and HITL interrupts. |
| Model provider | OpenRouter-compatible `gpt-4o-mini` | Tool calling and structured supervisor output. |
| API | FastAPI + Uvicorn | Async HTTP endpoints and Server-Sent Events. |
| Persistence | PostgreSQL + `AsyncPostgresSaver` | Durable graph checkpoints and thread deletion/resume. |
| Tool protocol | Model Context Protocol (MCP) | Standard connection layer for external research tools. |
| Research tools | Tavily, Aviationstack, local Weather MCP server | Hotel, flight/airport, and weather context. |
| Frontend | HTML, CSS, vanilla JavaScript | Lightweight UI, SSE stream reader, HITL controls, and Markdown renderer. |
| Configuration | `python-dotenv` | Local environment-variable management. |

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Travel planner UI. |
| `GET` | `/api/health` | Health check. |
| `POST` | `/api/threads` | Create a UUID trip thread. |
| `DELETE` | `/api/threads/{thread_id}` | Permanently delete a thread and its checkpoints. |
| `GET` | `/api/threads/{thread_id}/state` | Read safe data from the latest checkpoint. |
| `POST` | `/api/plan` | Start a trip workflow; response is `text/event-stream`. |
| `POST` | `/api/threads/{thread_id}/review` | Resume a HITL thread; response is `text/event-stream`. |

### Start a plan

```json
POST /api/plan
{
  "thread_id": "your-thread-id",
  "query": "Plan a 4-day mid-range trip from Delhi to Paris focused on museums and food."
}
```

### Revise a draft

```json
POST /api/threads/{thread_id}/review
{
  "approved": false,
  "feedback": "Reduce the budget and include more local food options."
}
```

## Project structure

```text
agent.py                  # LangGraph workflow, agents, tools, HITL interrupt
app.py                    # FastAPI routes, SSE bridge, checkpoint/thread lifecycle
templates/index.html      # Responsive frontend and stream/HITL client logic
test_mcp_client.py        # Tavily, Aviationstack, and Weather MCP setup
weather_mcp_server.py     # Local MCP weather tools
docs/INTERVIEW_GUIDE.md   # Architecture and interview walkthrough
```

## Key frontend features

- Live graph activity panel with actual completed LangGraph nodes.
- Real supervisor-selected agent pills.
- Streamed progress bar.
- HITL approve/revise panel.
- Persistent thread display and explicit destructive New Trip control.

## License

Add a license appropriate for your GitHub repository before publishing.
