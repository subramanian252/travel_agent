# LazyPlan: Interview Guide

LazyPlan is an asynchronous multi-agent travel planner built with LangGraph, FastAPI, MCP tools, PostgreSQL checkpoints, and Human-in-the-Loop (HITL) review.

## One-minute explanation

“Instead of using one prompt to make a generic travel plan, I model the trip planner as a LangGraph workflow. A guardrail validates the request, a supervisor extracts trip constraints and selects specialist agents, then flight, hotel, weather, and budget agents supply context. The system creates a draft itinerary and pauses at a durable HITL checkpoint. The user can approve it or provide feedback; the same thread resumes to create a final response.”

## Architecture

```text
Browser UI
   │  SSE progress + approval/revision actions
   ▼
FastAPI
   │  create / resume / delete a trip thread
   ▼
LangGraph workflow ───── PostgreSQL checkpoints
   │
   ├── Guardrail → Supervisor
   │                 │
   │                 ├── Flight agent + aviation tools
   │                 ├── Hotel agent + Tavily MCP tools
   │                 ├── Weather agent + Weather MCP tools
   │                 └── Budget agent
   │
   └── Draft itinerary → HITL approval → final response
```

## End-to-end request flow

1. The UI requests a UUID with `POST /api/threads`.
2. It posts the trip brief and UUID to `POST /api/plan`.
3. FastAPI calls the async graph and converts its real node updates into Server-Sent Events (SSE).
4. The supervisor stores `selected_agents`; the frontend shows those exact agents in the live graph panel.
5. The itinerary agent creates a draft. `interrupt()` saves the graph state and pauses execution.
6. The user approves or rejects from the browser. `POST /api/threads/{thread_id}/review` resumes that same checkpoint.
7. Approval reaches the final agent. Rejection sends feedback to the supervisor, which can select only the agents that need to rerun.

## Small implementation snippets

### Async start and resume bridge

`agent.py` keeps the workflow logic in one place. The HTTP layer supplies a request, a thread ID, a callback for updates, and—when required—the review response.

```python
async def main(user_query, thread_id, on_update=None, resume=None):
    graph_input = Command(resume=resume) if resume is not None else {
        "user_query": user_query
    }

    async for update in app.astream(
        graph_input,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        if on_update:
            await on_update(update)
```

Why it matters: a new trip and a resumed HITL trip use the same async entry point without copying graph logic into FastAPI.

### Supervisor-driven agent selection

```python
def supervisor_router(state):
    return [Send(agent, state) for agent in state["selected_agents"]]
```

The supervisor produces structured constraints—destination, origin, duration, budget, travel style, and special requests—and chooses a subset of `flight`, `hotel`, `weather`, and `budget` agents. This avoids unnecessary tool calls.

### Tool-aware research agents

```python
response = await flight_llm.ainvoke(messages)

if not response.tool_calls:
    result["flight_results"] = response.content
```

When an LLM requests a tool, LangGraph routes to a `ToolNode` and loops back to the same research agent. When no tool call remains, the agent writes a concise result into state for the itinerary agent.

### Durable Human-in-the-Loop checkpoint

```python
review = interrupt({
    "draft_iternary": state["iternary"],
    "expected_response": {"approved": True, "feedback": "optional"},
})

approved = bool(review.get("approved", True))
feedback = str(review.get("feedback", ""))
```

`interrupt()` persists state in PostgreSQL. The app resumes with:

```python
Command(resume={"approved": False, "feedback": "Make the plan more budget friendly"})
```

### Real progress, not a fake loading sequence

```python
for node_name in update:
    await queue.put(("node", {"node": node_name, "status": "completed"}))
```

The FastAPI SSE bridge forwards actual LangGraph updates. The browser highlights completed graph nodes and receives the real `selected_agents` output from the supervisor.

### Simple thread lifecycle

```python
@app.post("/api/threads")
async def create_thread():
    return {"thread_id": str(uuid4())}
```

The UUID is stored in the browser. Choosing **New Trip** first calls `adelete_thread()` for the old UUID, after a clear irreversible-action warning, then creates a replacement UUID.

## Agent responsibilities

| Component | Responsibility |
| --- | --- |
| Guardrail | Rejects unsafe requests or requests without a clear destination. |
| Supervisor | Extracts constraints and selects the appropriate agents. |
| Flight | Uses airport/flight tools for route research. |
| Hotel | Uses Tavily MCP tools for accommodation research. |
| Weather | Uses the local Weather MCP server for forecast-aware activities and packing advice. |
| Budget | Produces clearly labelled travel-cost estimates. |
| Itinerary | Combines research into a reviewable draft. |
| HITL node | Pauses for approval or revision feedback. |
| Final agent | Creates the final response after approval. |

## Features to show an interviewer

- Live graph activity and actual supervisor-selected agents.
- SSE progress from LangGraph node updates.
- Markdown itinerary rendering, including lists and tables.
- Approve/revise HITL controls.
- Same-thread resume after feedback.
- PostgreSQL-backed checkpoint state endpoint.
- New-trip action with explicit permanent deletion warning.

## Interview questions and strong answers

### Why use LangGraph rather than a single prompt?

It gives explicit graph nodes, state, conditional routing, tool loops, and durable interrupts. The workflow is inspectable, testable, and easier to extend.

### How is it asynchronous?

The LLM calls use `ainvoke`, the graph uses `astream`, FastAPI streams SSE asynchronously, and persistence uses `AsyncPostgresSaver`. The server does not block while providers or MCP tools are running.

### What happens when a user rejects a draft?

The feedback is persisted in the thread state. The supervisor updates trip constraints and chooses the agents that need another pass before producing a revised draft.

### How do you make the UI progress truthful?

The frontend does not infer completion from a timer. FastAPI receives LangGraph update events, streams completed node names, and forwards the supervisor’s actual selected agents.

## Suggested live demo

1. Submit: “Plan a 4-day mid-range trip from Delhi to Paris, focused on museums and food.”
2. Show the supervisor-selected agents and live node flow.
3. Explain that the graph pauses before the final response.
4. Choose **Revise plan** and request a lower budget.
5. Approve the new draft and show the final response.
6. Point out that **New Trip** protects against silently losing checkpoint history.

## Stack

- Python 3.13+, FastAPI, Uvicorn
- LangGraph and LangChain
- OpenRouter-compatible `gpt-4o-mini`
- MCP tools: Tavily, Aviationstack, local Weather server
- PostgreSQL with `AsyncPostgresSaver`
- Vanilla HTML/CSS/JavaScript with SSE
