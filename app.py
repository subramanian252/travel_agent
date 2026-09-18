"""FastAPI interface for the streaming, HITL travel-planning workflow."""

from __future__ import annotations

import asyncio
import importlib
import json
from functools import lru_cache
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field


app = FastAPI(title="Routewise | AI Travel Planner", version="1.2.0")
templates = Jinja2Templates(directory="templates")


class PlanRequest(BaseModel):
    query: str = Field(..., min_length=8, max_length=1200)
    thread_id: str = Field(..., min_length=1, max_length=100)


class ReviewRequest(BaseModel):
    approved: bool
    feedback: str = Field(default="", max_length=1200)


STAGES = {
    "gaurdrail": (10, "Checking your trip brief…", 0),
    "supervisor": (20, "Assigning the right travel agents…", 0),
    "flight": (36, "Researching flight options…", 1),
    "flight_tools": (45, "Reading flight details…", 1),
    "hotel": (55, "Researching places to stay…", 1),
    "tavily_tools": (63, "Comparing hotel research…", 1),
    "weather": (72, "Checking destination weather…", 2),
    "weather_tools": (78, "Reading weather conditions…", 2),
    "budget": (84, "Building a practical budget…", 2),
    "iternary": (93, "Drafting your itinerary…", 3),
    "final": (98, "Preparing your final trip plan…", 3),
}


@lru_cache(maxsize=1)
def get_agent_module() -> Any:
    return importlib.import_module("agent")


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def interruption_from_update(update: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = update.get("__interrupt__", [])
    for item in interrupts:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            return value
    return None


def final_response_from_update(update: dict[str, Any]) -> str | None:
    for node_update in update.values():
        if not isinstance(node_update, dict):
            continue
        response = node_update.get("final_response")
        if isinstance(response, str) and response.strip():
            return response.strip()
    return None


async def workflow_events(
    thread_id: str,
    query: str = "",
    review: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Convert LangGraph updates and interrupts into browser-friendly SSE events."""
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
    final_response: str | None = None
    waiting_for_review = False

    async def on_update(update: dict[str, Any]) -> None:
        nonlocal final_response, waiting_for_review
        interrupt_payload = interruption_from_update(update)
        if interrupt_payload:
            waiting_for_review = True
            await queue.put(("node", {"node": "human_approval", "status": "waiting"}))
            await queue.put(("approval", interrupt_payload))
            return

        for node_name in update:
            await queue.put(("node", {"node": node_name, "status": "completed"}))
            if node_name in STAGES:
                value, label, stage = STAGES[node_name]
                await queue.put(("progress", {"value": value, "label": label, "stage": stage}))
            if node_name == "supervisor":
                supervisor_update = update[node_name]
                if isinstance(supervisor_update, dict):
                    await queue.put(("routing", {
                        "selected_agents": supervisor_update.get("selected_agents", []),
                        "reasoning": supervisor_update.get("supervisor_reasoning", ""),
                        "trip_constraints": supervisor_update.get("trip_constraints", {}),
                    }))
        final_response = final_response_from_update(update) or final_response

    async def run_workflow() -> None:
        try:
            await queue.put(("progress", {"value": 4, "label": "Starting your travel workflow…", "stage": 0}))
            agent = get_agent_module()
            await agent.main(query, thread_id, on_update, review)
            if final_response:
                await queue.put(("complete", {"itinerary": final_response, "thread_id": thread_id}))
            elif not waiting_for_review:
                raise ValueError("The workflow ended without a result or an approval request.")
        except Exception:
            await queue.put(("error", {"detail": "The travel planner is temporarily unavailable. Check the configured API keys and database connection, then try again."}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_workflow())
    try:
        while (event := await queue.get()) is not None:
            event_name, payload = event
            yield sse(event_name, payload)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health", tags=["System"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "routewise", "mode": "streaming-hitl"}


@app.post("/api/threads", tags=["Threads"])
async def create_thread() -> dict[str, str]:
    return {"thread_id": str(uuid4())}


@app.delete("/api/threads/{thread_id}", status_code=204, tags=["Threads"])
async def delete_thread(thread_id: str) -> None:
    try:
        agent = get_agent_module()
        async with AsyncPostgresSaver.from_conn_string(agent.get_database_url()) as checkpointer:
            await checkpointer.setup()
            await checkpointer.adelete_thread(thread_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Unable to remove the previous trip thread.") from exc


@app.get("/api/threads/{thread_id}/state", tags=["Threads"])
async def thread_state(thread_id: str) -> dict[str, Any]:
    """Return the small, UI-safe part of the latest persisted graph state."""
    try:
        agent = get_agent_module()
        config = {"configurable": {"thread_id": thread_id}}
        async with AsyncPostgresSaver.from_conn_string(agent.get_database_url()) as checkpointer:
            await checkpointer.setup()
            checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return {"thread_id": thread_id, "exists": False}

        values = checkpoint_tuple.checkpoint["channel_values"]
        return {
            "thread_id": thread_id,
            "exists": True,
            "selected_agents": values.get("selected_agents", []),
            "trip_constraints": values.get("trip_constraints", {}),
            "approval_request": values.get("approval_request"),
            "final_response": values.get("final_response"),
            "updated_channels": checkpoint_tuple.checkpoint.get("updated_channels", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Unable to load the trip state.") from exc


@app.post("/api/plan", tags=["Planner"])
async def create_plan(payload: PlanRequest) -> StreamingResponse:
    return StreamingResponse(
        workflow_events(payload.thread_id, query=payload.query.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/threads/{thread_id}/review", tags=["Planner"])
async def review_plan(thread_id: str, payload: ReviewRequest) -> StreamingResponse:
    return StreamingResponse(
        workflow_events(thread_id, review=payload.model_dump()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
