"""FastAPI entry point for the streaming Travel Multi-Agent Planner."""

from __future__ import annotations

import asyncio
import importlib
import json
from functools import lru_cache
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


app = FastAPI(
    title="Routewise | AI Travel Planner",
    description="Streaming FastAPI interface for the LangGraph travel workflow.",
    version="1.1.0",
)
templates = Jinja2Templates(directory="templates")


class PlanRequest(BaseModel):
    query: str = Field(..., min_length=8, max_length=1200)


STAGES = {
    "flight": (25, "Flight agent is mapping your route…", 0),
    "flight_tools": (42, "Flight options found. Reading the details…", 0),
    "hotel": (58, "Hotel agent is researching stays…", 1),
    "tavily_tools": (70, "Comparing hotel research…", 1),
    "iternary": (86, "Itinerary agent is shaping your days…", 2),
    "final": (96, "Finalizing your trip blueprint…", 2),
}


@lru_cache(maxsize=1)
def get_agent_module() -> Any:
    """Import the asynchronous agent only when a plan is requested."""
    return importlib.import_module("agent")


def content_from_update(update: dict[str, Any]) -> str | None:
    """Extract the final AI message from a LangGraph node update."""
    final_update = update.get("final")
    if not isinstance(final_update, dict):
        return None
    messages = final_update.get("messages", [])
    if not messages:
        return None
    content = getattr(messages[-1], "content", None)
    return content.strip() if isinstance(content, str) and content.strip() else None


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def plan_events(query: str) -> AsyncIterator[str]:
    """Run the graph and expose its node updates as server-sent events."""
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
    thread_id = str(uuid4())
    final_itinerary: str | None = None

    async def on_update(update: dict[str, Any]) -> None:
        nonlocal final_itinerary
        for node_name in update:
            stage = STAGES.get(node_name)
            if stage:
                value, label, active_stage = stage
                await queue.put(("progress", {"value": value, "label": label, "stage": active_stage}))
        final_itinerary = content_from_update(update) or final_itinerary

    async def run_workflow() -> None:
        try:
            await queue.put(("progress", {"value": 8, "label": "Travel agents are starting…", "stage": 0}))
            agent = get_agent_module()
            await agent.main(query, thread_id, on_update)
            if not final_itinerary:
                raise ValueError("The travel workflow returned no final itinerary.")
            await queue.put(("complete", {"itinerary": final_itinerary, "thread_id": thread_id}))
        except Exception:
            await queue.put(("error", {"detail": "The travel planner is temporarily unavailable. Check the configured API keys and database connection, then try again."}))
        finally:
            await queue.put(None)

    workflow = asyncio.create_task(run_workflow())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            event_name, payload = event
            yield sse(event_name, payload)
    finally:
        if not workflow.done():
            workflow.cancel()
        await asyncio.gather(workflow, return_exceptions=True)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health", tags=["System"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "routewise", "mode": "streaming"}


@app.post("/api/plan", tags=["Planner"])
async def create_plan(payload: PlanRequest) -> StreamingResponse:
    return StreamingResponse(
        plan_events(payload.query.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
