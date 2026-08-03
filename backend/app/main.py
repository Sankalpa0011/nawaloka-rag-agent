"""FastAPI service.

The agent is exposed behind a small, documented HTTP API rather than being
embedded in the UI, so the UI stays a thin client and the agent can be exercised
by curl, by tests, or by anything else.

`POST /chat` streams **typed SSE events**, not just answer tokens:

    route       which source was chosen, with confidence and rationale
    tool_call   which SQL tool ran, and how many rows it returned
    sources     retrieved chunks with their rerank scores
    cache       FAQ / semantic-cache hit or miss
    token       answer text
    trace       the full per-node trace
    done        final payload
    error       something failed

That is what lets the UI render a trace panel showing *why* an answer looks the
way it does, instead of asking anyone to take the routing on trust.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import answer as run_agent
from app.agent.graph import get_graph
from app.agent.state import new_state
from app.config import settings
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.memory import store
from app.retrieval.qdrant_store import collection_stats
from app.tools.db import healthcheck as db_health

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.starting")

    # Postgres seeds itself from db/init, but Qdrant comes up empty, so build
    # the index here on first boot - that keeps `docker compose up` the only
    # command anyone needs. Idempotent: a restart costs one count query.
    # Deliberately not fatal; a degraded /health beats an API that won't start.
    try:
        from app.ingest.ensure_seeded import ensure_seeded

        await asyncio.to_thread(ensure_seeded)
    except Exception as exc:
        log.warning("api.seed_failed", error=str(exc)[:200])

    # Warm the ONNX models and compile the graph so the first real request is
    # not paying a ~5 s cold start.
    try:
        from app.retrieval.embedder import warm_up

        await asyncio.to_thread(warm_up)
        get_graph()
        store.ensure_schema()
    except Exception as exc:
        log.warning("api.warmup_failed", error=str(exc)[:200])
    log.info("api.ready", providers=get_router().active_providers)
    yield
    log.info("api.stopping")


app = FastAPI(
    title="Nawaloka Hospitals - Agentic Assistant",
    description=(
        "Hybrid agent routing patient questions between a relational database "
        "(doctors, fees, schedules, lab tests, packages) and a vector store "
        "built from nawaloka.com."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = Field(None, description="omit to start a new thread")


class ChatResponse(BaseModel):
    answer: str
    route: str
    thread_id: str
    citations: list[dict] = []
    sql_calls: list[dict] = []
    trace: list[dict] = []
    cache: dict = {}
    memory: dict = {}
    elapsed_ms: int = 0
    abstained: bool = False


# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    db = db_health()
    try:
        vectors = collection_stats()
    except Exception as exc:
        vectors = {"exists": False, "error": str(exc)[:160]}

    llm = get_router()
    ok = bool(db.get("ok")) and bool(vectors.get("exists")) and bool(llm.active_providers)
    return {
        "status": "ok" if ok else "degraded",
        "database": db,
        "vector_store": vectors,
        "llm": llm.status(),
    }


@app.get("/", tags=["ops"])
async def root() -> dict[str, str]:
    return {
        "service": "nawaloka-agent",
        "docs": "/docs",
        "chat": "POST /chat (SSE) or POST /chat/sync",
    }


# ---------------------------------------------------------------------------
def _sse(event: str, data: Any) -> dict:
    return {"event": event, "data": json.dumps(data, default=str)}


def _events_for(node: str, update: dict) -> list[dict]:
    """Translate a LangGraph node update into typed SSE events."""
    events: list[dict] = []

    if node == "contextualize":
        mem = update.get("memory") or {}
        if mem.get("rewritten"):
            events.append(_sse("rewrite", {
                "original": mem.get("original_query"),
                "resolved": update.get("contextual_query"),
                "focus_entities": mem.get("focus_entities", {}),
            }))

    elif node == "check_safety":
        guard = update.get("guardrail") or {}
        if guard.get("verdict") not in (None, "CLEAR"):
            events.append(_sse("guardrail", guard))

    elif node == "faq":
        events.append(_sse("cache", update.get("cache") or {}))

    elif node == "classify":
        decision = update.get("route_decision") or {}
        events.append(_sse("route", {
            "route": update.get("route"),
            "confidence": decision.get("confidence"),
            "rationale": decision.get("rationale"),
            "overridden": decision.get("overridden"),
            "override_reason": decision.get("override_reason"),
            "signals": decision.get("signals"),
            "provider": decision.get("provider"),
            "sql_intent": decision.get("sql_intent"),
            "vector_intent": decision.get("vector_intent"),
        }))

    elif node == "sql_agent":
        for call in update.get("sql_calls") or []:
            events.append(_sse("tool_call", {
                "tool": call.get("tool"),
                "row_count": call.get("row_count"),
                "sql": call.get("sql_executed"),
                "notes": call.get("notes"),
                "elapsed_ms": call.get("elapsed_ms"),
                "error": call.get("error"),
            }))

    elif node == "retrieve":
        events.append(_sse("sources", {
            "retrieval": update.get("retrieval") or {},
            "documents": [
                {
                    "citation": d.get("citation"),
                    "url": d.get("url"),
                    "content_type": d.get("content_type"),
                    "rerank_score": d.get("rerank_score"),
                    "rank": d.get("rank"),
                }
                for d in update.get("documents") or []
            ],
        }))

    return events


async def _stream_answer(text: str) -> AsyncIterator[str]:
    """Emit the answer in word groups.

    Synthesis is a single non-streaming call (the answer is post-checked for
    unsupported prices before anyone sees it), so this replays it at a readable
    pace rather than dumping a wall of text.
    """
    words = text.split(" ")
    for i in range(0, len(words), 4):
        yield " ".join(words[i : i + 4]) + (" " if i + 4 < len(words) else "")
        await asyncio.sleep(0.02)


@app.post("/chat", tags=["chat"])
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or f"web-{uuid.uuid4().hex[:12]}"

    async def generator():
        started = time.perf_counter()
        final: dict[str, Any] = {}
        try:
            yield _sse("start", {"thread_id": thread_id, "message": request.message})

            state = new_state(request.message, thread_id=thread_id)
            async for chunk in get_graph().astream(state, stream_mode="updates"):
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    # In "updates" mode each node yields only its own additions.
                    # A plain dict.update() therefore REPLACES the accumulating
                    # lists, leaving the trace with just the last node's event -
                    # exactly the data the UI panel needs most.
                    for key in ("trace", "sql_calls", "llm_calls"):
                        if key in update:
                            final[key] = (final.get(key) or []) + list(update[key])
                    final.update({k: v for k, v in update.items()
                                  if k not in ("trace", "sql_calls", "llm_calls")})
                    for event in _events_for(node, update):
                        yield event

            answer_text = final.get("answer") or ""
            async for piece in _stream_answer(answer_text):
                yield _sse("token", {"text": piece})

            elapsed = int((time.perf_counter() - started) * 1000)
            yield _sse("trace", {"trace": final.get("trace") or []})
            yield _sse("done", {
                "answer": answer_text,
                "route": final.get("route"),
                "thread_id": thread_id,
                "citations": final.get("citations") or [],
                "abstained": final.get("abstained", False),
                "elapsed_ms": elapsed,
            })
        except Exception as exc:
            log.error("api.chat_failed", error=str(exc)[:300])
            yield _sse("error", {
                "message": "The assistant hit an internal error.",
                "detail": str(exc)[:300],
            })

    return EventSourceResponse(generator())


@app.post("/chat/sync", response_model=ChatResponse, tags=["chat"])
async def chat_sync(request: ChatRequest) -> ChatResponse:
    """Non-streaming variant, for tests and simple clients."""
    thread_id = request.thread_id or f"web-{uuid.uuid4().hex[:12]}"
    try:
        state = await run_agent(request.message, thread_id=thread_id)
    except Exception as exc:
        log.error("api.chat_sync_failed", error=str(exc)[:300])
        raise HTTPException(status_code=500, detail="agent failure") from exc

    return ChatResponse(
        answer=state.get("answer", ""),
        route=state.get("route", ""),
        thread_id=thread_id,
        citations=state.get("citations") or [],
        sql_calls=state.get("sql_calls") or [],
        trace=state.get("trace") or [],
        cache=state.get("cache") or {},
        memory=state.get("memory") or {},
        elapsed_ms=state.get("elapsed_ms", 0),
        abstained=state.get("abstained", False),
    )


# ---------------------------------------------------------------------------
@app.get("/threads/{thread_id}", tags=["chat"])
async def get_thread(thread_id: str) -> dict[str, Any]:
    state = store.get_state(thread_id)
    return {
        "thread_id": thread_id,
        "history": store.get_history(thread_id),
        "summary": state.summary,
        "focus_entities": state.focus_entities,
        "turn_count": state.turn_count,
    }


@app.delete("/threads/{thread_id}", tags=["chat"])
async def delete_thread(thread_id: str) -> dict[str, str]:
    store.clear_thread(thread_id)
    return {"status": "cleared", "thread_id": thread_id}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
