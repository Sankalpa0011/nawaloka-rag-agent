from __future__ import annotations

import asyncio

from app.agent.state import AgentState, trace_event
from app.logging_conf import get_logger
from app.retrieval.search import hybrid_search

log = get_logger(__name__)


async def retrieve_node(state: AgentState) -> dict:

    fallback = state.get("route") == "SQL"
    query = (
        state.get("vector_intent")
        or state.get("contextual_query")
        or state["user_query"]
    )

    result = await asyncio.to_thread(hybrid_search, query)

    documents = []
    for c in result.chunks:
        documents.append({
            "chunk_id": c.chunk_id,
            "citation": c.citation,
            "url": c.url,
            "page_label": c.page_label,
            "section_heading": c.section_heading,
            "content_type": c.content_type,
            "body": c.body,
            "has_pricing": c.has_pricing,
            "rerank_score": c.rerank_score,
            "rank": c.rank,
        })

    log.info("retrieve.done", query=query[:80], returned=len(documents),
             dropped=result.dropped_below_floor, ms=result.elapsed_ms,
             fallback=fallback)

    return {
        "documents": documents,
        "vector_fallback_used": fallback,
        "retrieval": {
            "query": query,
            "candidates": result.candidates_considered,
            "returned": len(documents),
            "dropped_below_floor": result.dropped_below_floor,
            "reranked": result.reranked,
            "abstained": result.abstained,
            "elapsed_ms": result.elapsed_ms,
            "fallback": fallback,
        },
        "trace": [
            trace_event(
                "retrieve",
                returned=len(documents),
                dropped=result.dropped_below_floor,
                abstained=result.abstained,
                ms=result.elapsed_ms,
                fallback=fallback,
            )
        ],
    }
