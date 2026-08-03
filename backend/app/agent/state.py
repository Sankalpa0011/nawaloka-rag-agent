"""Shared state for the agent graph.

Two things are deliberate here:

* SQL evidence and vector evidence live in **separate keys**. That is what makes
  the HYBRID fan-out safe - the two branches run concurrently and never write
  the same field, so no reducer or locking is needed.
* Everything needed to explain a turn (route, rationale, SQL executed, chunks
  with scores, which provider served it) is kept on the state rather than
  logged and discarded, because the UI renders it as a trace panel.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

Route = Literal["SQL", "VECTOR", "HYBRID", "SMALLTALK", "OUT_OF_SCOPE",
                "EMERGENCY", "REFUSE"]


@dataclass
class Citation:
    index: int
    label: str
    url: str = ""
    source: str = "web"        # "web" | "database"
    chunk_id: str = ""
    score: float | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "url": self.url,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4) if self.score is not None else None,
        }


class AgentState(TypedDict, total=False):
    # --- identity / input -------------------------------------------------
    thread_id: str
    user_query: str
    contextual_query: str          # coreference-resolved; used for routing
    history: list[list[str]]       # [[user, assistant], ...]

    # --- routing ----------------------------------------------------------
    route: str
    route_decision: dict           # full RouteDecision.to_dict()
    sql_intent: str
    vector_intent: str

    # --- guardrail --------------------------------------------------------
    guardrail: dict
    disclaimer_required: bool

    # --- evidence (written by independent branches) -----------------------
    sql_calls: Annotated[list[dict], operator.add]
    documents: list[dict]
    retrieval: dict

    # --- memory -----------------------------------------------------------
    memory: dict

    # --- FAQ / cache ------------------------------------------------------
    cache: dict

    # --- output -----------------------------------------------------------
    answer: str
    citations: list[dict]
    abstained: bool

    # --- observability ----------------------------------------------------
    trace: Annotated[list[dict], operator.add]
    llm_calls: Annotated[list[dict], operator.add]
    error: str
    elapsed_ms: int


def new_state(
    query: str,
    thread_id: str = "default",
    history: list[list[str]] | None = None,
) -> AgentState:
    return AgentState(
        thread_id=thread_id,
        user_query=query,
        contextual_query=query,
        history=history or [],
        sql_calls=[],
        documents=[],
        trace=[],
        llm_calls=[],
        citations=[],
    )


def trace_event(node: str, **fields: Any) -> dict:
    return {"node": node, **fields}
