"""Shared TypedDict state passed between agent graph nodes.

SQL and vector evidence live in separate keys so the HYBRID fan-out has no
reducer conflicts, and enough of each turn is kept on state to drive the UI trace panel.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

# Every route a turn can end up on, including the FAQ fast-path.
Route = Literal["SQL", "VECTOR", "HYBRID", "FAQ", "SMALLTALK", "OUT_OF_SCOPE",
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
    # identity / input 
    thread_id: str
    user_query: str
    contextual_query: str          # coreference-resolved; used for routing
    history: list[list[str]]       # [[user, assistant], ...]

    # routing 
    route: Route
    route_decision: dict           # full RouteDecision.to_dict()
    sql_intent: str
    vector_intent: str

    # guardrail 
    guardrail: dict
    disclaimer_required: bool

    #  evidence (written by independent branches) 
    sql_calls: Annotated[list[dict], operator.add]
    documents: list[dict]
    retrieval: dict

    # SQL self-correction 
    # Plain keys, not reducers: each retry pass overwrites; clearing last_error ends the cycle.
    sql_attempts: int
    last_error: str
    # Every error so far, shown to the model so it stops repeating itself.
    tool_feedback: list[str]
    # Set when a tool ran fine but matched nothing under applied filters; triggers one retry.
    empty_with_filters: str
    # Subject that retry failed to find, carried forward so a later broadened match isn't mistaken for a hit.
    unmatched_subject: str

    # zero-row outcome 
    zero_rows: bool
    #{"tool": str, "filters": dict, "subject": str}; empty when zero_rows is False.
    zero_row_context: dict
    # Set once the SQL -> vector fallback has run, so it fires at most once per turn.
    vector_fallback_used: bool

    # memory 
    memory: dict

    # FAQ / cache 
    cache: dict

    # output 
    answer: str
    citations: list[dict]
    abstained: bool

    # observability 
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
