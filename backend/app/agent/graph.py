"""The agent graph.

    ingest -> guardrail -> route ->  SQL      -> sql_agent  ->\\
                    |                VECTOR   -> retrieve   -> synthesize -> finalize
                    |                HYBRID   -> both, concurrently ->/
                    |                              ^
                    |            SQL found nothing |
                    |                              |
                    +-[EMERGENCY|REFUSE]-> direct_answer -> finalize
                    +-[SMALLTALK|OUT_OF_SCOPE]-> direct_answer -> finalize

HYBRID fans out to sql_agent and retrieve concurrently; they write different
state keys so there's no reducer conflict. A SQL turn that ends with no rows
takes one pass through retrieve before synthesis (_wants_vector_fallback);
this fires at most once per turn so it cannot become a cycle.
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.agent import guardrail
from app.agent.nodes.contextualize import contextualize_node
from app.agent.nodes.faq import cache_write, faq_node
from app.agent.nodes.retrieve import retrieve_node
from app.agent.nodes.sql_agent import sql_agent_node, sql_found_nothing
from app.agent.nodes.synthesize import synthesize_node
from app.agent.router import route_query
from app.agent.signals import extract_signals
from app.agent.state import AgentState, new_state, trace_event
from app.config import settings
from app.logging_conf import get_logger
from app.memory import decisions, store
from app.memory.summarizer import should_summarise, summarise

log = get_logger(__name__)

SMALLTALK_REPLY = """Hello. I'm the Nawaloka Hospitals assistant.

I can help you with:
- **Doctors** - who practises here, their qualifications and consultation fees
- **Channeling** - which days and times a consultant is available
- **Lab tests** - prices, fasting requirements and report turnaround
- **Health packages** - what each one includes and what it costs
- **The hospital** - our centres, services and facilities

What would you like to know?"""

OUT_OF_SCOPE_REPLY = """I can only help with questions about Nawaloka Hospitals \
- our doctors, channeling times, lab tests, health check packages, and the \
services we offer.

Is there something about the hospital I can help you with?"""


async def ingest_node(state: AgentState) -> dict:
    return {
        "contextual_query": state.get("contextual_query") or state["user_query"],
        "trace": [trace_event("ingest", query=state["user_query"][:120])],
    }


async def guardrail_node(state: AgentState) -> dict:
    verdict = guardrail.check(state.get("contextual_query") or state["user_query"])
    payload = verdict.to_dict()
    # carry the response text through so direct_answer doesn't recheck the guardrail
    payload["response"] = verdict.response
    out: dict = {
        "guardrail": payload,
        "disclaimer_required": verdict.disclaimer_required,
        "trace": [trace_event("guardrail", verdict=verdict.verdict,
                              reason=verdict.reason)],
    }
    # route must be set here - an intercepted turn never reaches the classifier
    if verdict.short_circuits:
        out["route"] = verdict.verdict
    return out


async def route_node(state: AgentState) -> dict:
    decision = await route_query(
        state.get("contextual_query") or state["user_query"],
        history=state.get("history"),
    )
    return {
        "route": decision.route,
        "route_decision": decision.to_dict(),
        "sql_intent": decision.sql_intent,
        "vector_intent": decision.vector_intent,
        "disclaimer_required": state.get("disclaimer_required")
        or decision.guardrail.disclaimer_required,
        "trace": [
            trace_event("route", route=decision.route,
                        confidence=round(decision.confidence, 2),
                        overridden=decision.overridden,
                        provider=decision.provider, ms=decision.latency_ms)
        ],
    }


# greetings, refusals, emergencies - nothing to retrieve
async def direct_answer_node(state: AgentState) -> dict:
    route = state.get("route")
    guard = state.get("guardrail") or {}

    if route in ("EMERGENCY", "REFUSE"):
        answer = guard.get("response") or ""
        if not answer:
            # Only reachable if the route was set without check_safety running.
            answer = guardrail.check(
                state.get("contextual_query") or state["user_query"]
            ).response
    elif route == "SMALLTALK":
        answer = SMALLTALK_REPLY
    else:
        answer = OUT_OF_SCOPE_REPLY

    return {
        "answer": answer,
        "citations": [],
        "abstained": route in ("OUT_OF_SCOPE",),
        "trace": [trace_event("direct_answer", route=route,
                              guardrail=guard.get("verdict"))],
    }


async def finalize_node(state: AgentState) -> dict:
    # cache only after the answer has passed every check
    cached = cache_write(state)
    memory_saved = await _persist_memory(state)
    logged = _log_decision(state)
    return {
        "trace": [
            trace_event("finalize", route=state.get("route"), cached=cached,
                        memory=memory_saved, logged=logged,
                        citations=len(state.get("citations") or []))
        ]
    }


# Write the audit row for this turn. Done here rather than in route_node
# because finalize is the only node every path passes through.
def _log_decision(state: AgentState) -> bool:
    if not settings.enable_decision_log:
        return False

    route = state.get("route") or ""
    if not route:
        return False

    decision = state.get("route_decision") or {}
    raw_query = state.get("user_query", "")
    contextual = state.get("contextual_query") or ""

    if decision:
        signals = decision.get("signals") or {}
        llm_route = decision.get("llm_route", "")
        confidence = decision.get("confidence")
        rationale = decision.get("rationale", "")
        overridden = bool(decision.get("overridden"))
        latency_ms = decision.get("latency_ms")
        if overridden and decision.get("override_reason"):
            rationale = f"{rationale} [override: {decision['override_reason']}]"
    elif route == "FAQ":
        # a cache hit is a routing decision too - it decided not to route
        cache = state.get("cache") or {}
        signals = {"source": "faq", **cache}
        llm_route = ""
        confidence = cache.get("score")
        rationale = f"{cache.get('tier', 'faq')} fast-path hit, no classification run"
        overridden = False
        latency_ms = cache.get("elapsed_ms")
    else:
        # guardrail interception: EMERGENCY and REFUSE never see a model at all
        guard = state.get("guardrail") or {}
        signals = {"source": "guardrail", **{k: v for k, v in guard.items()
                                             if k != "response"}}
        llm_route = ""
        confidence = 1.0
        rationale = f"guardrail: {guard.get('reason', 'short-circuited')}"
        overridden = False
        latency_ms = None

    return decisions.log_decision(
        thread_id=state.get("thread_id", "default"),
        raw_query=raw_query,
        contextual_query=contextual if contextual != raw_query else "",
        final_route=route,
        signals=signals,
        llm_route=llm_route,
        confidence=confidence,
        rationale=rationale,
        overridden=overridden,
        latency_ms=latency_ms,
    )


# Persist the turn; failures here must never fail the turn itself.
async def _persist_memory(state: AgentState) -> bool:
    if not settings.enable_memory:
        return False

    thread_id = state.get("thread_id", "default")
    answer_text = state.get("answer") or ""
    if not answer_text.strip():
        return False

    try:
        store.append_turn(thread_id, state["user_query"], answer_text,
                          route=state.get("route", ""))

        thread = store.get_state(thread_id)
        thread.turn_count += 1

        # Roll focus slots forward from whatever this turn mentioned.
        signals = extract_signals(state.get("contextual_query") or state["user_query"])
        thread.focus_entities = store.update_focus(thread.focus_entities, signals.entities)

        if should_summarise(thread):
            history = store.get_history(
                thread_id, limit=settings.memory_summary_history_turns
            )
            summary = await summarise(history, thread.summary)
            if summary:
                thread.summary = summary

        store.save_state(thread)
        return True
    except Exception as exc:
        log.warning("finalize.memory_failed", error=str(exc)[:160])
        return False


# emergencies and refusals are already decided, so skip the FAQ
def _after_guardrail(state: AgentState) -> str:
    verdict = (state.get("guardrail") or {}).get("verdict")
    if verdict in ("EMERGENCY", "REFUSE"):
        return "direct_answer"
    return "faq"


# a cache hit is already an answer - no routing, retrieval or synthesis needed
def _after_faq(state: AgentState) -> str:
    return "finalize" if state.get("route") == "FAQ" else "classify"


#: Which node(s) each route fans out to; HYBRID runs both concurrently.
_BRANCHES: dict[str, list[str] | str] = {
    "EMERGENCY": "direct_answer",
    "REFUSE": "direct_answer",
    "SMALLTALK": "direct_answer",
    "OUT_OF_SCOPE": "direct_answer",
    "SQL": ["sql_agent"],
    "VECTOR": ["retrieve"],
    "HYBRID": ["sql_agent", "retrieve"],
}


def _branch(state: AgentState) -> list[str] | str:
    return _BRANCHES.get(state.get("route", "VECTOR"), ["retrieve"])


def _sql_exhausted(state: AgentState) -> bool:
    """Whether the SQL retry budget is used up. Attempt cap wins over last_error
    so a call that fails the same way every time doesn't loop forever."""
    if state.get("sql_attempts", 0) >= settings.sql_max_attempts:
        return True
    return not (state.get("last_error") or state.get("empty_with_filters"))


def _wants_vector_fallback(state: AgentState) -> bool:
    """A SQL turn that found zero rows gets one pass through the vector branch
    instead of dead-ending, since the DB not having a record doesn't mean the
    hospital doesn't offer it. Guarded to fire at most once per turn, and never
    for HYBRID (which already scheduled retrieve at classify)."""
    if not settings.enable_sql_vector_fallback:
        return False
    if state.get("route") != "SQL":
        return False
    if state.get("vector_fallback_used") or state.get("documents"):
        return False
    return sql_found_nothing(state)


def _after_sql(state: AgentState) -> str:
    """Retry sql_agent on error or an empty-with-filters result; once the retry
    budget is spent, fall back to the vector branch if nothing was found at
    all, otherwise go to synthesize."""
    if not _sql_exhausted(state):
        return "sql_agent"
    if _wants_vector_fallback(state):
        return "retrieve"
    return "synthesize"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("ingest", ingest_node)
    g.add_node("contextualize", contextualize_node)
    g.add_node("check_safety", guardrail_node)
    g.add_node("faq", faq_node)
    g.add_node("classify", route_node)
    g.add_node("sql_agent", sql_agent_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "contextualize")
    g.add_edge("contextualize", "check_safety")

    # FAQ sits after the guardrail (never cache an emergency) and before
    # routing (a hit costs no LLM call).
    g.add_conditional_edges(
        "check_safety",
        _after_guardrail,
        {"faq": "faq", "direct_answer": "direct_answer"},
    )
    g.add_conditional_edges(
        "faq", _after_faq, {"classify": "classify", "finalize": "finalize"}
    )

    g.add_conditional_edges(
        "classify",
        _branch,
        {"sql_agent": "sql_agent", "retrieve": "retrieve",
         "direct_answer": "direct_answer"},
    )

    # sql_agent loops back on itself while a call keeps failing; once exhausted
    # it falls through to retrieve (if nothing found) or straight to synthesize.
    g.add_conditional_edges(
        "sql_agent",
        _after_sql,
        {"sql_agent": "sql_agent", "retrieve": "retrieve",
         "synthesize": "synthesize"},
    )
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "finalize")
    g.add_edge("direct_answer", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.info("graph.compiled")
    return _graph


async def answer(
    query: str,
    thread_id: str = "default",
    history: list[list[str]] | None = None,
) -> AgentState:
    started = time.perf_counter()
    state = new_state(query, thread_id=thread_id, history=history)
    result = await get_graph().ainvoke(state)
    result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    log.info("graph.answered", route=result.get("route"),
             ms=result["elapsed_ms"], chars=len(result.get("answer") or ""))
    return result


def main() -> int:
    """Interactive CLI for testing the agent."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Ask the Nawaloka agent a question")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--trace", action="store_true", help="show the agent trace")
    args = parser.parse_args()

    async def ask(q: str, history: list[list[str]]) -> str:
        state = await answer(q, history=history)
        rd = state.get("route_decision") or {}
        print(f"\n\033[2m[{state.get('route')}] "
              f"conf={rd.get('confidence')} {state.get('elapsed_ms')}ms\033[0m")
        print(state.get("answer", ""))
        for c in state.get("citations") or []:
            print(f"\033[2m  [{c['index']}] {c['label']}  {c['url']}\033[0m")
        if args.trace:
            print("\n\033[2m--- trace ---")
            for ev in state.get("trace") or []:
                print("   ", ev)
            for call in state.get("sql_calls") or []:
                print(f"    SQL: {call.get('sql_executed', '')[:150]}")
            print("\033[0m")
        return state.get("answer", "")

    async def run() -> None:
        history: list[list[str]] = []
        if args.question:
            await ask(" ".join(args.question), history)
            return
        print("Nawaloka assistant. Ctrl-C to exit.\n")
        while True:
            try:
                q = input("\033[1myou >\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not q:
                continue
            reply = await ask(q, history)
            history.append([q, reply])

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
