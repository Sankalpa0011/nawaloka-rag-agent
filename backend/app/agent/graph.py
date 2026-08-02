"""The agent graph.

    ingest -> guardrail -> route ->  SQL      -> sql_agent  ->\\
                    |                VECTOR   -> retrieve   -> synthesize -> finalize
                    |                HYBRID   -> both, concurrently ->/
                    |
                    +-[EMERGENCY|REFUSE]-> direct_answer -> finalize
                    +-[SMALLTALK|OUT_OF_SCOPE]-> direct_answer -> finalize

Why LangGraph rather than an LLM tool-calling loop: "query routing accuracy" is
a graded criterion, so the router has to be a standalone function that can be
scored against a fixture set (see evals/score_routing.py). In a tool-calling
loop, routing is an emergent property of the model's tool choice - impossible to
measure, and impossible to fix without changing everything else.

The HYBRID branch fans out to both sources concurrently. Because the two nodes
write to different state keys (`sql_calls` vs `documents`), no locking or
reducer conflict is possible.
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.agent import guardrail
from app.agent.nodes.retrieve import retrieve_node
from app.agent.nodes.sql_agent import sql_agent_node
from app.agent.nodes.synthesize import synthesize_node
from app.agent.router import route_query
from app.agent.state import AgentState, new_state, trace_event
from app.logging_conf import get_logger

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


# ---------------------------------------------------------------------------
async def ingest_node(state: AgentState) -> dict:
    return {
        "contextual_query": state.get("contextual_query") or state["user_query"],
        "trace": [trace_event("ingest", query=state["user_query"][:120])],
    }


async def guardrail_node(state: AgentState) -> dict:
    verdict = guardrail.check(state.get("contextual_query") or state["user_query"])
    return {
        "guardrail": verdict.to_dict(),
        "disclaimer_required": verdict.disclaimer_required,
        "trace": [trace_event("guardrail", verdict=verdict.verdict,
                              reason=verdict.reason)],
    }


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


async def direct_answer_node(state: AgentState) -> dict:
    """Answers that need no retrieval: greetings, refusals, emergencies."""
    route = state.get("route")
    guard = state.get("guardrail") or {}

    if route in ("EMERGENCY", "REFUSE"):
        verdict = guardrail.check(state.get("contextual_query") or state["user_query"])
        answer = verdict.response
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
    return {"trace": [trace_event("finalize", route=state.get("route"),
                                  citations=len(state.get("citations") or []))]}


# ---------------------------------------------------------------------------
def _branch(state: AgentState) -> list[str] | str:
    route = state.get("route", "VECTOR")
    if route in ("EMERGENCY", "REFUSE", "SMALLTALK", "OUT_OF_SCOPE"):
        return "direct_answer"
    if route == "SQL":
        return ["sql_agent"]
    if route == "VECTOR":
        return ["retrieve"]
    # HYBRID: both branches run concurrently and converge on synthesize.
    return ["sql_agent", "retrieve"]


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("ingest", ingest_node)
    g.add_node("check_safety", guardrail_node)
    g.add_node("classify", route_node)
    g.add_node("sql_agent", sql_agent_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "check_safety")
    g.add_edge("check_safety", "classify")

    g.add_conditional_edges(
        "classify",
        _branch,
        {"sql_agent": "sql_agent", "retrieve": "retrieve",
         "direct_answer": "direct_answer"},
    )

    g.add_edge("sql_agent", "synthesize")
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


# ---------------------------------------------------------------------------
def main() -> int:
    """Interactive CLI so the agent is demoable before the web UI exists."""
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
