from __future__ import annotations

import re
import time

from app.agent.state import AgentState, trace_event
from app.config import settings
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.prompts.templates import CONTEXTUALIZE, render
from app.memory import store

log = get_logger(__name__)

# pronouns and demonstratives - they all need a referent from earlier
_PRONOUNS = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|it|its|that one|this one|"
    r"the same|those|these)\b",
    re.I,
)

# bare follow-ups: "what about Friday?", "and the price?", "how about tomorrow"
_ELLIPTICAL = re.compile(
    r"^\s*(and|but|so|also|what about|how about|ok|okay|then)\b"
    r"|^\s*(what|how) about\b",
    re.I,
)

# short questions are nearly always continuations, not new topics
_SHORT_TOKENS = 5
MAX_FOCUS_CHARS = settings.contextualize_max_focus_chars


# quick check: does this turn lean on something said earlier?
def needs_rewrite(query: str, has_history: bool) -> tuple[bool, str]:
    if not has_history:
        return False, "no prior turns"

    text = (query or "").strip()
    if not text:
        return False, "empty query"

    if _ELLIPTICAL.match(text):
        return True, "elliptical opener"
    if _PRONOUNS.search(text):
        return True, "contains a referring pronoun"
    # if len(text.split()) <= _SHORT_TOKENS and "?" in text:
    #     return True, "very short follow-up question"
    return False, "already self-contained"

def last_assistant_turn(history: list) -> str:
    for _user, assistant in reversed(history or []):
        text = (assistant or "").strip()
        if text:
            return text[: settings.contextualize_max_focus_chars]
    return ""


REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "changed": {"type": "boolean"},
    },
    "required": ["standalone_question"],
}


async def contextualize_node(state: AgentState) -> dict:
    started = time.perf_counter()
    query = state["user_query"]
    thread_id = state.get("thread_id", "default")

    if not settings.enable_memory:
        return {"contextual_query": query,
                "trace": [trace_event("contextualize", status="disabled")]}

    # use the caller's history if they gave us one, otherwise load the thread
    history = state.get("history") or []
    try:
        thread = store.get_state(thread_id)
        if not history:
            history = store.get_history(
                thread_id, limit=settings.contextualize_history_turns
            )
    except Exception as exc:
        log.warning("contextualize.memory_unavailable", error=str(exc)[:160])
        thread = store.ThreadState(thread_id=thread_id)

    should, reason = needs_rewrite(query, bool(history) or bool(thread.focus_entities))

    if not should:
        return {
            "contextual_query": query,
            "history": history,
            "memory": {
                "rewritten": False,
                "reason": reason,
                "focus_entities": thread.focus_entities,
                "summary": thread.summary,
                "turn_count": thread.turn_count,
            },
            "trace": [trace_event("contextualize", rewritten=False, reason=reason,
                                  ms=int((time.perf_counter() - started) * 1000))],
        }
    
    parts = []
    focus = thread.focus_block()
    if focus:
        parts.append(f"Known entities in focus:\n{focus}")
    parts.append(f"Q: {query}")

    rewritten = query
    provider = ""
    try:
        data, response = await get_router().complete_json(
            render(
                CONTEXTUALIZE,
                input="\n\n".join(parts),
                long_term_summary=thread.summary or "",
                immediate_focus=last_assistant_turn(history),
            ),
            REWRITE_SCHEMA,
            tier="fast",
            max_tokens=settings.contextualize_max_tokens,
        )
        candidate = str(data.get("standalone_question", "")).strip()
        provider = response.provider
        # sanity check - a rewrite this short has thrown the question away
        floor = max(
            settings.contextualize_min_rewrite_chars,
            len(query) // settings.contextualize_min_rewrite_ratio,
        )
        if candidate and len(candidate) >= floor:
            rewritten = candidate
    except Exception as exc:
        # never fatal, the original query always works
        log.warning("contextualize.rewrite_failed", error=str(exc)[:160])

    elapsed = int((time.perf_counter() - started) * 1000)
    changed = rewritten.strip().lower() != query.strip().lower()
    if changed:
        log.info("contextualize.rewritten", before=query[:60], after=rewritten[:80])

    return {
        "contextual_query": rewritten,
        "history": history,
        "memory": {
            "rewritten": changed,
            "reason": reason,
            "original_query": query,
            "focus_entities": thread.focus_entities,
            "summary": thread.summary,
            "turn_count": thread.turn_count,
            "provider": provider,
        },
        "trace": [
            trace_event("contextualize", rewritten=changed, reason=reason,
                        provider=provider, ms=elapsed)
        ],
    }
