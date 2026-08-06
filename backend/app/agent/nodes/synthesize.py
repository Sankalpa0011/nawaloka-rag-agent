"""Builds the synthesis prompt from SQL/vector evidence and runs the answer LLM call.

DB and documents go in as separate labelled blocks; the database wins on
anything transactional, and tool `notes` are passed through verbatim as
mandatory context.
"""

from __future__ import annotations

import json
import re
import time

from app.agent.guardrail import MEDICAL_DISCLAIMER
from app.agent.state import AgentState, Citation, trace_event
from app.config import settings
from app.contact import HOTLINE
from app.llm.base import Msg
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.prompts.templates import SYNTHESIZE, render

log = get_logger(__name__)

MAX_ROWS_IN_PROMPT = settings.synthesis_max_rows_in_prompt
MAX_DOC_CHARS = settings.synthesis_max_doc_chars
SYNTHESIS_MAX_TOKENS = settings.synthesis_max_tokens
SYNTHESIS_RETRY_MAX_TOKENS = settings.synthesis_retry_max_tokens

BREVITY_NUDGE = (
    "\n\nYour previous attempt was cut off before it finished. Answer again, "
    "much shorter: one line per item, no preamble, no closing paragraph."
)


def _superseded(sql_calls: list[dict]) -> list[dict]:
    """Drop retry attempts that a later call replaced, keeping only useful ones for the prompt."""
    useful = [c for c in sql_calls if c.get("rows") and not c.get("error")]
    if useful:
        return useful

    clean = [c for c in sql_calls if not c.get("error")]
    return (clean or sql_calls)[-1:]


def _format_sql(sql_calls: list[dict]) -> str:
    if not sql_calls:
        return ""
    blocks: list[str] = []
    for call in _superseded(sql_calls):
        if call.get("error"):
            blocks.append(
                f"[tool {call.get('tool')}] FAILED: {call['error']}\n"
                "Tell the user you could not look this up."
            )
            continue

        rows = call.get("rows") or []
        shown = settings.synthesis_max_rows_in_prompt
        part = [f"[tool {call.get('tool')}] {len(rows)} row(s)"]
        if rows:
            part.append(json.dumps(rows[:shown], ensure_ascii=False,
                                   indent=1, default=str))
            if len(rows) > shown:
                part.append(f"... {len(rows) - shown} more row(s) not shown")
        else:
            part.append("NO ROWS - the database has no record matching this.")
        for note in call.get("notes") or []:
            part.append(f"note: {note}")
        blocks.append("\n".join(part))
    return "\n\n".join(blocks)


def _format_docs(documents: list[dict]) -> tuple[str, list[Citation]]:
    if not documents:
        return "", []
    blocks, citations = [], []
    for i, doc in enumerate(documents, start=1):
        citations.append(
            Citation(
                index=i,
                label=doc.get("citation") or doc.get("page_label", ""),
                url=doc.get("url", ""),
                source="web",
                chunk_id=doc.get("chunk_id", ""),
                score=doc.get("rerank_score"),
            )
        )
        body = (doc.get("body") or "")[: settings.synthesis_max_doc_chars]
        blocks.append(f"[{i}] {doc.get('citation', '')}\n{body}")
    return "\n\n".join(blocks), citations


_MARKER = re.compile(r"\[([\d,\s]+)\]")


def cited_indexes(answer: str) -> set[int]:
    found: set[int] = set()
    for marker in _MARKER.findall(answer):
        for number in marker.split(","):
            number = number.strip()
            if number.isdigit():
                found.add(int(number))
    return found


def build_messages(state: AgentState) -> tuple[list[Msg], list[Citation]]:
    sql_block = _format_sql(state.get("sql_calls") or [])
    doc_block, citations = _format_docs(state.get("documents") or [])

    parts: list[str] = []
    if state.get("history"):
        recent = state["history"][-settings.synthesis_history_turns:]
        limit = settings.synthesis_history_answer_chars
        convo = "\n".join(f"Patient: {u}\nAssistant: {a[:limit]}" for u, a in recent)
        parts.append(f"<conversation>\n{convo}\n</conversation>")

    if sql_block:
        parts.append(f"<database authoritative=\"true\">\n{sql_block}\n</database>")
    if doc_block:
        parts.append(f"<documents>\n{doc_block}\n</documents>")
    if not sql_block and not doc_block:
        parts.append(
            "<evidence>none - no database rows and no matching documents</evidence>"
        )

    if state.get("disclaimer_required"):
        parts.append(
            "<framing>The patient mentioned a symptom. Help them find the right "
            "clinic; do not suggest a cause or diagnosis.</framing>"
        )

    # DB empty but docs found: flag it so the model doesn't imply a priced record exists.
    if state.get("zero_rows") and doc_block:
        parts.append(
            "<framing>The database holds no record matching this, and the "
            "documents below describe what the hospital offers. Rule 11 "
            "applies.</framing>"
        )

    parts.append(f"Patient's question: {state.get('contextual_query') or state['user_query']}")
    return render(SYNTHESIZE, input="\n\n".join(parts), hotline=HOTLINE), citations


#: Numbers that are units rather than money, so they are not price claims.
_NON_PRICE = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(hours?|hrs?|am|pm|years?|minutes?)\b", re.I)
_MONEY = re.compile(r"(?:LKR|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?)", re.I)


def _unsupported_amounts(answer: str, state: AgentState) -> list[str]:
    """Money figures in the answer that don't appear anywhere in the evidence."""
    evidence = json.dumps(state.get("sql_calls") or [], default=str)
    evidence += " ".join((d.get("body") or "") for d in state.get("documents") or [])
    evidence_digits = re.sub(r"[,\s]", "", evidence)

    unsupported = []
    for raw in _MONEY.findall(answer):
        if _NON_PRICE.search(raw):
            continue
        normalised = raw.replace(",", "")
        stem = normalised.split(".")[0]
        if stem and stem not in evidence_digits:
            unsupported.append(raw)
    return unsupported


def _with_brevity_nudge(messages: list[Msg]) -> list[Msg]:
    """Copy of the prompt with the brevity nudge appended to the last user message."""
    out = [Msg(m.role, m.content) for m in messages]
    for msg in reversed(out):
        if msg.role == "user":
            msg.content += BREVITY_NUDGE
            break
    return out


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_TRAILING_WORD = re.compile(r"(\w+)$")

_ABBREVIATIONS = {"dr", "prof", "mr", "mrs", "ms", "st", "no", "vs", "approx"}


def _last_sentence_end(body: str) -> int:
    """Index just past the last real sentence terminator, or -1 if none."""
    end = -1
    for match in _SENTENCE_END.finditer(body):
        word = _TRAILING_WORD.search(body[: match.start()])
        if word and word.group(1).lower() in _ABBREVIATIONS:
            continue
        end = match.end()
    return end


def _trim_to_boundary(answer: str) -> str:
    """Cut a still-truncated answer back to its last complete line/sentence; "" if none complete."""
    body = answer.rstrip()
    if not body:
        return body
    lines = body.split("\n")
    if len(lines) > 1:
        # Last line is partial since generation stopped mid-token.
        return "\n".join(lines[:-1]).rstrip()
    end = _last_sentence_end(body)
    return body[:end] if end > 0 else ""


def _has_evidence(state: AgentState) -> bool:
    """Whether there's anything for the model to work from (a zero-row result with notes still counts)."""
    if state.get("documents"):
        return True
    for call in state.get("sql_calls") or []:
        if call.get("rows") or call.get("notes") or call.get("error"):
            return True
    return False


async def synthesize_node(state: AgentState) -> dict:
    started = time.perf_counter()
    messages, citations = build_messages(state)

    rows_found = any((c.get("rows") for c in state.get("sql_calls") or []))
    # Used for cache eligibility only; doesn't gate whether the model runs.
    abstained = not rows_found and not state.get("documents")

    if not _has_evidence(state):
        elapsed = int((time.perf_counter() - started) * 1000)
        log.info("synthesize.fast_exit", ms=elapsed)
        return {
            "answer": f"I couldn't find specific information about that in my current records. Please contact Nawaloka Hospitals on {HOTLINE} for assistance.",
            "citations": [],
            "abstained": True,
            "llm_calls": [],
            "trace": [trace_event("synthesize", ms=elapsed, status="fast_exit_no_evidence")]
        }

    flags: list[str] = []
    try:
        router = get_router()
        response = await router.complete(
            messages,
            tier="strong",
            max_tokens=settings.synthesis_max_tokens,
            temperature=settings.synthesis_temperature,
        )

        if response.truncated:
            # Retry once with a shorter target rather than looping.
            log.warning("synthesize.truncated", attempt=1,
                        provider=response.provider, model=response.model,
                        output_tokens=response.usage.output_tokens,
                        max_tokens=settings.synthesis_max_tokens)
            flags.append("truncated_retried")
            response = await router.complete(
                _with_brevity_nudge(messages),
                tier="strong",
                max_tokens=settings.synthesis_retry_max_tokens,
                temperature=settings.synthesis_temperature,
            )

        answer = response.text.strip()
        llm_trace = response.to_trace()

        if response.truncated:
            log.warning("synthesize.truncated_after_retry",
                        provider=response.provider, model=response.model,
                        output_tokens=response.usage.output_tokens,
                        max_tokens=settings.synthesis_retry_max_tokens)
            flags.append("truncated")
            kept = _trim_to_boundary(answer)
            answer = (
                f"{kept}\n\n_This answer was cut short. Please call {HOTLINE} "
                "for the rest._"
                if kept
                else "I could not finish that answer. Please call Nawaloka "
                     f"Hospitals on {HOTLINE} and they will go through it with you."
            )
    except Exception as exc:
        log.error("synthesize.failed", error=str(exc)[:200])
        return {
            "answer": (
                "I'm having trouble reaching my language service right now. "
                "Please try again in a moment, or call Nawaloka Hospitals on "
                f"{HOTLINE}."
            ),
            "citations": [],
            "abstained": True,
            "error": str(exc)[:300],
            "trace": [trace_event("synthesize", error=str(exc)[:160])],
        }

    unsupported = _unsupported_amounts(answer, state)
    if unsupported:
        flags.append(f"unsupported_amounts:{','.join(unsupported)}")
        log.warning("synthesize.unsupported_amount", amounts=unsupported)
        answer += (
            f"\n\n_Please confirm current prices with the hospital on {HOTLINE}._"
        )

    if state.get("disclaimer_required") and MEDICAL_DISCLAIMER not in answer:
        answer = f"{answer}\n\n{MEDICAL_DISCLAIMER}"

    # Only cite documents the answer actually references.
    used = cited_indexes(answer)
    kept = [c.to_dict() for c in citations if c.index in used] or [
        c.to_dict() for c in citations[: settings.synthesis_fallback_citations]
    ]

    elapsed = int((time.perf_counter() - started) * 1000)
    log.info("synthesize.done", chars=len(answer), citations=len(kept),
             flags=flags, ms=elapsed)

    return {
        "answer": answer,
        "citations": kept,
        "abstained": abstained,
        "llm_calls": [llm_trace],
        "trace": [
            trace_event("synthesize", ms=elapsed, citations=len(kept),
                        provider=llm_trace["provider"], flags=flags)
        ],
    }
