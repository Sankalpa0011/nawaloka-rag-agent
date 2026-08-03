"""Answer synthesis.

Evidence from the two sources is presented to the model in clearly separated,
labelled blocks, with an explicit precedence rule: **the database wins on
anything transactional**. That is not a stylistic preference - the website
advertises health packages that do not exist in the relational database (e.g.
"Premier Under 40 Men", LKR 45,000), so a model left to blend the two will
quote a price the hospital's system of record cannot honour.

Tool `notes` are injected verbatim and marked as mandatory. They are how the
deterministic layer tells the model things it cannot infer from rows alone -
"this doctor is recorded as 'Daily' and therefore includes Tuesday", "these two
doctors have no published sessions, do not imply availability".
"""

from __future__ import annotations

import json
import re
import time

from app.agent.guardrail import MEDICAL_DISCLAIMER
from app.agent.state import AgentState, Citation, trace_event
from app.llm.base import Msg
from app.llm.router import get_router
from app.logging_conf import get_logger

log = get_logger(__name__)

MAX_ROWS_IN_PROMPT = 25
MAX_DOC_CHARS = 1100

SYSTEM = """You are the assistant for Nawaloka Hospitals in Colombo, Sri Lanka. \
Answer patients clearly and warmly, in a few short sentences or a compact list.

EVIDENCE RULES - these override your own knowledge:
1. Use ONLY the evidence provided. If it does not answer the question, say so.
2. <database> is the hospital's system of record. For any fee, price, schedule,
   room, package or test detail, the database is authoritative. If <documents>
   disagree with it, the database is right and the document is stale.
3. Never state a price, fee or time that is not present in the evidence.
4. Text under "notes" is verified fact from the database layer. Follow it
   exactly and reflect it in your answer.
5. Quote test preparation instructions (fasting, timing) word for word. Never
   paraphrase them.
6. Cite documents inline as [1], [2] using the numbers given. Do not cite the
   database.
7. Amounts are Sri Lankan Rupees. Write them as "LKR 3,500".
8. If nothing supports an answer, say you do not have that information and
   suggest calling 0115 777 777. Do not guess.

FORMATTING:
- Keep one item per bullet. A doctor's name, time and room belong on the SAME
  line, never split into separate bullets - "**Dr. X** (Cardiology) - Monday
  2:00 PM to 5:00 PM, Room 108".
- Write times the way people say them: "2:00 PM to 5:00 PM", not "14:00:00".
- Bold only names and amounts. Do not bold labels like "Time:" or "Location:".
- No greeting unless the patient greeted you first. Answer the question.
- Keep it short. A list of three doctors needs three lines, not twelve.

Never diagnose, never recommend medication or dosage, never interpret a \
patient's own test results. Pointing someone to the right specialty is fine."""


def _format_sql(sql_calls: list[dict]) -> str:
    if not sql_calls:
        return ""
    blocks: list[str] = []
    for call in sql_calls:
        if call.get("error"):
            blocks.append(
                f"[tool {call.get('tool')}] FAILED: {call['error']}\n"
                "Tell the user you could not look this up."
            )
            continue

        rows = call.get("rows") or []
        part = [f"[tool {call.get('tool')}] {len(rows)} row(s)"]
        if rows:
            part.append(json.dumps(rows[:MAX_ROWS_IN_PROMPT], ensure_ascii=False,
                                   indent=1, default=str))
            if len(rows) > MAX_ROWS_IN_PROMPT:
                part.append(f"... {len(rows) - MAX_ROWS_IN_PROMPT} more row(s) not shown")
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
        body = (doc.get("body") or "")[:MAX_DOC_CHARS]
        blocks.append(f"[{i}] {doc.get('citation', '')}\n{body}")
    return "\n\n".join(blocks), citations


def build_messages(state: AgentState) -> tuple[list[Msg], list[Citation]]:
    sql_block = _format_sql(state.get("sql_calls") or [])
    doc_block, citations = _format_docs(state.get("documents") or [])

    parts: list[str] = []
    if state.get("history"):
        recent = state["history"][-2:]
        convo = "\n".join(f"Patient: {u}\nAssistant: {a[:200]}" for u, a in recent)
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

    parts.append(f"Patient's question: {state.get('contextual_query') or state['user_query']}")
    return [Msg("system", SYSTEM), Msg("user", "\n\n".join(parts))], citations


#: Numbers that are units rather than money, so they are not price claims.
_NON_PRICE = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(hours?|hrs?|am|pm|years?|minutes?)\b", re.I)
_MONEY = re.compile(r"(?:LKR|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?)", re.I)


def _unsupported_amounts(answer: str, state: AgentState) -> list[str]:
    """Money in the answer that appears in no piece of evidence.

    Hallucinated prices are the most damaging failure available in this domain,
    and they are cheap to detect: every legitimate amount must appear somewhere
    in the rows or documents that produced the answer.
    """
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


async def synthesize_node(state: AgentState) -> dict:
    started = time.perf_counter()
    messages, citations = build_messages(state)

    has_evidence = bool(state.get("sql_calls")) or bool(state.get("documents"))
    rows_found = any((c.get("rows") for c in state.get("sql_calls") or []))
    abstained = not has_evidence or (not rows_found and not state.get("documents"))

    try:
        response = await get_router().complete(
            messages,
            tier="strong",
            # Generous, because the Gemini 3.x flash models reason internally
            # and those thinking tokens are charged against max_output_tokens.
            # At 800 the reasoning consumed the budget and answers arrived
            # truncated mid-sentence - a list of 11 doctors stopped at the
            # second one. The installed google-genai (0.5.0) has no
            # thinking_config to turn it off, so the budget has to absorb it.
            max_tokens=2500,
            temperature=0.2,
        )
        answer = response.text.strip()
        llm_trace = response.to_trace()
    except Exception as exc:
        log.error("synthesize.failed", error=str(exc)[:200])
        return {
            "answer": (
                "I'm having trouble reaching my language service right now. "
                "Please try again in a moment, or call Nawaloka Hospitals on "
                "0115 777 777."
            ),
            "citations": [],
            "abstained": True,
            "error": str(exc)[:300],
            "trace": [trace_event("synthesize", error=str(exc)[:160])],
        }

    # --- post-checks ------------------------------------------------------
    flags: list[str] = []
    unsupported = _unsupported_amounts(answer, state)
    if unsupported:
        flags.append(f"unsupported_amounts:{','.join(unsupported)}")
        log.warning("synthesize.unsupported_amount", amounts=unsupported)
        answer += (
            "\n\n_Please confirm current prices with the hospital on 0115 777 777._"
        )

    if state.get("disclaimer_required") and MEDICAL_DISCLAIMER not in answer:
        answer = f"{answer}\n\n{MEDICAL_DISCLAIMER}"

    # Only cite documents the answer actually references.
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    kept = [c.to_dict() for c in citations if c.index in used] or [
        c.to_dict() for c in citations[:3]
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
