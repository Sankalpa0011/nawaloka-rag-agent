"""SQL branch: pick a curated tool via LLM, or fall back to guarded Text-to-SQL."""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

from app.agent.state import AgentState, trace_event
from app.config import settings
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.prompts.templates import TOOL_CHOICE, render
from pydantic import ValidationError

from app.tools import schemas as tool_schemas
from app.tools import sql_tools
from app.tools.schemas import TOOL_ARGS
from app.tools.sql_tools import SqlToolResult

log = get_logger(__name__)

TEXT2SQL = "text2sql"

TOOL_NAMES: tuple[str, ...] = (*tool_schemas.tool_names(), TEXT2SQL)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": list(TOOL_NAMES)},
        "arguments": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["tool", "arguments"],
}


@lru_cache(maxsize=1)
def _specialty_roster() -> str:
    try:
        from app.tools.db import fetch

        rows, _ = fetch("SELECT name FROM specialties ORDER BY name")
        return ", ".join(str(r["name"]) for r in rows)
    except Exception as exc:
        log.warning("sql_agent.specialty_roster_unavailable", error=str(exc)[:160])
        return ""


async def _choose_tool(
    question: str, feedback: list[str] | None = None
) -> tuple[str, dict, str, str]:

    user = f"Question: {question}"
    if feedback:
        problems = "\n".join(f"- {f}" for f in feedback)
        user = (
            f"{user}\n\nYour previous attempt(s) failed:\n{problems}\n\n"
            "Fix the call. Do not repeat the same mistake."
        )

    data, response = await get_router().complete_json(
        render(TOOL_CHOICE, roster=_specialty_roster(), input=user),
        TOOL_SCHEMA,
        tier="fast",
        max_tokens=settings.sql_tool_choice_max_tokens,
    )
    tool = str(data.get("tool", "")).strip()
    args = data.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    reason = str(data.get("reason", ""))[: settings.sql_tool_reason_max_chars]
    return tool, args, reason, response.provider


def _validate_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:

    schema = TOOL_ARGS.get(tool)
    if schema is None:
        raise KeyError(tool)

    supplied = {}
    for key, value in args.items():
        if value is None or value == "":
            continue
        supplied[key] = value

    validated = schema(**supplied)
    return validated.model_dump(exclude_none=True)


def _validation_detail(exc: ValidationError) -> str:
    """One readable line per rejected argument, for the model to act on."""
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", ())) or "argument"
        parts.append(f"{field}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _applied(args: dict[str, Any]) -> dict[str, Any]:
    """Arguments that actually narrowed the search."""
    return {k: v for k, v in args.items() if v not in (None, "", False)}


def _recognised(value: str) -> bool:
    """Does the gazetteer or the symptom ontology know what this phrase refers
    to? Returns False on outage rather than raising."""
    text = (value or "").strip()
    if not text:
        return False

    try:
        from app.agent.signals import extract_signals

        if extract_signals(text).entities:
            return True
    except Exception as exc:
        log.warning("sql_agent.gazetteer_unavailable", error=str(exc)[:160])

    try:
        from app.retrieval.ontology import match_specialties

        return bool(match_specialties(text))
    except Exception as exc:
        log.warning("sql_agent.ontology_unavailable", error=str(exc)[:160])
        return False


# Which argument carries the doctor's name on each doctor tool.
_DOCTOR_NAME_ARG = {"find_doctors": "name", "get_doctor_schedule": "doctor"}


def _restore_named_doctor(tool: str, args: dict[str, Any], question: str) -> dict[str, Any]:
    field = _DOCTOR_NAME_ARG.get(tool)
    if not field or args.get(field):
        return args

    try:
        from app.agent.signals import extract_signals

        named = extract_signals(question).entities.get("doctor") or []
    except Exception as exc:
        log.warning("sql_agent.gazetteer_unavailable", error=str(exc)[:160])
        return args

    if len(named) != 1:
        return args

    args = dict(args)
    args[field] = named[0]
    log.info("sql_agent.restored_doctor_name", tool=tool, doctor=named[0])
    return args


def _names_something_specific(value: str) -> bool:
    """Whether this phrase points at a particular record vs. a whole topic."""
    text = (value or "").strip().lower()
    if not text:
        return False

    try:
        from app.agent import signals

        generic = (
            signals.build_gazetteer().shared_words
            | signals.SQL_KEYWORDS
            | signals.RECORD_NOUNS
            | signals.VECTOR_KEYWORDS
            | signals._STOPWORDS
        )
    except Exception as exc:
        log.warning("sql_agent.vocabulary_unavailable", error=str(exc)[:160])
        return True

    for word in re.findall(r"[a-z][a-z'-]{2,}", text):
        if word in generic or word.rstrip("s") in generic or f"{word}s" in generic:
            continue
        return True
    return False


# which argument (if any) names the thing the patient asked about
def _complaint_subject(tool: str, cleaned: dict[str, Any]) -> str | None:
    for field in tool_schemas.fields_with_role(tool, "subject"):
        if cleaned.get(field):
            return str(cleaned[field])
    return None


def _only_subject_args(tool: str, cleaned: dict[str, Any]) -> bool:
    """Whether the empty result is explained purely by the named subject, with no other constraint applied."""
    constraints = set(tool_schemas.fields_with_role(tool, "constraint"))
    return not (constraints & set(_applied(cleaned)))


def _substituted_note(subject: str) -> str:
    return (
        f"IMPORTANT: nothing in the database matched '{subject}'. The rows below are "
        "what is available instead, NOT an answer to what was asked. Tell the patient "
        "plainly first that Nawaloka Hospitals does not have what they asked for, then "
        "offer these as the closest alternatives. Never present them as if they matched."
    )


def _retry_as_complaint(subject: str) -> SqlToolResult | None:
    """Re-read an unmatched value as a symptom via the ontology; None if nothing matches."""
    try:
        from app.retrieval.ontology import match_specialties

        matched = match_specialties(subject)
    except Exception as exc:

        log.warning("sql_agent.ontology_unavailable", error=str(exc)[:160])
        return None

    if not matched:
        log.info("sql_agent.no_ontology_match", subject=subject[:60])
        return None

    result = sql_tools.suggest_specialty(subject, matched=matched)
    result.notes.insert(
        0,
        f"'{subject}' is not a specialty in the database, so it was re-read as a "
        "description of the problem. Answer the question that was asked - do not "
        "mention the failed lookup, and do not present this as a diagnosis.",
    )
    log.info("sql_agent.retried_as_complaint", subject=subject[:60],
             specialties=matched, rows=result.row_count)
    return result


def _call_curated(tool: str, args: dict[str, Any]) -> SqlToolResult:
    fn = sql_tools.TOOLS.get(tool)
    if fn is None:
        raise KeyError(tool)

    try:
        cleaned = _validate_args(tool, args)
    except ValidationError as exc:
        symptom = tool_schemas.complaint_value(exc)
        if symptom is None:
            raise
        recovered = _retry_as_complaint(symptom)
        if recovered is None:
            raise
        return recovered

    result = fn(**cleaned)

    result.requested_filters = dict(cleaned)

    subject = _complaint_subject(tool, cleaned)
    if (
        not result.error
        and result.row_count == 0
        and subject
        and _only_subject_args(tool, cleaned)
    ):
        recovered = _retry_as_complaint(subject)
        if recovered is not None:
            result = recovered

    return result


MAX_SQL_ATTEMPTS = settings.sql_max_attempts


def _empty_with_filters(
    result: SqlToolResult, tool: str, args: dict[str, Any], attempt: int
) -> tuple[str, str]:
    """Whether an empty filtered result is worth retrying broader; returns (instruction, subject)."""
    if (
        result.error
        or result.row_count
        or result.relaxed
        or attempt > settings.sql_broaden_max_attempt
    ):
        return "", ""

    applied = {k: v for k, v in _applied(args).items() if k != "limit"}
    if not applied:
        return "", ""

    subject = next(
        (applied[f] for f in tool_schemas.fields_with_role(tool, "subject")
         if f in applied),
        None,
    )
    if subject is None:
        values = list(applied.values())
        subject = next((v for v in values if isinstance(v, str)), values[0])
    subject = str(subject)

    if _names_something_specific(subject) and not _recognised(subject):
        log.info("sql_agent.no_broadening", tool=tool, subject=subject[:60],
                 reason="names a specific record unknown to the database and the ontology")
        return "", subject


    if tool_schemas.has_required_args(tool):
        needed = ", ".join(tool_schemas.required_fields(tool))
        retry = (f"a different tool - {tool} cannot run without {needed}, and the "
                 f"patient named no {needed}")
    else:
        retry = f"{tool} with NO arguments at all"
    instruction = (
        f"{tool}({applied}) ran fine and matched nothing, so those filters are "
        f"wrong. The patient did not name a specific record - they named a topic. "
        f"Do NOT justify the filters and do NOT adjust them. Call {retry}, and let "
        "the full list come back."
    )
    return instruction, subject


async def sql_agent_node(state: AgentState) -> dict:
    """Pick a tool and run it; failures come back as state so the graph can retry."""
    started = time.perf_counter()
    question = state.get("sql_intent") or state.get("contextual_query") or state["user_query"]
    attempt = state.get("sql_attempts", 0) + 1
    feedback = list(state.get("tool_feedback") or [])

    try:
        tool, args, reason, provider = await _choose_tool(question, feedback)
    except Exception as exc:
        log.warning("sql_agent.tool_choice_failed", error=str(exc)[:200])
        tool = "text2sql"
        args = {"question": question}
        reason = "dynamic text2sql fallback due to tool choice failure"
        provider = "system-dynamic"
        
    # The rewritten intent can drop the name the patient actually typed, so look
    # for it in both. Entity names come back de-duplicated, so overlap is fine.
    args = _restore_named_doctor(tool, args, f"{question} {state['user_query']}")

    log.info("sql_agent.tool_chosen", tool=tool, args=args, provider=provider,
             attempt=attempt)

    if tool == "text2sql" or tool not in sql_tools.TOOLS:
        if not settings.enable_text2sql_fallback:
            result = sql_tools.list_specialties()
            result.notes.append("Text-to-SQL fallback is disabled by configuration.")
        else:
            from app.tools.text2sql import generate_and_run

            t2s = await generate_and_run(args.get("question") or question,
                                         feedback=feedback)
            result = t2s.to_tool_result()
            reason = reason or t2s.explanation
    else:
        try:
            result = _call_curated(tool, args)
        except ValidationError as exc:
            detail = _validation_detail(exc)
            log.warning("sql_agent.invalid_arguments", tool=tool, detail=detail,
                        attempt=attempt)
            result = SqlToolResult(
                tool=tool, error=f"{tool}({args}) rejected - {detail}"
            )
        except Exception as exc:
            log.warning("sql_agent.tool_failed", tool=tool, error=str(exc)[:200])
            result = SqlToolResult(tool=tool, error=f"{type(exc).__name__}: {exc}")


    carried = state.get("unmatched_subject") or ""
    if carried and result.rows and not result.error:
        result.relaxed = True
        result.notes.insert(0, _substituted_note(carried))

    payload = result.to_dict()
    payload["chosen_because"] = reason
    payload["attempt"] = attempt
    payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)

    instruction, subject = _empty_with_filters(result, tool, args, attempt)

    zero_rows = bool(not result.error and result.row_count == 0 and not result.relaxed)

    out: dict = {
        "sql_calls": [payload],
        "sql_attempts": attempt,
        "last_error": result.error or "",
        "empty_with_filters": instruction,
        # Set for the retry to pick up, cleared once this pass has used it.
        "unmatched_subject": subject,
        "zero_rows": zero_rows,
        "zero_row_context": {
            "tool": result.tool,
            "filters": dict(result.requested_filters),
            "subject": subject
            or _complaint_subject(result.tool, result.requested_filters)
            or carried,
        } if zero_rows else {},
    }

    if instruction:
        feedback.append(instruction)
        out["tool_feedback"] = feedback
        log.info("sql_agent.empty_retrying", tool=tool, args=args, subject=subject[:60])

    if result.error:
        feedback.append(result.error)
        out["tool_feedback"] = feedback
        exhausted = attempt >= settings.sql_max_attempts
        if exhausted:
            result.notes.append(
                "The database lookup failed and could not be corrected. Tell the "
                "user you were unable to look this up - do not guess an answer."
            )
            payload["notes"] = result.notes
        log.warning("sql_agent.failed", attempt=attempt, exhausted=exhausted,
                    error=result.error[:160])

    out["trace"] = [
        trace_event(
            "sql_agent",
            tool=result.tool,
            rows=result.row_count,
            attempt=attempt,
            ms=payload["elapsed_ms"],
            error=result.error,
            zero_rows=zero_rows,
        )
    ]
    return out


def sql_found_nothing(state: AgentState) -> bool:
    """Whether every SQL call this turn (not just the last) came back with no rows."""
    calls = state.get("sql_calls") or []
    return bool(calls) and not any(c.get("rows") for c in calls)
