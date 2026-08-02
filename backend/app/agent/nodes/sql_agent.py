"""SQL branch: choose a curated tool, or fall back to guarded Text-to-SQL.

Tool selection is itself an LLM call with schema-constrained output, but the
*space* it chooses from is small and every option is safe. That is the point of
the hybrid design: the model picks which question to ask, not how to ask it.
Only when no curated tool fits does generated SQL come into play, and that path
is still validated by the AST guard and executed read-only.
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.state import AgentState, trace_event
from app.config import settings
from app.llm.base import Msg
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.tools import sql_tools
from app.tools.sql_tools import SqlToolResult

log = get_logger(__name__)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": [
                "find_doctors",
                "get_doctor_schedule",
                "search_lab_tests",
                "search_health_packages",
                "list_specialties",
                "suggest_specialty",
                "text2sql",
            ],
        },
        "arguments": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["tool", "arguments"],
}

TOOL_PROMPT = """Pick ONE tool to answer a hospital database question. JSON only.

find_doctors            {specialty?, name?, department?, weekday?, max_fee?, include_schedule?}
                        doctor lists, fees, who is available on a day
get_doctor_schedule     {doctor}                 one named doctor's channeling times
search_lab_tests        {query?, category?, max_price?, fasting_required?, max_report_hours?}
                        lab test prices, fasting rules, report turnaround
search_health_packages  {query?, includes_service?, max_price?, target_audience?, category?}
                        packages; use includes_service to search INSIDE package contents
list_specialties        {}                       all specialties and departments
suggest_specialty       {complaint}              map a symptom to the right clinic
text2sql                {question}               ONLY if no tool above fits
                        (aggregates, counts, averages, superlatives across tables)

Every tool ALWAYS returns full details (fees, prices, schedules) for whatever it
finds. Arguments are FILTERS that narrow the search - they are never requests
for a field.

  max_fee / max_price / max_report_hours   a NUMBER: keep only results at or
                                           below it. Never true/false.
  fasting_required / include_schedule      true or false
  everything else                          text to match

So "what is Dr X's fee?" is {"name": "Dr X"} - NOT {"name": "Dr X",
"max_fee": true}. Asking for a fee is not a filter on the fee.

Rules:
- weekday must be a weekday name; the tool handles doctors recorded as 'Daily'
- "which package includes X" -> search_health_packages with includes_service: X
- omit arguments you do not need; never invent argument names
- prefer a curated tool; text2sql is the last resort"""


async def _choose_tool(question: str) -> tuple[str, dict, str, str]:
    """Returns (tool_name, arguments, reason, provider)."""
    data, response = await get_router().complete_json(
        [Msg("system", TOOL_PROMPT), Msg("user", f"Question: {question}")],
        TOOL_SCHEMA,
        tier="fast",
        max_tokens=400,
    )
    tool = str(data.get("tool", "")).strip()
    args = data.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    return tool, args, str(data.get("reason", ""))[:200], response.provider


#: Parameters that filter on a number. A boolean here is always a mistake.
_NUMERIC_ARGS = {"max_fee", "max_price", "max_report_hours", "limit"}
#: Parameters that are genuinely true/false switches.
_BOOLEAN_ARGS = {"include_schedule", "fasting_required"}


def _coerce_args(args: dict[str, Any], accepted: set[str]) -> tuple[dict, list[str]]:
    """Validate argument types, dropping values that cannot mean what they say.

    Models routinely pass a *flag* where a *filter value* is expected - the
    observed failure was ``max_fee: true`` for "what is Dr X's fee?", meaning
    "yes, I want the fee". Coerced naively that becomes ``fee <= 1.0``, which
    returns nothing and the agent then reports, with total confidence, that the
    doctor does not exist. Dropping the bad filter degrades to a broader query,
    which is recoverable; keeping it produces a wrong answer, which is not.
    """
    cleaned: dict[str, Any] = {}
    problems: list[str] = []

    for key, value in args.items():
        if key not in accepted:
            problems.append(f"{key} (not a parameter of this tool)")
            continue
        if value is None or value == "":
            continue

        if key in _NUMERIC_ARGS:
            if isinstance(value, bool):
                problems.append(f"{key}={value} (expected a number, not true/false)")
                continue
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                problems.append(f"{key}={value!r} (not a number)")
            continue

        if key in _BOOLEAN_ARGS:
            if isinstance(value, bool):
                cleaned[key] = value
            elif isinstance(value, str) and value.lower() in ("true", "false", "yes", "no"):
                cleaned[key] = value.lower() in ("true", "yes")
            else:
                problems.append(f"{key}={value!r} (expected true/false)")
            continue

        # Everything else is free text.
        cleaned[key] = value if isinstance(value, str) else str(value)

    return cleaned, problems


def _call_curated(tool: str, args: dict[str, Any]) -> SqlToolResult:
    fn = sql_tools.TOOLS.get(tool)
    if fn is None:
        raise KeyError(tool)

    import inspect

    accepted = set(inspect.signature(fn).parameters)
    cleaned, problems = _coerce_args(args, accepted)

    result = fn(**cleaned)
    if problems:
        log.warning("sql_agent.bad_arguments", tool=tool, problems=problems)
        result.notes.append("Ignored invalid argument(s): " + "; ".join(problems))
    return result


async def sql_agent_node(state: AgentState) -> dict:
    started = time.perf_counter()
    question = state.get("sql_intent") or state.get("contextual_query") or state["user_query"]

    try:
        tool, args, reason, provider = await _choose_tool(question)
    except Exception as exc:
        log.warning("sql_agent.tool_choice_failed", error=str(exc)[:200])
        # Without a choice, the broadest useful default is the doctor directory.
        tool, args, reason, provider = "find_doctors", {}, "tool choice failed", ""

    log.info("sql_agent.tool_chosen", tool=tool, args=args, provider=provider)

    if tool == "text2sql" or tool not in sql_tools.TOOLS:
        if not settings.enable_text2sql_fallback:
            result = sql_tools.list_specialties()
            result.notes.append("Text-to-SQL fallback is disabled by configuration.")
        else:
            from app.tools.text2sql import generate_and_run

            t2s = await generate_and_run(args.get("question") or question)
            result = t2s.to_tool_result()
            reason = reason or t2s.explanation
    else:
        try:
            result = _call_curated(tool, args)
        except Exception as exc:
            log.warning("sql_agent.tool_failed", tool=tool, error=str(exc)[:200])
            result = SqlToolResult(tool=tool, error=f"{type(exc).__name__}: {exc}")

    payload = result.to_dict()
    payload["chosen_because"] = reason
    payload["elapsed_ms"] = int((time.perf_counter() - started) * 1000)

    return {
        "sql_calls": [payload],
        "trace": [
            trace_event(
                "sql_agent",
                tool=result.tool,
                rows=result.row_count,
                ms=payload["elapsed_ms"],
                error=result.error,
            )
        ],
    }
