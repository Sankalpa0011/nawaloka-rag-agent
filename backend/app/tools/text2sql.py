"""Guarded Text-to-SQL fallback.

The curated tools in ``sql_tools.py`` handle the common cases and are always
preferred. This exists for the long tail - "which test has the longest
turnaround under LKR 3000?", "how many doctors per department?" - where writing
a tool for every shape of question is not realistic.

Three things make a generated-SQL path acceptable here:

1. **It is pointed at the same trap-proof views** as the curated tools, via the
   schema card, so it inherits their correctness rather than re-learning the
   data hazards from scratch.
2. **Every statement passes the sqlglot AST guard** before execution, and runs
   as a role that cannot write. Generation being wrong is a quality problem,
   not a safety one.
3. **Failures are repaired with the error in hand**, twice, and then given up
   on honestly instead of being retried until something returns rows.

On an empty result the query is *relaxed once* rather than retried - blind
retry-on-empty is how agents end up inventing data that does not exist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import PROMPTS_DIR
from app.llm.base import Msg
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.tools.db import fetch
from app.tools.sql_guard import SqlGuardError, validate_sql
from app.tools.sql_tools import SqlToolResult

log = get_logger(__name__)

MAX_REPAIRS = 2

_SCHEMA_CARD: str | None = None


def schema_card() -> str:
    global _SCHEMA_CARD
    if _SCHEMA_CARD is None:
        _SCHEMA_CARD = (PROMPTS_DIR / "schema_card.md").read_text(encoding="utf-8")
    return _SCHEMA_CARD


SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["sql"],
}

SYSTEM = """You write a single PostgreSQL SELECT for a hospital assistant.

Return JSON: {"sql": "...", "explanation": "one sentence"}

Absolute rules:
- exactly ONE SELECT statement, no semicolon, no DDL, no writes
- only the relations in the schema below
- always include a LIMIT
- match user text with ILIKE '%...%' or similarity(col, '...') > 0.3, never =
- prefer the views; they resolve known data hazards the base tables do not

""" + "{schema}"


@dataclass
class Text2SqlResult:
    question: str
    sql: str = ""
    explanation: str = ""
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    attempts: int = 0
    repairs: list[str] = field(default_factory=list)
    relaxed: bool = False
    elapsed_ms: int = 0
    error: str | None = None
    provider: str = ""
    notes: list[str] = field(default_factory=list)

    def to_tool_result(self) -> SqlToolResult:
        return SqlToolResult(
            tool="text2sql",
            rows=self.rows,
            row_count=self.row_count,
            sql_executed=self.sql,
            elapsed_ms=self.elapsed_ms,
            notes=self.notes,
            error=self.error,
        )

    def to_dict(self) -> dict:
        return {
            "tool": "text2sql",
            "question": self.question,
            "sql": " ".join(self.sql.split()),
            "explanation": self.explanation,
            "row_count": self.row_count,
            "rows": self.rows[:50],
            "attempts": self.attempts,
            "repairs": self.repairs,
            "relaxed": self.relaxed,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "notes": self.notes,
        }


def _messages(question: str, failures: list[tuple[str, str]]) -> list[Msg]:
    msgs = [Msg("system", SYSTEM.replace("{schema}", schema_card()))]

    if failures:
        # Accumulate every failed attempt so the model does not loop on the
        # same mistake; showing only the latest error is not enough.
        lines = ["Previous attempts failed. Do not repeat these mistakes:"]
        for sql, err in failures:
            lines.append(f"\nSQL:\n{sql}\nError: {err}")
        msgs.append(Msg("user", "\n".join(lines)))

    msgs.append(Msg("user", f"Question: {question}"))
    return msgs


async def generate_and_run(question: str, allow_relax: bool = True) -> Text2SqlResult:
    started = time.perf_counter()
    result = Text2SqlResult(question=question)
    failures: list[tuple[str, str]] = []

    for attempt in range(1, MAX_REPAIRS + 2):
        result.attempts = attempt
        try:
            data, response = await get_router().complete_json(
                _messages(question, failures),
                SQL_SCHEMA,
                tier="strong",
                # A generous ceiling: a truncated response is unparseable JSON,
                # and 600 tokens was measured to cut off mid-statement on
                # multi-column SELECTs.
                max_tokens=1200,
            )
            result.provider = response.provider
        except Exception as exc:
            # Generation failures (truncation, malformed JSON, a provider
            # hiccup) are retryable like any other. Abandoning on the first one
            # wasted the repair budget that exists precisely for this.
            message = f"{type(exc).__name__}: {exc}"
            result.error = f"generation failed: {message}"
            result.repairs.append(message)
            failures.append(("<no SQL produced>", message))
            log.warning("text2sql.generation_failed", attempt=attempt,
                        error=str(exc)[:200])
            continue

        raw_sql = str(data.get("sql", "")).strip()
        result.explanation = str(data.get("explanation", ""))[:300]

        # --- validate -----------------------------------------------------
        try:
            guarded = validate_sql(raw_sql)
        except SqlGuardError as exc:
            failures.append((raw_sql, f"rejected by safety guard: {exc}"))
            result.repairs.append(str(exc))
            log.warning("text2sql.rejected", attempt=attempt, reason=str(exc)[:160])
            continue

        result.sql = guarded.sql

        # --- execute ------------------------------------------------------
        try:
            rows, _ = fetch(guarded.sql)
        except Exception as exc:
            message = str(exc).split("\n")[0][:300]
            failures.append((guarded.sql, message))
            result.repairs.append(message)
            log.warning("text2sql.execution_failed", attempt=attempt, error=message)
            continue

        result.rows = rows
        result.row_count = len(rows)
        result.error = None

        if rows:
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            log.info("text2sql.ok", attempts=attempt, rows=len(rows))
            return result

        # --- empty result: relax once, then stop --------------------------
        if allow_relax and not result.relaxed:
            result.relaxed = True
            relaxed = await _relax(question, guarded.sql)
            if relaxed:
                try:
                    guarded2 = validate_sql(relaxed)
                    rows2, _ = fetch(guarded2.sql)
                except Exception as exc:
                    log.warning("text2sql.relax_failed", error=str(exc)[:160])
                    rows2 = []
                    guarded2 = None
                if rows2 and guarded2:
                    result.sql = guarded2.sql
                    result.rows = rows2
                    result.row_count = len(rows2)
                    result.notes.append(
                        "IMPORTANT: nothing in the database matched the question "
                        "as asked. The query was then broadened, so the rows "
                        "below do NOT satisfy the original criteria - they are "
                        "only the nearest available records. Say plainly that "
                        "there is no match before offering them as context, and "
                        "never present them as if they answered the question."
                    )
                    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
                    return result

        result.notes.append(
            "The database returned no matching records for this question. Say so "
            "plainly - do not substitute information from anywhere else."
        )
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    # Exhausted repairs.
    if not result.error:
        result.error = f"could not produce valid SQL after {result.attempts} attempts"
    result.notes.append(
        "The database query could not be completed. Tell the user you were "
        "unable to look this up rather than guessing an answer."
    )
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.error("text2sql.gave_up", attempts=result.attempts, repairs=result.repairs)
    return result


async def _relax(question: str, sql: str) -> str | None:
    """Ask for the same query with its most restrictive filter dropped."""
    try:
        data, _ = await get_router().complete_json(
            [
                Msg("system", SYSTEM.replace("{schema}", schema_card())),
                Msg(
                    "user",
                    "This query returned no rows:\n\n"
                    f"{sql}\n\n"
                    f"Original question: {question}\n\n"
                    "Rewrite it with ONLY the single most restrictive filter removed, "
                    "so it returns closely related rows. Keep every other condition. "
                    "Do not invent tables or columns.",
                ),
            ],
            SQL_SCHEMA,
            tier="fast",
            max_tokens=400,
        )
        return str(data.get("sql", "")).strip() or None
    except Exception as exc:
        log.warning("text2sql.relax_generation_failed", error=str(exc)[:160])
        return None
