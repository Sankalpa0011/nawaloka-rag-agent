"""Text-to-SQL fallback for the long tail sql_tools.py doesn't cover.

Generated SQL is pointed at the same views (inherits the data-hazard fixes),
runs through the sqlglot AST guard as a non-writing role, and gets one attempt
per call - the retry loop lives in the graph (sql_agent). Empty results are
relaxed once, never retried blindly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import PROMPTS_DIR, settings
from app.llm.base import Msg
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.prompts.schema import table_ddl
from app.prompts.templates import TEXT2SQL, render
from app.tools.db import fetch
from app.tools.sql_guard import SqlGuardError, validate_sql
from app.tools.sql_tools import SqlToolResult

log = get_logger(__name__)


# Two halves: the column list comes from the database, the hazard notes come
# from the file. Neither can produce the other - introspection doesn't know that
# an inner join drops two doctors, and a markdown file can't notice a rename.
@lru_cache(maxsize=1)
def schema_card() -> str:
    notes = (PROMPTS_DIR / "schema_card.md").read_text(encoding="utf-8")
    return f"{table_ddl()}\n\n{notes}"


SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["sql"],
}


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
            "rows": self.rows[: settings.text2sql_max_rows_in_payload],
            "attempts": self.attempts,
            "repairs": self.repairs,
            "relaxed": self.relaxed,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "notes": self.notes,
        }


def _messages(question: str, failures: list[tuple[str, str]]) -> list[Msg]:
    msgs = render(TEXT2SQL, schema=schema_card(), input=f"Question: {question}")

    if failures:
        # Show every failed attempt so the model doesn't repeat the same mistake.
        lines = ["Previous attempts failed. Do not repeat these mistakes:"]
        for sql, err in failures:
            lines.append(f"\nSQL:\n{sql}\nError: {err}")
        msgs.insert(1, Msg("user", "\n".join(lines)))

    return msgs


async def generate_and_run(
    question: str,
    allow_relax: bool = True,
    feedback: list[str] | None = None,
) -> Text2SqlResult:
    """Generate one SQL statement, guard it, run it - a single attempt (retries
    live in the graph). `feedback` is what the graph collected on earlier passes.
    Relaxing an empty result stays here, since that's not error recovery.
    """
    started = time.perf_counter()
    result = Text2SqlResult(question=question)
    failures: list[tuple[str, str]] = []
    for prior in feedback or []:
        failures.append(("<earlier attempt this turn>", prior))

    for attempt in range(1, 2):
        result.attempts = attempt
        try:
            data, response = await get_router().complete_json(
                _messages(question, failures),
                SQL_SCHEMA,
                tier="strong",
                # Generous ceiling - a truncated response is unparseable JSON.
                max_tokens=settings.text2sql_max_tokens,
            )
            result.provider = response.provider
        except Exception as exc:
            # Truncation, bad JSON, a provider hiccup: retryable like any other failure.
            message = f"{type(exc).__name__}: {exc}"
            result.error = f"generation failed: {message}"
            result.repairs.append(message)
            failures.append(("<no SQL produced>", message))
            log.warning("text2sql.generation_failed", attempt=attempt,
                        error=str(exc)[:200])
            continue

        raw_sql = str(data.get("sql", "")).strip()
        result.explanation = str(data.get("explanation", ""))[
            : settings.text2sql_explanation_max_chars
        ]

        #  validate 
        try:
            guarded = validate_sql(raw_sql)
        except SqlGuardError as exc:
            failures.append((raw_sql, f"rejected by safety guard: {exc}"))
            result.repairs.append(str(exc))
            log.warning("text2sql.rejected", attempt=attempt, reason=str(exc)[:160])
            continue

        result.sql = guarded.sql

        # execute 
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

        # empty result: relax once, then stop 
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

    # Attempt failed. The graph decides whether that's worth another go.
    if not result.error:
        result.error = "could not produce valid SQL"
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.warning("text2sql.attempt_failed", repairs=result.repairs)
    return result


# ask for the same query again with its tightest filter dropped
async def _relax(question: str, sql: str) -> str | None:
    try:
        data, _ = await get_router().complete_json(
            render(
                TEXT2SQL,
                schema=schema_card(),
                input="This query returned no rows:\n\n"
                f"{sql}\n\n"
                f"Original question: {question}\n\n"
                "Rewrite it with ONLY the single most restrictive filter removed, "
                "so it returns closely related rows. Keep every other condition. "
                "Do not invent tables or columns.",
            ),
            SQL_SCHEMA,
            tier="fast",
            max_tokens=settings.text2sql_relax_max_tokens,
        )
        return str(data.get("sql", "")).strip() or None
    except Exception as exc:
        log.warning("text2sql.relax_generation_failed", error=str(exc)[:160])
        return None
