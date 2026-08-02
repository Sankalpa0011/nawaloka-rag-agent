"""Database connections.

Two engines, deliberately separate:

``ro_engine``     connects as ``chatbot_ro`` and is used for EVERY query the
                  agent issues, whether it came from a curated tool or from
                  generated SQL. The role cannot write, cannot read the
                  conversation logs, and is capped at a 5 s statement timeout
                  (see db/init/004_ro_role.sh).

``owner_engine``  connects as the owning role and is used only by application
                  code the model cannot influence: checkpoints, routing logs,
                  user memory.

Keeping them apart is what makes the SQL tool safe by construction rather than
by validation alone. The sqlglot guard in sql_guard.py is the first line of
defence; this is the one that holds if the guard is ever bypassed.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

# Rows returned to the model. Everything the agent legitimately asks for on
# this dataset fits well inside this; the cap exists so a mistaken query cannot
# push tens of thousands of rows into a prompt.
MAX_ROWS = 200


def _sqlalchemy_url(url: str) -> str:
    """Normalise a libpq URL to the psycopg3 driver SQLAlchemy expects."""
    if url.startswith("postgresql+"):
        return url
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache(maxsize=1)
def ro_engine() -> Engine:
    return create_engine(
        _sqlalchemy_url(settings.database_ro_url),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        # Belt and braces: the role already defaults to read-only transactions.
        execution_options={"postgresql_readonly": True},
    )


@lru_cache(maxsize=1)
def owner_engine() -> Engine:
    return create_engine(
        _sqlalchemy_url(settings.database_url), pool_pre_ping=True, pool_size=5
    )


def fetch(
    sql: str,
    params: dict[str, Any] | None = None,
    engine: Engine | None = None,
    max_rows: int = MAX_ROWS,
) -> tuple[list[dict], int]:
    """Run a read-only query with bound parameters.

    Returns ``(rows, elapsed_ms)``. Rows are plain dicts with JSON-friendly
    values so they can go straight into a prompt or an SSE payload.
    """
    engine = engine or ro_engine()
    started = time.perf_counter()

    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        cols = list(result.keys())
        rows = [dict(zip(cols, _coerce(r))) for r in result.fetchmany(max_rows)]

    elapsed = int((time.perf_counter() - started) * 1000)
    return rows, elapsed


def _coerce(row: Sequence[Any]) -> list[Any]:
    from datetime import date, time as dtime, timedelta
    from decimal import Decimal

    out: list[Any] = []
    for v in row:
        if isinstance(v, Decimal):
            out.append(float(v))
        elif isinstance(v, (dtime, date)):
            out.append(v.isoformat())
        elif isinstance(v, timedelta):
            out.append(str(v))
        else:
            out.append(v)
    return out


def healthcheck() -> dict:
    try:
        rows, ms = fetch("SELECT count(*) AS n FROM v_doctor_directory")
        return {"ok": True, "doctors": rows[0]["n"], "elapsed_ms": ms}
    except Exception as exc:  # surfaced by /health, never raised into a turn
        log.error("db.healthcheck_failed", error=str(exc))
        return {"ok": False, "error": str(exc)}
