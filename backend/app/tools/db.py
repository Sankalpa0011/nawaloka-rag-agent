"""DB connections: ro_engine (chatbot_ro, read-only) handles all agent queries;
owner_engine is only used by code the model can't influence, as defense in
depth if sql_guard.py misses something.
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

# Row cap for anything going to the model, so a bad query can't dump rows into a prompt.
MAX_ROWS = settings.db_max_rows


# libpq URL -> the psycopg3 driver URL SQLAlchemy wants.
def _sqlalchemy_url(url: str) -> str:
    if url.startswith("postgresql+"):
        return url
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache(maxsize=1)
def ro_engine() -> Engine:
    return create_engine(
        _sqlalchemy_url(settings.database_ro_url),
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        # Belt and braces - the role is already read-only by default.
        execution_options={"postgresql_readonly": True},
    )


@lru_cache(maxsize=1)
def owner_engine() -> Engine:
    return create_engine(
        _sqlalchemy_url(settings.database_url),
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
    )


def fetch(
    sql: str,
    params: dict[str, Any] | None = None,
    engine: Engine | None = None,
    max_rows: int = MAX_ROWS,
) -> tuple[list[dict], int]:
    """Run a read-only query with bound params. Returns (rows, elapsed_ms) as
    plain JSON-friendly dicts."""
    engine = engine or ro_engine()
    started = time.perf_counter()

    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        cols = list(result.keys())
        rows = []
        for r in result.fetchmany(max_rows):
            rows.append(dict(zip(cols, _coerce(r))))

    elapsed = int((time.perf_counter() - started) * 1000)
    return rows, elapsed


# Make DB types JSON-safe.
def _coerce(row: Sequence[Any]) -> list[Any]:
    # TODO: these imports should be at module level, they're only in here
    # because _coerce started life as a throwaway helper
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
