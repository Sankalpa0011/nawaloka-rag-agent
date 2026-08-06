"""Audit log of routing decisions, one row per turn.

Written with owner_engine only - chatbot_ro is revoked from this table.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from app.config import settings
from app.logging_conf import get_logger
from app.tools.db import owner_engine

log = get_logger(__name__)

# Keep the query text bounded - this is an audit trail, not a transcript store.
MAX_QUERY_CHARS = settings.decision_log_max_query_chars
MAX_RATIONALE_CHARS = settings.decision_log_max_rationale_chars

# Mirrors db/init/001_schema.sql - keep both in sync.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    thread_id        TEXT        NOT NULL,
    raw_query        TEXT        NOT NULL,
    contextual_query TEXT,
    signals          JSONB,
    llm_route        TEXT,
    final_route      TEXT        NOT NULL,
    confidence       REAL,
    rationale        TEXT,
    overridden       BOOLEAN     NOT NULL DEFAULT FALSE,
    latency_ms       INT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_thread
    ON routing_decisions (thread_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_created
    ON routing_decisions (created_at DESC);
"""

_schema_ready = False


def ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with owner_engine().begin() as conn:
        for statement in _SCHEMA.strip().split(";"):
            if statement.strip():
                conn.execute(text(statement))
    _schema_ready = True
    log.debug("decisions.schema_ready")


def log_decision(
    thread_id: str,
    raw_query: str,
    final_route: str,
    contextual_query: str = "",
    signals: dict[str, Any] | None = None,
    llm_route: str = "",
    confidence: float | None = None,
    rationale: str = "",
    overridden: bool = False,
    latency_ms: int | None = None,
) -> bool:
    """Append one decision; returns False on failure instead of raising."""
    if not final_route:
        return False

    try:
        ensure_schema()
        with owner_engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO routing_decisions (
                        thread_id, raw_query, contextual_query, signals,
                        llm_route, final_route, confidence, rationale,
                        overridden, latency_ms
                    )
                    VALUES (
                        :thread_id, :raw_query, :contextual_query, CAST(:signals AS jsonb),
                        :llm_route, :final_route, :confidence, :rationale,
                        :overridden, :latency_ms
                    )
                    """
                ),
                {
                    "thread_id": thread_id or "default",
                    "raw_query": (raw_query or "")[:settings.decision_log_max_query_chars],
                    "contextual_query": (contextual_query or "")[
                        : settings.decision_log_max_query_chars
                    ] or None,
                    "signals": json.dumps(signals or {}),
                    "llm_route": llm_route or None,
                    "final_route": final_route,
                    "confidence": confidence,
                    "rationale": (rationale or "")[
                        : settings.decision_log_max_rationale_chars
                    ] or None,
                    "overridden": bool(overridden),
                    "latency_ms": latency_ms,
                },
            )
        return True
    except Exception as exc:
        log.warning("decisions.write_failed", error=str(exc)[:160],
                    route=final_route)
        return False


def recent(
    limit: int = settings.decision_log_recent_limit, thread_id: str = ""
) -> list[dict]:
    """Most recent decisions, newest first."""
    ensure_schema()
    sql = (
        "SELECT id, thread_id, raw_query, contextual_query, signals, llm_route, "
        "       final_route, confidence, rationale, overridden, latency_ms, created_at "
        "FROM routing_decisions "
    )
    params: dict[str, Any] = {"n": limit}
    if thread_id:
        sql += "WHERE thread_id = :t "
        params["t"] = thread_id
    sql += "ORDER BY id DESC LIMIT :n"

    with owner_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()

    out = []
    for row in rows:
        record = dict(row)
        record["created_at"] = record["created_at"].isoformat()
        if isinstance(record["signals"], str):
            record["signals"] = json.loads(record["signals"])
        out.append(record)
    return out


def route_counts(thread_id: str = "") -> dict[str, int]:
    """Count of turns per route."""
    ensure_schema()
    sql = "SELECT final_route, count(*) AS n FROM routing_decisions "
    params: dict[str, Any] = {}
    if thread_id:
        sql += "WHERE thread_id = :t "
        params["t"] = thread_id
    sql += "GROUP BY final_route ORDER BY n DESC"

    with owner_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {r["final_route"]: r["n"] for r in rows}


def clear_thread(thread_id: str) -> None:
    ensure_schema()
    with owner_engine().begin() as conn:
        conn.execute(text("DELETE FROM routing_decisions WHERE thread_id = :t"),
                     {"t": thread_id})
