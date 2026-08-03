"""Conversation memory, persisted in Postgres.

Three layers, each with a different lifetime:

**Turn history** - the literal exchange, per thread. Bounded when injected: the
last few turns go into prompts, never the whole transcript.

**Thread state** - a rolling summary plus `focus_entities`, the explicit slots
tracking what the conversation is currently *about* (which doctor, which test,
which weekday). Slots are what make "and what is his fee?" resolvable; asking a
model to find the referent in raw history is far less reliable, and costs a
call every turn.

**User memory** - survives across sessions, and is written through a strict
allowlist. Symptoms and health conditions are deliberately never persisted:
that is identifiable health data, and a chatbot has no business keeping it.

The tables are created on demand rather than in db/init, because those scripts
only run when the Postgres volume is first initialised - a schema change would
otherwise silently not apply to an existing database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from app.logging_conf import get_logger
from app.tools.db import owner_engine

log = get_logger(__name__)

#: Cross-session keys that may be persisted. Anything not listed is dropped.
#: Deliberately excludes symptoms, conditions, medications and test results.
USER_MEMORY_ALLOWLIST = frozenset({
    "preferred_language",
    "preferred_specialty",
    "last_doctor_viewed",
    "interested_package",
    "preferred_weekday",
})

#: Entity slots carried between turns.
FOCUS_SLOTS = ("doctor", "specialty", "lab_test", "package", "weekday")

MAX_HISTORY_TURNS = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_messages (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    thread_id  TEXT        NOT NULL,
    role       TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    route      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_thread_messages_thread
    ON thread_messages (thread_id, id);

CREATE TABLE IF NOT EXISTS thread_state (
    thread_id      TEXT PRIMARY KEY,
    summary        TEXT,
    focus_entities JSONB       NOT NULL DEFAULT '{}'::jsonb,
    turn_count     INT         NOT NULL DEFAULT 0,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
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
    log.debug("memory.schema_ready")


@dataclass
class ThreadState:
    thread_id: str
    summary: str = ""
    focus_entities: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0

    def focus_block(self) -> str:
        """Rendered for prompts. Empty when nothing is in focus."""
        live = {k: v for k, v in self.focus_entities.items() if v}
        if not live:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in live.items())


# ---------------------------------------------------------------------------
def get_state(thread_id: str) -> ThreadState:
    ensure_schema()
    with owner_engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT summary, focus_entities, turn_count "
                "FROM thread_state WHERE thread_id = :t"
            ),
            {"t": thread_id},
        ).mappings().first()

    if not row:
        return ThreadState(thread_id=thread_id)

    focus = row["focus_entities"]
    if isinstance(focus, str):
        focus = json.loads(focus)
    return ThreadState(
        thread_id=thread_id,
        summary=row["summary"] or "",
        focus_entities=focus or {},
        turn_count=row["turn_count"] or 0,
    )


def save_state(state: ThreadState) -> None:
    ensure_schema()
    with owner_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO thread_state (thread_id, summary, focus_entities, turn_count, updated_at)
                VALUES (:t, :s, CAST(:f AS jsonb), :c, now())
                ON CONFLICT (thread_id) DO UPDATE
                   SET summary = EXCLUDED.summary,
                       focus_entities = EXCLUDED.focus_entities,
                       turn_count = EXCLUDED.turn_count,
                       updated_at = now()
                """
            ),
            {
                "t": state.thread_id,
                "s": state.summary,
                "f": json.dumps(state.focus_entities),
                "c": state.turn_count,
            },
        )


def append_turn(thread_id: str, user: str, assistant: str, route: str = "") -> None:
    ensure_schema()
    with owner_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO thread_messages (thread_id, role, content, route) "
                "VALUES (:t, 'user', :u, :r), (:t, 'assistant', :a, :r)"
            ),
            {"t": thread_id, "u": user, "a": assistant, "r": route},
        )


def get_history(thread_id: str, limit: int = MAX_HISTORY_TURNS) -> list[list[str]]:
    """Recent turns as [[user, assistant], ...], oldest first."""
    ensure_schema()
    with owner_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT role, content FROM thread_messages "
                "WHERE thread_id = :t ORDER BY id DESC LIMIT :n"
            ),
            {"t": thread_id, "n": limit * 2},
        ).mappings().all()

    turns: list[list[str]] = []
    pending_assistant: str | None = None
    for row in rows:  # newest first
        if row["role"] == "assistant":
            pending_assistant = row["content"]
        elif row["role"] == "user":
            turns.append([row["content"], pending_assistant or ""])
            pending_assistant = None
    turns.reverse()
    return turns


def clear_thread(thread_id: str) -> None:
    ensure_schema()
    with owner_engine().begin() as conn:
        conn.execute(text("DELETE FROM thread_messages WHERE thread_id = :t"),
                     {"t": thread_id})
        conn.execute(text("DELETE FROM thread_state WHERE thread_id = :t"),
                     {"t": thread_id})


# ---------------------------------------------------------------------------
def update_focus(current: dict[str, str], new_entities: dict[str, Any]) -> dict[str, str]:
    """Merge freshly-detected entities into the focus slots.

    Newly mentioned entities replace the slot; unmentioned slots persist, which
    is what lets a follow-up like "and on Friday?" keep the doctor from the
    previous turn.
    """
    merged = dict(current)
    for slot in FOCUS_SLOTS:
        values = new_entities.get(slot)
        if values:
            merged[slot] = values[0] if isinstance(values, list) else str(values)
    return merged


# ---------------------------------------------------------------------------
def set_user_memory(user_id: str, key: str, value: str, thread_id: str = "") -> bool:
    """Persist a cross-session preference. Returns False if the key is not allowed."""
    if key not in USER_MEMORY_ALLOWLIST:
        log.warning("memory.rejected_key", key=key,
                    reason="not on the non-sensitive allowlist")
        return False

    ensure_schema()
    with owner_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO user_memory (user_id, key, value, source_thread, updated_at)
                VALUES (:u, :k, :v, :t, now())
                ON CONFLICT (user_id, key) DO UPDATE
                   SET value = EXCLUDED.value,
                       source_thread = EXCLUDED.source_thread,
                       updated_at = now()
                """
            ),
            {"u": user_id, "k": key, "v": value[:500], "t": thread_id or None},
        )
    return True


def get_user_memory(user_id: str) -> dict[str, str]:
    ensure_schema()
    with owner_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT key, value FROM user_memory WHERE user_id = :u"),
            {"u": user_id},
        ).mappings().all()
    return {r["key"]: r["value"] for r in rows}
