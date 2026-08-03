"""Rolling conversation summary.

Injecting a whole transcript into every prompt grows tokens without bound,
which on a free tier means a long conversation eventually stops working. After
a threshold the older turns are compressed into a short summary and dropped
from the injected window, so prompt size stays roughly constant however long
the chat runs.

The summary is regenerated only every few turns, not on each one - it is a
background cost, not part of the critical path.
"""

from __future__ import annotations

from app.llm.base import Msg
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.memory.store import ThreadState

log = get_logger(__name__)

#: Start summarising once a thread passes this many turns.
SUMMARY_AFTER_TURNS = 6
#: Regenerate at most this often.
SUMMARY_EVERY_N_TURNS = 4
MAX_SUMMARY_WORDS = 120

PROMPT = """Summarise a patient's conversation with a hospital assistant in at \
most 120 words.

Keep: which doctors, tests, packages or centres were discussed, what the patient
is trying to arrange, and any preference they stated.
Drop: pleasantries, the assistant's phrasing, and anything already superseded.
Do NOT record symptoms, conditions or test results - those must not be retained.

Write it as plain prose, third person, no preamble."""


def should_summarise(state: ThreadState) -> bool:
    if state.turn_count < SUMMARY_AFTER_TURNS:
        return False
    if not state.summary:
        return True
    return state.turn_count % SUMMARY_EVERY_N_TURNS == 0


async def summarise(
    history: list[list[str]], previous: str = ""
) -> str | None:
    """Compress older turns. Returns None if the call fails (summary is optional)."""
    if not history:
        return None

    lines = []
    if previous:
        lines.append(f"Summary so far: {previous}")
    for user, assistant in history:
        lines.append(f"Patient: {user}")
        lines.append(f"Assistant: {assistant[:200]}")

    try:
        response = await get_router().complete(
            [Msg("system", PROMPT), Msg("user", "\n".join(lines))],
            tier="fast",
            max_tokens=400,
        )
    except Exception as exc:
        log.warning("summarizer.failed", error=str(exc)[:160])
        return None

    text = response.text.strip()
    if not text:
        return None

    words = text.split()
    if len(words) > MAX_SUMMARY_WORDS:
        text = " ".join(words[:MAX_SUMMARY_WORDS]) + "..."

    log.info("summarizer.updated", words=len(text.split()))
    return text
