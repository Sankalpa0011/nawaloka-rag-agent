"""Rolling summary of a conversation, so prompt size stays bounded as a thread grows.

Regenerated every few turns rather than every turn.
"""

from __future__ import annotations

from app.config import settings
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.prompts.templates import SUMMARIZE, render
from app.memory.store import ThreadState

log = get_logger(__name__)

# don't bother summarising until a thread is this long
SUMMARY_AFTER_TURNS = settings.memory_summary_after_turns
# and after that, only redo it every N turns
SUMMARY_EVERY_N_TURNS = settings.memory_summary_every_n_turns
MAX_SUMMARY_WORDS = settings.memory_max_summary_words


def should_summarise(state: ThreadState) -> bool:
    if state.turn_count < settings.memory_summary_after_turns:
        return False
    if not state.summary:
        return True
    return state.turn_count % settings.memory_summary_every_n_turns == 0


# Returns None on failure - summarising is optional, never fatal to the turn.
async def summarise(
    history: list[list[str]], previous: str = ""
) -> str | None:
    if not history:
        return None

    lines = []
    if previous:
        lines.append(f"Summary so far: {previous}")
    for user, assistant in history:
        lines.append(f"Patient: {user}")
        lines.append(f"Assistant: {assistant[: settings.memory_summary_answer_chars]}")

    try:
        response = await get_router().complete(
            render(SUMMARIZE, input="\n".join(lines)),
            tier="fast",
            max_tokens=settings.memory_summary_max_tokens,
        )
    except Exception as exc:
        log.warning("summarizer.failed", error=str(exc)[:160])
        return None

    text = response.text.strip()
    if not text:
        return None

    words = text.split()
    if len(words) > settings.memory_max_summary_words:
        text = " ".join(words[: settings.memory_max_summary_words]) + "..."

    log.info("summarizer.updated", words=len(text.split()))
    return text
