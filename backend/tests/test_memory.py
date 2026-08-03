"""Conversation memory: persistence, focus slots, and the rewrite gate.

Almost all of this is deliberately testable without an LLM call - the storage
layer, the slot merge, and the decision of *whether* to rewrite are all
deterministic. Only the rewrite itself needs a model.
"""

from __future__ import annotations

import uuid

import pytest

from app.agent.nodes.contextualize import needs_rewrite
from app.memory import store
from app.memory.store import USER_MEMORY_ALLOWLIST, ThreadState
from app.memory.summarizer import SUMMARY_AFTER_TURNS, should_summarise
from app.tools.db import fetch


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    try:
        fetch("SELECT 1")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres unreachable ({exc}). Run: docker compose up -d")
    store.ensure_schema()


@pytest.fixture
def thread_id():
    tid = f"test-{uuid.uuid4().hex[:12]}"
    yield tid
    store.clear_thread(tid)


# ---------------------------------------------------------------------------
class TestTurnHistory:
    def test_roundtrip(self, thread_id):
        store.append_turn(thread_id, "hello", "hi there", route="SMALLTALK")
        store.append_turn(thread_id, "who is the cardiologist?",
                          "Dr. Prakash Priyadarshan.", route="SQL")

        history = store.get_history(thread_id)
        assert len(history) == 2
        assert history[0] == ["hello", "hi there"]
        assert history[1][0] == "who is the cardiologist?"

    def test_history_is_oldest_first(self, thread_id):
        for i in range(4):
            store.append_turn(thread_id, f"q{i}", f"a{i}")
        history = store.get_history(thread_id)
        assert [h[0] for h in history] == ["q0", "q1", "q2", "q3"]

    def test_limit_returns_the_most_recent(self, thread_id):
        for i in range(6):
            store.append_turn(thread_id, f"q{i}", f"a{i}")
        history = store.get_history(thread_id, limit=2)
        assert [h[0] for h in history] == ["q4", "q5"]

    def test_threads_are_isolated(self, thread_id):
        other = f"{thread_id}-other"
        store.append_turn(thread_id, "mine", "a")
        store.append_turn(other, "theirs", "b")
        try:
            assert [h[0] for h in store.get_history(thread_id)] == ["mine"]
            assert [h[0] for h in store.get_history(other)] == ["theirs"]
        finally:
            store.clear_thread(other)

    def test_unknown_thread_is_empty(self):
        assert store.get_history("no-such-thread-xyz") == []


class TestThreadState:
    def test_defaults_for_new_thread(self, thread_id):
        state = store.get_state(thread_id)
        assert state.summary == ""
        assert state.focus_entities == {}
        assert state.turn_count == 0

    def test_save_and_reload(self, thread_id):
        store.save_state(ThreadState(
            thread_id=thread_id,
            summary="Patient asked about cardiology fees.",
            focus_entities={"doctor": "Dr. Prakash Priyadarshan"},
            turn_count=3,
        ))
        reloaded = store.get_state(thread_id)
        assert reloaded.summary.startswith("Patient asked")
        assert reloaded.focus_entities["doctor"] == "Dr. Prakash Priyadarshan"
        assert reloaded.turn_count == 3

    def test_save_is_idempotent(self, thread_id):
        for n in (1, 2, 3):
            store.save_state(ThreadState(thread_id=thread_id, turn_count=n))
        assert store.get_state(thread_id).turn_count == 3

    def test_focus_block_renders_for_prompts(self):
        state = ThreadState("t", focus_entities={"doctor": "Dr. X", "weekday": "Tuesday"})
        block = state.focus_block()
        assert "doctor: Dr. X" in block
        assert "weekday: Tuesday" in block

    def test_focus_block_empty_when_nothing_in_focus(self):
        assert ThreadState("t").focus_block() == ""


class TestFocusSlots:
    def test_new_entities_fill_slots(self):
        merged = store.update_focus({}, {"doctor": ["Dr. A"], "weekday": ["Tuesday"]})
        assert merged == {"doctor": "Dr. A", "weekday": "Tuesday"}

    def test_mentioned_entities_replace(self):
        merged = store.update_focus({"doctor": "Dr. A"}, {"doctor": ["Dr. B"]})
        assert merged["doctor"] == "Dr. B"

    def test_unmentioned_slots_persist(self):
        """This is what lets 'and on Friday?' keep the previous doctor."""
        merged = store.update_focus(
            {"doctor": "Dr. A", "specialty": "Cardiology"},
            {"weekday": ["Friday"]},
        )
        assert merged["doctor"] == "Dr. A"
        assert merged["specialty"] == "Cardiology"
        assert merged["weekday"] == "Friday"

    def test_irrelevant_entity_kinds_are_ignored(self):
        merged = store.update_focus({}, {"nonsense": ["x"]})
        assert merged == {}


# ---------------------------------------------------------------------------
class TestRewriteGate:
    """Rewriting costs an LLM call, so it must only fire when needed."""

    @pytest.mark.parametrize("query", [
        "and what is his fee?",
        "what about Tuesday?",
        "does that one need fasting?",
        "how about Friday",
        "and the price?",
        "is she available?",
    ])
    def test_referring_questions_need_rewriting(self, query: str):
        should, reason = needs_rewrite(query, has_history=True)
        assert should, f"{query!r} depends on context ({reason})"

    @pytest.mark.parametrize("query", [
        "What is Dr. Maiya Gunasekara's consultation fee?",
        "Which doctors are available on Tuesday?",
        "What services does the Heart Centre offer?",
        "How much is the dengue test and what preparation is needed?",
    ])
    def test_self_contained_questions_skip_the_llm(self, query: str):
        should, reason = needs_rewrite(query, has_history=True)
        assert not should, f"{query!r} should not need rewriting ({reason})"

    def test_first_turn_never_rewrites(self):
        should, reason = needs_rewrite("and what is his fee?", has_history=False)
        assert not should
        assert reason == "no prior turns"

    def test_empty_query_is_safe(self):
        assert not needs_rewrite("", has_history=True)[0]


class TestSummarisation:
    def test_not_triggered_early(self):
        assert not should_summarise(ThreadState("t", turn_count=2))

    def test_triggered_once_threshold_passed(self):
        assert should_summarise(ThreadState("t", turn_count=SUMMARY_AFTER_TURNS))

    def test_not_regenerated_every_turn(self):
        """Summarising is a background cost, not part of every turn."""
        state = ThreadState("t", summary="existing", turn_count=SUMMARY_AFTER_TURNS + 1)
        assert not should_summarise(state)


# ---------------------------------------------------------------------------
class TestUserMemoryPrivacy:
    """Health data must never be persisted across sessions."""

    def test_allowlisted_keys_are_stored(self, thread_id):
        assert store.set_user_memory(thread_id, "preferred_specialty", "Cardiology")
        assert store.get_user_memory(thread_id)["preferred_specialty"] == "Cardiology"

    @pytest.mark.parametrize("key", [
        "symptoms", "conditions", "diagnosis", "medications",
        "test_results", "blood_pressure", "allergies",
    ])
    def test_health_data_is_refused(self, thread_id, key: str):
        assert not store.set_user_memory(thread_id, key, "something sensitive")
        assert key not in store.get_user_memory(thread_id)

    def test_allowlist_contains_no_health_terms(self):
        banned = ("symptom", "condition", "diagnos", "medication",
                  "result", "disease", "illness", "allerg")
        for key in USER_MEMORY_ALLOWLIST:
            assert not any(b in key.lower() for b in banned), key

    def test_values_are_length_capped(self, thread_id):
        store.set_user_memory(thread_id, "preferred_language", "x" * 5000)
        assert len(store.get_user_memory(thread_id)["preferred_language"]) <= 500
