"""FAQ fast path: what it may answer, and what it must never answer.

A cache that returns a wrong answer quickly is worse than no cache, so most of
these tests are about restraint rather than hit rate.
"""

from __future__ import annotations

import pytest

from app.agent.nodes.faq import (
    CACHE_THRESHOLD,
    CURATED_THRESHOLD,
    cache_eligibility,
    faq_lookup,
    faq_node,
    needs_live_data,
)
from app.ingest.faq_seed import FAQS
from app.retrieval.qdrant_store import get_client
from app.tools.db import fetch


@pytest.fixture(scope="module", autouse=True)
def _require_services():
    try:
        fetch("SELECT 1")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres unreachable ({exc})")
    try:
        get_client().get_collections()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qdrant unreachable ({exc})")


# ---------------------------------------------------------------------------
class TestLiveDataBypass:
    """The structural guard: specific questions never get a canned answer."""

    @pytest.mark.parametrize("query,kind", [
        ("Which doctors are available on Tuesday?", "weekday"),
        ("What is Dr. Maiya Gunasekara's consultation fee?", "doctor"),
        ("How much is the Dengue NS1 test?", "lab_test"),
        ("What services does the Heart Centre offer?", "specialty"),
        ("I need a cardiologist", "specialty"),
        ("When does Prof. De Silva channel?", "doctor"),
    ])
    def test_specific_questions_bypass_the_faq(self, query: str, kind: str):
        blocking = needs_live_data(query)
        assert blocking, f"{query!r} should bypass the FAQ (expected {kind})"
        assert kind in blocking

    @pytest.mark.parametrize("query", [
        "what's the ambulance number",
        "where are you located",
        "what is the hospital phone number",
        "do you treat international patients",
        "do you have a blood bank",
    ])
    def test_general_questions_may_use_the_faq(self, query: str):
        assert not needs_live_data(query), f"{query!r} should stay on the fast path"

    @pytest.mark.parametrize("query", [
        "how do I book an appointment",
        "how do I channel a doctor",
    ])
    def test_booking_questions_take_the_slow_path_by_design(self, query: str):
        """An accepted cost of the transactional-term rule.

        "book" and "appointment" are transactional vocabulary, so these defer to
        the router even though a curated FAQ answers them well. The rule is
        deliberately blunt in this direction: losing the fast path costs a few
        seconds, whereas answering a live-data question from a static string
        costs correctness. See docs/faq-calibration.md.
        """
        assert needs_live_data(query)

    def test_tuesday_regression(self):
        """Regression: this was answered with the channeling hotline."""
        assert "weekday" in needs_live_data("Which doctors are available on Tuesday?")

    def test_heart_centre_regression(self):
        """Regression: this returned a list of every centre instead."""
        assert needs_live_data("What services does the Heart Centre offer?")

    @pytest.mark.parametrize("query", [
        "who are the all doctors in nawaloka?",
        "who are the doctors ?",
        "list your consultants",
        "which surgeons work here",
        "do you have physicians",
    ])
    def test_asking_for_a_whole_table_bypasses_the_faq(self, query: str):
        """Regression: naming a record *type* without naming a record.

        The gazetteer only fires on a specific name and the transactional
        lexicon only on words like "fee", so these triggered neither. "who are
        the all doctors in nawaloka?" matched the hospital-overview FAQ at 0.80
        and was answered with a list of specialist centres, while the same
        question without "nawaloka" scored under threshold and routed correctly
        - a coin flip no similarity cutoff can settle.
        """
        blocking = needs_live_data(query)
        assert blocking, f"{query!r} asks for database rows and must bypass the FAQ"
        assert any("record type" in b for b in blocking)

    @pytest.mark.parametrize("query", [
        "my brother got cold who needs to contact?",
        "my child has flu, who should we see?",
    ])
    async def test_a_symptom_never_gets_a_canned_answer(self, query: str):
        """Regression: a common cold was answered with the ambulance number.

        The guardrail runs before this node and already classifies these as
        SYMPTOM; the fast path simply was not asking. Someone describing an
        illness needs a clinic, not a stored string.
        """
        state = {
            "user_query": query,
            "contextual_query": query,
            "guardrail": {"flags": ["symptom_context"]},
        }
        out = await faq_node(state)
        assert out["cache"]["status"] == "bypassed"
        assert "symptom" in out["cache"]["reason"]
        assert "answer" not in out

    def test_the_hospital_overview_faq_still_works(self):
        """The bypass must not swallow the general question it sits next to."""
        assert not needs_live_data("Tell me about Nawaloka Hospitals")


# ---------------------------------------------------------------------------
class TestCuratedLookup:
    @pytest.mark.parametrize("query,expected_id", [
        ("what's the ambulance number", "faq-emergency"),
        ("where are you located", "faq-address"),
        ("hospital phone number", "faq-phone"),
        ("do you take foreign patients", "faq-international"),
        ("how do I book a doctor", "faq-book-channeling"),
    ])
    def test_common_questions_hit_the_right_entry(self, query: str, expected_id: str):
        hit = faq_lookup(query)
        assert hit is not None, f"no FAQ hit for {query!r}"
        assert hit["faq_id"] == expected_id
        assert hit["score"] >= CURATED_THRESHOLD

    @pytest.mark.parametrize("query", [
        "what is the capital of France",
        "write me a python script",
        "what is the weather tomorrow",
    ])
    def test_unrelated_questions_miss(self, query: str):
        assert faq_lookup(query) is None

    def test_every_answer_is_non_empty(self):
        for entry in FAQS:
            assert entry.answer.strip(), f"{entry.id} has no answer"
            assert entry.question.strip()

    def test_no_faq_hardcodes_a_price(self):
        """Prices live in the database; a cached price would go stale silently."""
        import re

        for entry in FAQS:
            assert not re.search(r"(?:LKR|Rs\.?)\s*[\d,]{3,}", entry.answer), (
                f"{entry.id} hard-codes a price - route price questions to SQL"
            )

    def test_faq_ids_are_unique(self):
        ids = [f.id for f in FAQS]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
class TestCacheEligibility:
    """Read broadly, write narrowly."""

    def test_vector_answers_are_cacheable(self):
        ok, _ = cache_eligibility({
            "route": "VECTOR", "answer": "text", "user_query": "a",
            "contextual_query": "a", "documents": [],
        })
        assert ok

    @pytest.mark.parametrize("route", ["SQL", "HYBRID"])
    def test_database_answers_are_never_cached(self, route: str):
        ok, reason = cache_eligibility({
            "route": route, "answer": "LKR 3,500", "user_query": "a",
            "contextual_query": "a",
        })
        assert not ok
        assert "mutable" in reason

    @pytest.mark.parametrize("route", ["EMERGENCY", "REFUSE"])
    def test_guardrail_answers_are_never_cached(self, route: str):
        ok, _ = cache_eligibility({
            "route": route, "answer": "call 1990", "user_query": "a",
            "contextual_query": "a",
        })
        assert not ok

    def test_context_dependent_answers_are_never_cached(self):
        ok, reason = cache_eligibility({
            "route": "VECTOR", "answer": "text",
            "user_query": "and his fee?", "contextual_query": "Dr Perera's fee?",
        })
        assert not ok
        assert "context" in reason

    def test_abstentions_are_never_cached(self):
        ok, _ = cache_eligibility({
            "route": "VECTOR", "answer": "I don't know", "abstained": True,
            "user_query": "a", "contextual_query": "a",
        })
        assert not ok

    @pytest.mark.parametrize("content_type", ["news", "promotion"])
    def test_time_sensitive_content_is_never_cached(self, content_type: str):
        ok, reason = cache_eligibility({
            "route": "VECTOR", "answer": "text", "user_query": "a",
            "contextual_query": "a",
            "documents": [{"content_type": content_type}],
        })
        assert not ok
        assert "time-sensitive" in reason

    def test_empty_answers_are_never_cached(self):
        ok, _ = cache_eligibility({
            "route": "VECTOR", "answer": "   ", "user_query": "a",
            "contextual_query": "a",
        })
        assert not ok


def test_cache_threshold_is_stricter_than_curated():
    """Generated answers were never human-reviewed, so they need a higher bar."""
    assert CACHE_THRESHOLD > CURATED_THRESHOLD
