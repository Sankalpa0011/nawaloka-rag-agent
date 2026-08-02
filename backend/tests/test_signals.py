"""Deterministic routing signals.

These are what make routing accuracy something other than "whatever the model
said". The gazetteer is derived from the live database, so these tests also
catch the case where entity extraction silently stops working because the
database is unreachable - which would degrade routing without any error.
"""

from __future__ import annotations

import pytest

from app.agent.signals import (
    SQL_LEXICON,
    VECTOR_LEXICON,
    build_gazetteer,
    extract_signals,
    reset_gazetteer,
)
from app.tools.db import fetch


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    try:
        fetch("SELECT 1")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres not reachable ({exc}). Run: docker compose up -d")
    reset_gazetteer()


class TestGazetteer:
    def test_is_built_from_the_database(self):
        gz = build_gazetteer()
        assert gz.size > 50, "gazetteer is empty - is the database reachable?"
        assert gz.doctor_terms and gz.specialty_terms
        assert gz.lab_terms and gz.package_terms

    def test_contains_real_entities(self):
        gz = build_gazetteer()
        assert "gunasekara" in gz.doctor_terms
        assert "cardiology" in gz.specialty_terms
        assert any("dengue" in t for t in gz.lab_terms)


class TestEntityDetection:
    @pytest.mark.parametrize("query,bucket,expected", [
        ("What is Dr. Maiya Gunasekara's fee?", "doctor", "Dr. Maiya Gunasekara"),
        ("gunasekara consultation charge", "doctor", "Dr. Maiya Gunasekara"),
        ("I need a cardiologist", "specialty", "Cardiology"),
        ("how much is the dengue test", "lab_test", None),
        ("tell me about the Well Woman Package (Under 40)", "package", None),
    ])
    def test_known_entities_are_recognised(self, query, bucket, expected):
        signals = extract_signals(query)
        assert bucket in signals.entities, f"{bucket} not detected in {query!r}"
        if expected:
            assert expected in signals.entities[bucket]

    def test_honorifics_do_not_block_matching(self):
        """Names are stored as 'Prof. Arjuna De Silva'; users type surnames."""
        assert "doctor" in extract_signals("what is De Silva's fee").entities

    def test_weekdays_are_detected(self):
        signals = extract_signals("who is available on tuesday?")
        assert "Tuesday".lower() in [d.lower() for d in signals.entities["weekday"]]

    def test_unrelated_text_yields_no_entities(self):
        assert extract_signals("what is the capital of France").entities == {}


class TestIntentLexicons:
    @pytest.mark.parametrize("query", [
        "how much is a consultation",
        "what is the price of the dengue test",
        "when is Dr. Lakshan available",
        "consultation fee",
        "channeling schedule",
    ])
    def test_transactional_queries_fire_sql_signals(self, query: str):
        assert extract_signals(query).has_sql, query

    @pytest.mark.parametrize("query", [
        "what services does the heart centre offer",
        "tell me about your facilities",
        "what is the elite vip centre",
        "hospital awards and accreditation",
    ])
    def test_descriptive_queries_fire_vector_signals(self, query: str):
        assert extract_signals(query).has_vector, query

    def test_mixed_query_fires_both(self):
        signals = extract_signals(
            "what does the heart centre offer and how much is a consultation"
        )
        assert signals.both, "a genuinely hybrid query must show both signal families"

    def test_currency_amount_is_a_transactional_cue(self):
        assert extract_signals("packages under LKR 20,000").has_sql

    def test_lexicons_do_not_overlap_destructively(self):
        """Shared terms would make every query look hybrid."""
        overlap = SQL_LEXICON & VECTOR_LEXICON
        assert not overlap, f"ambiguous terms in both lexicons: {overlap}"


class TestSignalReporting:
    def test_summary_is_human_readable_for_the_prompt(self):
        summary = extract_signals("what is Dr. Gunasekara's fee").summary()
        assert "Gunasekara" in summary
        assert "fee" in summary

    def test_summary_handles_no_signals(self):
        assert "no strong keyword signals" in extract_signals("xyzzy").summary()

    def test_dict_is_serialisable_for_the_trace_panel(self):
        d = extract_signals("cardiology fees on Monday").to_dict()
        assert set(d) >= {"sql_hits", "vector_hits", "entities", "has_sql", "has_vector"}

    def test_empty_query_is_safe(self):
        for value in ("", "   ", None):
            signals = extract_signals(value)
            assert not signals.has_sql and not signals.has_vector
