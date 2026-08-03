"""Resolving what specialty a patient meant.

Two failure modes, and they pull in opposite directions. Matching too loosely
offers a neurosurgery patient the ENT surgeon; matching too strictly dead-ends
a real question on "our database has no records". The tests below pin both
edges, plus the recovery that catches whatever falls between them.
"""

from __future__ import annotations

import pytest

from app.agent.nodes.sql_agent import _call_curated
from app.tools.db import fetch
from app.tools.sql_tools import _resolve_specialty, _specialty_tokens, find_doctors


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    try:
        fetch("SELECT 1")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres unreachable ({exc})")


class TestDistinctiveTokens:
    def test_shared_head_nouns_are_identified_from_the_data(self):
        """The generic words are derived, not hardcoded, so they cannot drift."""
        _, shared = _specialty_tokens()
        assert {"surgery", "medicine"} <= set(shared)

    def test_every_specialty_keeps_at_least_one_token(self):
        tokens, _ = _specialty_tokens()
        assert tokens
        for name, words in tokens.items():
            assert words, f"{name} was left with nothing to match on"


class TestResolution:
    @pytest.mark.parametrize("query,expected", [
        ("cardilogy", "Cardiology"),
        ("cardiolgy", "Cardiology"),
        ("nuerosurgery", "Neurosurgery"),
        ("neurosurgen", "Neurosurgery"),
        ("neuro surgery", "Neurosurgery"),
        ("pediatrics", "Paediatrics"),
        ("peadiatrics", "Paediatrics"),
        ("radioligy", "Radiology & Imaging"),
        ("rhumatology", "Rheumatology & Sports Medicine"),
        ("gastroentrology", "General Surgery & Gastroenterology"),
        ("ent surgury", "ENT Surgery"),
    ])
    def test_misspellings_resolve(self, query: str, expected: str):
        assert _resolve_specialty(query) == expected

    @pytest.mark.parametrize("query", [
        "bone surgery", "heart surgery", "eye surgery", "dental surgery",
        "cancer surgery", "plastic surgery", "kidney doctor", "skin problem",
        "liver specialist", "neuron issues",
    ])
    def test_specialties_that_do_not_exist_are_not_invented(self, query: str):
        """Whole-string similarity could not do this, which is why it was replaced.

        Measured on the live table, a phrase for a specialty that does not
        exist and a genuine typo land on the same score - 'eye surgery' and
        'ent surgury' both hit ENT Surgery at 0.600 - and a real typo
        ('rhumatology', 0.323) scores *below* a wrong match ('bone surgery',
        0.471). No cutoff separates them, because the score is dominated by the
        head noun the names share. Comparing distinctive words instead makes
        'eye' vs 'ent' the deciding comparison, which it always should have been.
        """
        assert _resolve_specialty(query) is None

    @pytest.mark.parametrize("query", ["surgery", "medicine", "general"])
    def test_words_shared_by_several_specialties_resolve_to_none(self, query: str):
        """Asking broadly is not a typo - a literal match is the right answer."""
        assert _resolve_specialty(query) is None

    @pytest.mark.parametrize("query", [
        "Neurosurgery", "Cardiology", "Paediatrics", "rheumatology", "ENT Surgery",
    ])
    def test_literal_names_are_left_to_the_substring_match(self, query: str):
        assert _resolve_specialty(query) is None
        assert find_doctors(specialty=query).row_count >= 1


class TestNoCrossSpecialtyBleed:
    def test_neurosurgery_does_not_return_the_ent_surgeon(self):
        """Regression: 'Neurosurgery' returned 2 rows, the second being ENT Surgery.

        pg_trgm scored them 0.316 against a 0.3 floor, purely on the shared
        letters of "Surgery". The synthesiser happened to drop the extra row,
        so the only visible symptom was a row count nobody would question.
        """
        for query in ("Neurosurgery", "neuro surgery"):
            result = find_doctors(specialty=query)
            assert {r["specialty"] for r in result.rows} == {"Neurosurgery"}

    def test_asking_broadly_for_surgery_still_returns_every_surgical_specialty(self):
        specialties = {r["specialty"] for r in find_doctors(specialty="surgery").rows}
        assert len(specialties) >= 3


class TestDepartmentArgument:
    """Departments carry internal names, so this argument is usually a near-miss."""

    def test_a_specialty_passed_as_a_department_still_finds_the_doctor(self):
        """Regression: department="Paediatrics" reported no paediatrician.

        The column holds "Paediatric Care", so the ILIKE matched nothing - next
        to a doctor whose specialty is exactly Paediatrics. The answer said no
        matching records were found, which was true of the query and false of
        the hospital.
        """
        result = _call_curated("find_doctors", {"department": "Paediatrics"})
        assert result.row_count >= 1
        assert "Paediatrics" in {r["specialty"] for r in result.rows}

    @pytest.mark.parametrize("value,expected", [
        ("Paediatrics", "Paediatrics"),          # specialty in the department slot
        ("Heart Centre", "Cardiology"),          # the real department name
        ("Cardiology", "Cardiology"),            # specialty again
        ("Neurosurgery", "Neurosurgery"),
    ])
    def test_either_vocabulary_is_accepted(self, value: str, expected: str):
        result = _call_curated("find_doctors", {"department": value})
        assert expected in {r["specialty"] for r in result.rows}

    def test_a_department_that_does_not_exist_reaches_the_complaint_path(self):
        result = _call_curated("find_doctors", {"department": "Oncology"})
        assert result.row_count > 0


class TestComplaintRecovery:
    @pytest.mark.parametrize("complaint,expected", [
        ("neuron issues", "Neurosurgery"),
        ("nerve problem", "Neurosurgery"),
        ("brain issue", "Neurosurgery"),
        ("heart surgery", "Cardiology"),
        ("my mother's stomach", "General Surgery & Gastroenterology"),
    ])
    def test_a_symptom_filed_as_a_specialty_still_reaches_a_clinic(
        self, complaint: str, expected: str
    ):
        """Regression: "my mother got neuron issue, who do I contact?"

        The tool model filed the patient's own words as a specialty. That
        matched nothing and the turn dead-ended on "our database does not have
        any records matching that criteria" - for a question with a perfectly
        good answer. Recovering deterministically means the model only has to
        reach the doctor table, not to have classified the phrasing first.
        """
        result = _call_curated("find_doctors", {"specialty": complaint})
        assert result.tool == "suggest_specialty"
        assert result.row_count > 0
        assert expected in {r["specialty"] for r in result.rows}

    def test_an_unmappable_complaint_still_offers_a_starting_point(self):
        result = _call_curated("find_doctors", {"specialty": "eye surgery"})
        assert result.row_count > 0
        assert "General Medicine" in {r["specialty"] for r in result.rows}

    def test_the_guidance_is_never_presented_as_a_diagnosis(self):
        result = _call_curated("find_doctors", {"specialty": "neuron issues"})
        assert any("not" in n and "diagnosis" in n for n in result.notes)

    @pytest.mark.parametrize("symptom", ["fever", "cold", "chest pain"])
    def test_a_symptom_filed_as_a_doctors_name_is_recovered(self, symptom: str):
        """Regression: ``find_doctors(name="fever")``.

        The model put the illness in the name slot, and the answer reported no
        such doctor - to someone asking who to see about a fever.
        """
        result = _call_curated("find_doctors", {"name": symptom})
        assert result.tool == "suggest_specialty"
        assert result.row_count > 0

    @pytest.mark.parametrize("name", ["Zzzabcd", "Dr. Nonexistent"])
    def test_an_unknown_doctor_is_still_reported_as_unknown(self, name: str):
        """The opposite error. A name is not a complaint, and inventing a
        referral for one would be worse than the dead end it replaced."""
        result = _call_curated("find_doctors", {"name": name})
        assert result.tool == "find_doctors"
        assert result.row_count == 0

    def test_a_narrowing_filter_is_not_silently_dropped(self):
        """A weekday can legitimately empty a result set.

        Re-reading the query as a complaint would answer a different question
        from the one asked, so the recovery must not fire when another filter
        could be the reason for the empty result.
        """
        result = _call_curated(
            "find_doctors", {"specialty": "Cardiology", "weekday": "Sunday"}
        )
        assert result.tool == "find_doctors"
