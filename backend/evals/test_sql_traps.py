"""Regression tests for the dataset's data hazards.

Each test states the naive wrong answer it prevents. These run against the live
Postgres container (docker compose up) because the whole point is to verify the
views and tools behave on the real data, not on a mock.
"""

from __future__ import annotations

import pytest

from app.tools.db import fetch
from app.tools.sql_tools import (
    find_doctors,
    get_doctor_schedule,
    search_health_packages,
    search_lab_tests,
    suggest_specialty,
)


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    try:
        fetch("SELECT 1 AS ok")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres not reachable ({exc}). Run: docker compose up -d")


# ===========================================================================
# TRAP 1 - day_of_week is free text and one row says 'Daily'
# Naive failure: WHERE day_of_week = 'Tuesday' silently omits Dr. Pathirana.
# ===========================================================================
class TestDailyWeekday:
    @pytest.mark.parametrize("day", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                     "Friday", "Saturday", "Sunday"])
    def test_daily_doctor_appears_on_every_weekday(self, day: str):
        res = find_doctors(weekday=day)
        names = [r["doctor_name"] for r in res.rows]
        assert "Dr. Duminda Pathirana" in names, (
            f"the 'Daily' doctor is missing from {day} - "
            "the query is not going through v_sessions_expanded"
        )

    def test_naive_query_would_have_missed_him(self):
        """Demonstrates the trap actually exists rather than asserting a tautology."""
        naive, _ = fetch(
            "SELECT DISTINCT d.name FROM channeling_sessions cs "
            "JOIN doctors d ON d.id = cs.doctor_id "
            "WHERE cs.day_of_week = 'Tuesday'"
        )
        naive_names = {r["name"] for r in naive}
        assert "Dr. Duminda Pathirana" not in naive_names

        correct, _ = fetch(
            "SELECT DISTINCT doctor_name AS name FROM v_sessions_expanded "
            "WHERE weekday = 'Tuesday'"
        )
        assert "Dr. Duminda Pathirana" in {r["name"] for r in correct}

    def test_result_explains_why_daily_matched(self):
        res = find_doctors(weekday="Tuesday")
        assert any("Daily" in n for n in res.notes), (
            "the answer model is not told why a 'Daily' doctor matched Tuesday"
        )

    def test_weekday_aliases_are_normalised(self):
        assert find_doctors(weekday="tue").row_count == find_doctors(weekday="Tuesday").row_count


# ===========================================================================
# TRAP 2 - doctors 8 and 9 have no channeling_sessions rows
# Naive failure: INNER JOIN drops them, so the bot claims they do not exist;
# or it invents a schedule for them.
# ===========================================================================
class TestDoctorsWithoutSessions:
    def test_directory_lists_all_eleven_doctors(self):
        res = find_doctors(limit=50)
        assert res.row_count == 11

    def test_sessionless_doctors_are_present_and_flagged(self):
        res = find_doctors(limit=50)
        flagged = [r for r in res.rows if not r["has_published_sessions"]]
        assert {r["doctor_name"] for r in flagged} == {
            "Dr. Usha Samarasinghe",
            "Dr. Sandeep K. Sharma",
        }

    def test_notes_warn_against_implying_availability(self):
        res = get_doctor_schedule("Usha Samarasinghe")
        assert res.row_count >= 1
        assert res.rows[0]["has_published_sessions"] is False
        assert res.rows[0]["sessions"] is None
        assert any("no published channeling sessions" in n for n in res.notes)

    def test_naive_inner_join_would_have_dropped_them(self):
        naive, _ = fetch(
            "SELECT count(DISTINCT d.id) AS n FROM doctors d "
            "JOIN channeling_sessions cs ON cs.doctor_id = d.id"
        )
        assert naive[0]["n"] == 9, "expected the inner join to lose 2 of 11 doctors"


# ===========================================================================
# TRAP 3 - package contents are free text, not joinable to lab_tests
# Naive failure: joining the two tables returns nothing (or nonsense), so the
# bot says no package includes a Pap Smear.
# ===========================================================================
class TestPackageContents:
    def test_pap_smear_resolves_to_three_packages(self):
        res = search_health_packages(includes_service="Pap Smear")
        assert res.row_count == 3
        assert all(r["matched_services"] for r in res.rows)

    def test_the_join_really_does_fail(self):
        rows, _ = fetch(
            "SELECT count(*) AS n FROM v_package_services vps "
            "JOIN lab_tests lt ON lower(btrim(lt.test_name)) = lower(btrim(vps.service_item))"
        )
        assert rows[0]["n"] == 0, (
            "package text now joins to lab_tests - the trap assumption changed "
            "and the tool guidance needs revisiting"
        )

    def test_naming_mismatch_is_real(self):
        """'Lipid Profile' in packages vs 'Advanced Lipid & ApoB Profile' in lab_tests."""
        pkg = search_health_packages(includes_service="Lipid Profile")
        assert pkg.row_count > 0
        tests = search_lab_tests(query="Lipid")
        assert tests.row_count > 0
        pkg_names = {r["matched_services"] for r in pkg.rows}
        test_names = {r["test_name"] for r in tests.rows}
        assert not (pkg_names & test_names)

    def test_notes_warn_against_mapping_to_lab_tests(self):
        res = search_health_packages(includes_service="Pap Smear")
        assert any("NOT linked to the lab_tests" in n for n in res.notes)


# ===========================================================================
# TRAP 4 - honorifics break exact name matching
# ===========================================================================
class TestHonorifics:
    def test_surname_only_lookup_finds_professor(self):
        res = get_doctor_schedule("De Silva")
        assert res.row_count >= 1
        assert res.rows[0]["doctor_name"] == "Prof. Arjuna De Silva"
        assert res.rows[0]["consultation_fee"] == 4200.00

    def test_exact_match_would_have_failed(self):
        rows, _ = fetch("SELECT count(*) AS n FROM doctors WHERE name = :n",
                        {"n": "Arjuna De Silva"})
        assert rows[0]["n"] == 0

    @pytest.mark.parametrize("query", ["Maiya Gunasekara", "gunasekara",
                                       "Dr. Maiya Gunasekara", "Maiya"])
    def test_name_variants_all_resolve(self, query: str):
        res = get_doctor_schedule(query)
        assert "Dr. Maiya Gunasekara" in [r["doctor_name"] for r in res.rows]

    def test_nonexistent_doctor_is_refused_not_invented(self):
        res = get_doctor_schedule("Dr. Nonexistent Person")
        assert res.row_count == 0
        assert any("do not invent" in n.lower() for n in res.notes)


# ===========================================================================
# TRAP 6 - room_number overflows its declared VARCHAR(20)
# ===========================================================================
class TestRoomNumberOverflow:
    def test_long_room_name_survived_the_port(self):
        rows, _ = fetch(
            "SELECT room_number FROM channeling_sessions "
            "WHERE room_number LIKE 'Room 004%'"
        )
        assert rows[0]["room_number"] == "Room 004 (Paediatric Wing)"
        assert len(rows[0]["room_number"]) == 26  # declared VARCHAR(20)


# ===========================================================================
# General tool behaviour
# ===========================================================================
class TestToolBehaviour:
    def test_fee_filter_and_ordering(self):
        res = find_doctors(max_fee=3000)
        fees = [r["consultation_fee"] for r in res.rows]
        assert fees == sorted(fees)
        assert all(f <= 3000 for f in fees)

    def test_fasting_tests_quote_instructions(self):
        res = search_lab_tests(fasting_required=True)
        assert res.row_count > 0
        assert all(r["fasting_required_hours"] > 0 for r in res.rows)
        assert all(r["preparation_instructions"] for r in res.rows)
        assert any("verbatim" in n for n in res.notes)

    def test_empty_result_tells_model_not_to_guess(self):
        res = find_doctors(specialty="Veterinary Dentistry")
        assert res.row_count == 0
        assert any("do not suggest one anyway" in n.lower() for n in res.notes)

    def test_symptom_maps_to_specialty_without_diagnosing(self):
        res = suggest_specialty("I have chest pain")
        assert "Cardiology" in {r["specialty"] for r in res.rows}
        assert any("not a diagnosis" in n.lower() for n in res.notes)

    def test_unknown_symptom_falls_back_to_general_medicine(self):
        res = suggest_specialty("I feel a bit odd")
        assert res.row_count > 0
        assert any("General Medicine" in n for n in res.notes)
