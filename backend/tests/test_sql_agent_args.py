"""Argument hygiene for the curated SQL tools.

The tool-selection model is reliable about *which* tool to call and much less
reliable about what to put in it. Both failures seen in practice were silent:
the tool ran, Postgres was happy, and the agent reported with total confidence
that nothing existed. These tests pin the two guards that catch that, because a
confidently wrong "no such doctor" is the worst output this system can produce.

Pure unit tests - no database, no LLM.
"""

from __future__ import annotations

import pytest

from app.agent.nodes.sql_agent import _coerce_args

DOCTOR_ARGS = {
    "specialty", "name", "department", "weekday", "max_fee",
    "include_schedule", "limit",
}


class TestStopValues:
    """Values that match every row, so they can only destroy a result set."""

    @pytest.mark.parametrize("value", [
        "Nawaloka Hospitals",
        "Nawaloka",
        "nawaloka hospitals plc",
        "the hospital",
        "all",
        "any",
        "N/A",
    ])
    def test_non_discriminating_values_are_dropped(self, value: str):
        cleaned, problems = _coerce_args({"department": value}, DOCTOR_ARGS)
        assert cleaned == {}, f"{value!r} must not become a WHERE clause"
        assert problems

    def test_hospital_name_regression(self):
        """Regression: "which doctors practise here?" returned nothing.

        The model read the institution out of the question and filed it as a
        filter, giving ``department ILIKE '%Nawaloka Hospitals%'`` - zero rows
        from a directory of eleven doctors, and an answer that flatly denied
        having the information.
        """
        cleaned, _ = _coerce_args({"department": "Nawaloka Hospitals"}, DOCTOR_ARGS)
        assert "department" not in cleaned

    @pytest.mark.parametrize("key,value", [
        ("specialty", "Cardiology"),
        ("department", "Paediatrics"),
        ("name", "Pathirana"),
        ("weekday", "Tuesday"),
    ])
    def test_real_filters_survive(self, key: str, value: str):
        """The guard must not eat genuine values - that would be the worse bug."""
        cleaned, problems = _coerce_args({key: value}, DOCTOR_ARGS)
        assert cleaned == {key: value}
        assert not problems


class TestTypeCoercion:
    def test_boolean_where_a_number_belongs_is_dropped(self):
        """Regression: ``max_fee: true`` meaning "yes, tell me the fee".

        Coerced naively that is ``fee <= 1.0``: no rows, and the agent then
        reports the doctor does not exist.
        """
        cleaned, problems = _coerce_args({"name": "Dr X", "max_fee": True}, DOCTOR_ARGS)
        assert cleaned == {"name": "Dr X"}
        assert any("max_fee" in p for p in problems)

    def test_numeric_strings_are_accepted(self):
        cleaned, _ = _coerce_args({"max_fee": "3000"}, DOCTOR_ARGS)
        assert cleaned["max_fee"] == 3000.0

    @pytest.mark.parametrize("value,expected", [
        (True, True), ("true", True), ("yes", True), ("false", False), ("no", False),
    ])
    def test_boolean_switches(self, value, expected):
        cleaned, _ = _coerce_args({"include_schedule": value}, DOCTOR_ARGS)
        assert cleaned["include_schedule"] is expected

    def test_invented_parameter_names_are_dropped(self):
        cleaned, problems = _coerce_args({"hospital_name": "Nawaloka"}, DOCTOR_ARGS)
        assert cleaned == {}
        assert any("not a parameter" in p for p in problems)

    def test_empty_values_are_dropped_quietly(self):
        cleaned, problems = _coerce_args({"specialty": "", "name": None}, DOCTOR_ARGS)
        assert cleaned == {}
        assert not problems
