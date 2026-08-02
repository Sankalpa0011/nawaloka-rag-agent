"""Validates the routing eval set itself.

The router does not exist yet - scoring against it comes later, in
evals/test_routing.py. This file guards the *dataset*, because an eval set is
only worth the number it produces:

* labels must come from the agreed taxonomy
* classes must be balanced enough that accuracy is not dominated by one route
* every trap must be represented
* the factual `must_include` expectations must match the real database, so the
  eval cannot assert something the data does not support
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.tools.db import fetch

EVAL_PATH = Path(__file__).parent / "routing_set.jsonl"

VALID_ROUTES = {
    "SQL",        # relational data only
    "VECTOR",     # scraped web knowledge only
    "HYBRID",     # genuinely needs both
    "EMERGENCY",  # hard intercept, no LLM
    "REFUSE",     # clinical advice the assistant must not give
    "OUT_OF_SCOPE",
    "SMALLTALK",
}

REQUIRED_TRAPS = {
    "trap:daily",
    "trap:no_sessions",
    "trap:package_text",
    "trap:honorific",
    "trap:varchar_overflow",
}


def load_cases() -> list[dict]:
    cases = []
    for i, line in enumerate(EVAL_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover
            pytest.fail(f"line {i} of routing_set.jsonl is not valid JSON: {exc}")
    return cases


CASES = load_cases()


def test_eval_set_is_a_reasonable_size():
    assert len(CASES) >= 45, f"only {len(CASES)} cases; too few to report accuracy from"


def test_ids_are_unique():
    ids = [c["id"] for c in CASES]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    assert not dupes, f"duplicate ids: {dupes}"


def test_queries_are_unique():
    queries = [c["query"].strip().lower() for c in CASES]
    dupes = [q for q, n in Counter(queries).items() if n > 1]
    assert not dupes, f"duplicate queries: {dupes}"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_case_has_required_fields(case: dict):
    assert case.get("id")
    assert case.get("query", "").strip()
    assert case.get("rationale", "").strip(), "every label needs a stated reason"
    assert case["expected_route"] in VALID_ROUTES, case["expected_route"]


def test_class_balance():
    """No single route may dominate, or accuracy becomes uninformative."""
    counts = Counter(c["expected_route"] for c in CASES)
    for route in ("SQL", "VECTOR", "HYBRID"):
        assert counts[route] >= 10, f"{route} has only {counts[route]} cases"
    for route in ("EMERGENCY", "REFUSE", "OUT_OF_SCOPE", "SMALLTALK"):
        assert counts[route] >= 3, f"{route} has only {counts[route]} cases"
    assert max(counts.values()) / len(CASES) < 0.40, f"one route dominates: {counts}"


def test_every_known_trap_is_covered():
    tags = {t for c in CASES for t in c.get("tags", [])}
    missing = REQUIRED_TRAPS - tags
    assert not missing, f"eval set does not exercise: {missing}"


def test_multiturn_cases_carry_history():
    ctx = [c for c in CASES if "multiturn" in c.get("tags", [])]
    assert len(ctx) >= 3, "coreference handling is a graded bonus; test it"
    for c in ctx:
        assert c.get("history"), f"{c['id']} is tagged multiturn but has no history"
        for turn in c["history"]:
            assert len(turn) == 2, f"{c['id']} history turns must be [user, assistant]"


def test_emergency_cases_expect_the_ambulance_number():
    for c in CASES:
        if c["expected_route"] == "EMERGENCY":
            assert "1990" in (c.get("must_include") or []), (
                f"{c['id']} should require the Suwa Seriya number in the response"
            )


# ---------------------------------------------------------------------------
# The important one: are the expected facts actually true of the data?
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def db_available():
    try:
        fetch("SELECT 1")
        return True
    except Exception:  # pragma: no cover
        pytest.skip("Postgres not reachable. Run: docker compose up -d")


NUMERIC_CLAIMS = [
    ("sql-001", "3500", "SELECT count(*) n FROM doctors WHERE consultation_fee = 3500"),
    ("sql-003", "3200", "SELECT count(*) n FROM lab_tests WHERE price = 3200"),
    ("sql-006", "4200", "SELECT count(*) n FROM doctors WHERE consultation_fee = 4200"),
    ("sql-008", "6450", "SELECT count(*) n FROM health_packages WHERE price = 6450"),
    ("sql-011", "24", "SELECT count(*) n FROM lab_tests WHERE test_name ILIKE '%Vitamin D3%' AND report_delivery_hours = 24"),
    ("sql-016", "16500", "SELECT count(*) n FROM health_packages WHERE price = 16500"),
    ("hyb-003", "4800", "SELECT count(*) n FROM lab_tests WHERE price = 4800"),
    ("hyb-006", "3000", "SELECT count(*) n FROM doctors WHERE consultation_fee = 3000"),
    ("hyb-009", "25150", "SELECT count(*) n FROM health_packages WHERE price = 25150"),
    ("ctx-001", "4000", "SELECT count(*) n FROM doctors WHERE name ILIKE '%Priyadarshan%' AND consultation_fee = 4000"),
    ("ctx-003", "8", "SELECT count(*) n FROM lab_tests WHERE test_name ILIKE '%B12%' AND fasting_required_hours = 8"),
]


@pytest.mark.parametrize("case_id,value,sql", NUMERIC_CLAIMS, ids=[c[0] for c in NUMERIC_CLAIMS])
def test_expected_value_exists_in_database(db_available, case_id: str, value: str, sql: str):
    case = next(c for c in CASES if c["id"] == case_id)
    assert value in (case.get("must_include") or []), (
        f"{case_id} no longer expects {value}; update this test with it"
    )
    rows, _ = fetch(sql)
    assert rows[0]["n"] >= 1, f"{case_id} expects {value} but no such row exists"


def test_doctor_names_referenced_actually_exist(db_available):
    referenced = {"Pathirana", "Priyadarshan", "Samarasinghe"}
    for name in referenced:
        rows, _ = fetch("SELECT count(*) n FROM doctors WHERE name ILIKE :p", {"p": f"%{name}%"})
        assert rows[0]["n"] == 1, f"eval set references {name!r}, which is not a unique doctor"


def test_package_names_referenced_actually_exist(db_available):
    for fragment in ("Well Woman", "Executive Female"):
        rows, _ = fetch(
            "SELECT count(*) n FROM health_packages WHERE package_name ILIKE :p",
            {"p": f"%{fragment}%"},
        )
        assert rows[0]["n"] >= 1, f"no package matches {fragment!r}"


def test_summary(capsys):
    """Not an assertion - prints the distribution for the README."""
    counts = Counter(c["expected_route"] for c in CASES)
    tags = Counter(t for c in CASES for t in c.get("tags", []))
    with capsys.disabled():
        print(f"\n  routing_set.jsonl: {len(CASES)} cases")
        for route, n in counts.most_common():
            print(f"    {route:<14}{n:>3}  ({n/len(CASES):.0%})")
        traps = {t: n for t, n in tags.items() if t.startswith("trap:")}
        print(f"    traps covered: {traps}")
