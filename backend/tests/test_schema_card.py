"""The schema card is a prompt, so its examples are effectively few-shot code.

Wrong examples are worse than none - the model copies them. These tests execute
every SQL block in app/prompts/schema_card.md against the real database and
assert each one passes the guard and returns rows.
"""

from __future__ import annotations

import re

import pytest

from app.config import PROMPTS_DIR
from app.tools.db import fetch
from app.tools.sql_guard import validate_sql

CARD = PROMPTS_DIR / "schema_card.md"


def _sql_blocks() -> list[tuple[str, str]]:
    text = CARD.read_text(encoding="utf-8")
    # The label must stay on one line: with re.DOTALL a lazy `.+?` happily spans
    # paragraphs and swallows unrelated **bold** text further up the document.
    blocks = re.findall(
        r"\*\*([^\n]+?)\*\*[^\n]*\n"       # bold label, rest of its line
        # At most a few prose lines may sit between a label and its block.
        # Unbounded, a **bold** phrase from the prose far above claims the
        # first fence and every example is paired with the wrong label.
        r"(?:(?!```)[^\n]*\n){0,3}"
        r"```sql\n(.*?)```",               # the SQL itself
        text,
        re.S,
    )
    return [(label.strip().strip('"'), sql.strip()) for label, sql in blocks]


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    try:
        fetch("SELECT 1 AS ok")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres not reachable ({exc}). Run: docker compose up -d")


def test_card_exists_and_has_examples():
    assert CARD.exists()
    assert len(_sql_blocks()) >= 6


@pytest.mark.parametrize("label,sql", _sql_blocks(), ids=lambda v: v[:40] if isinstance(v, str) else v)
def test_example_passes_the_guard(label: str, sql: str):
    result = validate_sql(sql)
    assert result.safe, label


@pytest.mark.parametrize("label,sql", _sql_blocks(), ids=lambda v: v[:40] if isinstance(v, str) else v)
def test_example_executes_and_returns_rows(label: str, sql: str):
    rows, _ = fetch(sql)
    assert rows, f"example produced no rows: {label}"


def test_tuesday_example_includes_the_daily_doctor():
    """The headline example must actually defuse trap 1."""
    sql = dict(_sql_blocks())["Which doctors are available on Tuesday?"]
    rows, _ = fetch(sql)
    assert "Dr. Duminda Pathirana" in {r["doctor_name"] for r in rows}


def test_directory_example_includes_sessionless_doctors():
    sql = dict(_sql_blocks())["List all doctors with their consultation fees."]
    rows, _ = fetch(sql)
    assert len(rows) == 11
    assert sum(1 for r in rows if not r["has_published_sessions"]) == 2


def test_pap_smear_example_returns_three_packages():
    sql = dict(_sql_blocks())["Which package includes a Pap Smear?"]
    rows, _ = fetch(sql)
    assert len({r["package_name"] for r in rows}) == 3


def test_de_silva_example_finds_the_professor():
    sql = dict(_sql_blocks())["What is Prof. De Silva's fee?"]
    rows, _ = fetch(sql)
    assert rows[0]["doctor_name"] == "Prof. Arjuna De Silva"
    assert rows[0]["consultation_fee"] == 4200.00


def test_every_documented_value_list_matches_the_database():
    """Guards against the card's inlined distinct values going stale."""
    text = CARD.read_text(encoding="utf-8")

    days, _ = fetch("SELECT DISTINCT day_of_week AS d FROM channeling_sessions")
    for row in days:
        assert row["d"] in text, f"day_of_week value {row['d']!r} missing from schema card"

    cats, _ = fetch("SELECT DISTINCT category AS c FROM lab_tests")
    for row in cats:
        assert row["c"] in text, f"lab category {row['c']!r} missing from schema card"

    specs, _ = fetch("SELECT name FROM specialties")
    for row in specs:
        assert row["name"] in text, f"specialty {row['name']!r} missing from schema card"
