"""Adversarial tests for the generated-SQL guard.

Every ATTACK below is something a prompt-injected model could plausibly emit.
The guard must reject all of them, and must accept the LEGITIMATE queries the
agent genuinely needs - a guard that blocks real work gets disabled.
"""

from __future__ import annotations

import pytest

from app.tools.sql_guard import SqlGuardError, validate_sql

# ---------------------------------------------------------------------------
# Must be REJECTED
# ---------------------------------------------------------------------------
ATTACKS = [
    pytest.param("DROP TABLE doctors", id="bare-drop"),
    pytest.param("DELETE FROM doctors", id="bare-delete"),
    pytest.param("UPDATE doctors SET consultation_fee = 0", id="bare-update"),
    pytest.param("INSERT INTO doctors (name) VALUES ('x')", id="bare-insert"),
    pytest.param("TRUNCATE TABLE doctors", id="truncate"),
    pytest.param("ALTER TABLE doctors ADD COLUMN x int", id="alter"),
    pytest.param("CREATE TABLE evil (x int)", id="create"),
    pytest.param("GRANT ALL ON doctors TO chatbot_ro", id="grant"),

    # Statement stacking - the classic
    pytest.param("SELECT 1; DROP TABLE doctors", id="stacked-drop"),
    pytest.param("SELECT * FROM doctors; DELETE FROM doctors", id="stacked-delete"),
    pytest.param("SELECT 1;;DROP TABLE doctors;", id="stacked-empty"),

    # Comment-obfuscated
    pytest.param("SELECT * FROM doctors -- \nUNION SELECT * FROM pg_shadow", id="comment-union-catalog"),
    pytest.param("SELECT 1 /* harmless */; /* now */ DROP TABLE doctors", id="comment-stacked"),

    # Writes hidden inside CTEs - regex-based filters miss these
    pytest.param(
        "WITH x AS (DELETE FROM doctors RETURNING *) SELECT * FROM x",
        id="cte-delete",
    ),
    pytest.param(
        "WITH x AS (UPDATE doctors SET consultation_fee=0 RETURNING id) SELECT * FROM x",
        id="cte-update",
    ),
    pytest.param(
        "WITH x AS (INSERT INTO doctors (name) VALUES ('e') RETURNING id) SELECT * FROM x",
        id="cte-insert",
    ),

    # System catalogue / credential access
    pytest.param("SELECT * FROM pg_shadow", id="pg-shadow"),
    pytest.param("SELECT * FROM information_schema.tables", id="information-schema"),
    pytest.param("SELECT usename, passwd FROM pg_catalog.pg_user", id="pg-user"),

    # Server-side file access and DoS
    pytest.param("SELECT pg_read_file('/etc/passwd')", id="read-file"),
    pytest.param("SELECT pg_sleep(30)", id="pg-sleep"),
    pytest.param("SELECT lo_import('/etc/passwd')", id="lo-import"),
    pytest.param("SELECT dblink('host=evil', 'SELECT 1')", id="dblink"),

    # Data exfiltration from application tables the agent must never see
    pytest.param("SELECT * FROM routing_decisions", id="routing-log"),
    pytest.param("SELECT * FROM user_memory", id="user-memory"),
    pytest.param(
        "SELECT d.name FROM doctors d JOIN user_memory u ON true", id="join-user-memory"
    ),

    # Table creation without a CREATE node
    pytest.param("SELECT * INTO evil FROM doctors", id="select-into"),

    # Row locking
    pytest.param("SELECT * FROM doctors FOR UPDATE", id="for-update"),

    # Unknown relations
    pytest.param("SELECT * FROM secrets", id="unknown-table"),

    # Nonsense
    pytest.param("this is not sql at all !!", id="unparseable"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
]


@pytest.mark.parametrize("sql", ATTACKS)
def test_attack_is_rejected(sql: str):
    with pytest.raises(SqlGuardError):
        validate_sql(sql)


# ---------------------------------------------------------------------------
# Must be ACCEPTED
# ---------------------------------------------------------------------------
LEGITIMATE = [
    pytest.param("SELECT * FROM doctors", id="simple-select"),
    pytest.param(
        "SELECT doctor_name, consultation_fee FROM v_doctor_directory "
        "WHERE specialty = 'Cardiology' ORDER BY consultation_fee",
        id="directory-view",
    ),
    pytest.param(
        "SELECT DISTINCT doctor_name FROM v_sessions_expanded WHERE weekday = 'Tuesday'",
        id="sessions-view",
    ),
    pytest.param(
        "SELECT package_name FROM v_package_services WHERE service_item ILIKE '%pap smear%'",
        id="package-services",
    ),
    pytest.param(
        "WITH cheap AS (SELECT * FROM lab_tests WHERE price < 1000) "
        "SELECT test_name FROM cheap ORDER BY price",
        id="read-only-cte",
    ),
    pytest.param(
        "SELECT s.name, count(*) FROM doctors d "
        "JOIN specialties s ON s.id = d.specialty_id GROUP BY s.name HAVING count(*) > 1",
        id="join-group-having",
    ),
    pytest.param(
        "SELECT test_name FROM lab_tests WHERE category = 'Biochemistry' "
        "UNION SELECT test_name FROM lab_tests WHERE price > 5000",
        id="union",
    ),
    pytest.param(
        "SELECT name FROM doctors WHERE id IN (SELECT doctor_id FROM channeling_sessions)",
        id="subquery",
    ),
]


@pytest.mark.parametrize("sql", LEGITIMATE)
def test_legitimate_query_is_allowed(sql: str):
    result = validate_sql(sql)
    assert result.safe
    assert result.sql


# ---------------------------------------------------------------------------
# LIMIT clamping
# ---------------------------------------------------------------------------
def test_limit_is_injected_when_absent():
    result = validate_sql("SELECT * FROM doctors", max_rows=200)
    assert result.limit_applied == 200
    assert "LIMIT 200" in result.sql.upper()


def test_oversized_limit_is_clamped():
    result = validate_sql("SELECT * FROM doctors LIMIT 100000", max_rows=200)
    assert result.limit_applied == 200
    assert "100000" not in result.sql


def test_small_limit_is_preserved():
    result = validate_sql("SELECT * FROM doctors LIMIT 5", max_rows=200)
    assert result.limit_applied == 5


def test_allowlist_is_reported():
    result = validate_sql("SELECT * FROM doctors JOIN specialties ON true")
    assert set(result.relations) == {"doctors", "specialties"}


def test_cte_name_is_not_treated_as_unknown_table():
    # A CTE alias is a relation reference but must not trip the allowlist.
    result = validate_sql(
        "WITH x AS (SELECT * FROM doctors) SELECT * FROM x"
    )
    assert result.safe
