"""Validation for model-generated SQL.

The agent's SQL tool is the one surface an attacker can reach through prompt
injection, so generated SQL is parsed and inspected as an **abstract syntax
tree** before execution - never regex-matched. String matching on "DROP" or
"DELETE" is trivially defeated (comments, casing, `/*!*/`, string literals
containing keywords, CTEs wrapping a write), whereas an AST cannot be talked
out of what it is.

Layers, in order:

1. Parse with sqlglot in the Postgres dialect. Unparseable SQL is rejected.
2. Exactly one statement - kills `SELECT 1; DROP TABLE doctors`.
3. The root must be a SELECT (optionally with CTEs) or a set operation.
4. No write/DDL node anywhere in the tree, at any depth.
5. Every referenced table must be on the allowlist.
6. No system catalogues, no server-side file access functions.
7. A LIMIT is injected or clamped by rewriting the AST, not by string
   concatenation.

This is defence in depth, not the only defence: the connection role itself
cannot write (db/init/004_ro_role.sh).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from app.logging_conf import get_logger
from app.tools.db import MAX_ROWS

log = get_logger(__name__)

DIALECT = "postgres"

# Tables and views the agent may read. Note routing_decisions and user_memory
# are absent: they hold conversation content, and the database role denies them
# too.
ALLOWED_RELATIONS: frozenset[str] = frozenset(
    {
        "specialties",
        "doctors",
        "channeling_sessions",
        "lab_tests",
        "health_packages",
        "v_sessions_expanded",
        "v_doctor_directory",
        "v_package_services",
    }
)

# Any of these appearing anywhere in the tree is fatal.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Grant,
    exp.Command,      # anything sqlglot could not classify: SET, COPY, VACUUM...
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Use,
)

# Functions that read the filesystem, run shell commands or leak server state.
FORBIDDEN_FUNCTIONS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_connect", "pg_sleep",
    "query_to_xml", "pg_terminate_backend", "pg_cancel_backend",
    "set_config", "pg_reload_conf", "current_setting",
}

FORBIDDEN_SCHEMA_RE = re.compile(
    r"\b(pg_catalog|information_schema|pg_class|pg_tables|pg_shadow|pg_authid|pg_user|pg_roles)\b",
    re.I,
)


class SqlGuardError(ValueError):
    """Raised when generated SQL is not safe to execute."""


@dataclass
class GuardResult:
    safe: bool
    sql: str                      # rewritten, limit-clamped SQL
    reason: str = ""
    relations: list[str] = field(default_factory=list)
    limit_applied: int | None = None


def _relation_names(tree: exp.Expression) -> tuple[set[str], set[str]]:
    """Return (referenced_relations, cte_names)."""
    ctes = {
        c.alias_or_name.lower()
        for c in tree.find_all(exp.CTE)
        if c.alias_or_name
    }
    tables: set[str] = set()
    for tbl in tree.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if not name:
            continue
        # Reject schema-qualified access to anything but public.
        db = (tbl.text("db") or "").lower()
        if db and db != "public":
            tables.add(f"{db}.{name}")
        else:
            tables.add(name)
    return tables, ctes


def validate_sql(sql: str, max_rows: int = MAX_ROWS) -> GuardResult:
    """Validate and normalise a generated SELECT. Raises SqlGuardError."""
    if not sql or not sql.strip():
        raise SqlGuardError("empty statement")

    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except Exception as exc:
        raise SqlGuardError(f"could not parse SQL: {exc}") from exc

    if len(statements) != 1:
        raise SqlGuardError(
            f"expected exactly one statement, found {len(statements)}"
        )

    tree = statements[0]

    # --- 3. must be a read ---------------------------------------------------
    if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)):
        raise SqlGuardError(
            f"only SELECT statements are permitted, got {type(tree).__name__.upper()}"
        )

    # --- 4. no write or DDL node, at any depth -------------------------------
    for node_type in FORBIDDEN_NODES:
        found = tree.find(node_type)
        if found is not None:
            raise SqlGuardError(
                f"statement contains a forbidden {node_type.__name__.upper()} operation"
            )

    # SELECT ... INTO creates a table without an exp.Create node.
    if tree.find(exp.Into) is not None:
        raise SqlGuardError("SELECT ... INTO is not permitted")
    # FOR UPDATE takes row locks.
    if tree.find(exp.Lock) is not None:
        raise SqlGuardError("locking clauses are not permitted")

    # --- 6a. no system catalogues -------------------------------------------
    if FORBIDDEN_SCHEMA_RE.search(sql):
        raise SqlGuardError("access to system catalogues is not permitted")

    # --- 6b. no dangerous functions -----------------------------------------
    for fn in tree.find_all(exp.Anonymous):
        name = (fn.this or "").lower() if isinstance(fn.this, str) else ""
        if name in FORBIDDEN_FUNCTIONS:
            raise SqlGuardError(f"function {name}() is not permitted")
    lowered = sql.lower()
    for fn in FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{re.escape(fn)}\s*\(", lowered):
            raise SqlGuardError(f"function {fn}() is not permitted")

    # --- 5. table allowlist --------------------------------------------------
    tables, ctes = _relation_names(tree)
    unknown = sorted(t for t in tables if t not in ALLOWED_RELATIONS and t not in ctes)
    if unknown:
        raise SqlGuardError(
            "query references relation(s) outside the allowlist: "
            + ", ".join(unknown)
            + ". Allowed: "
            + ", ".join(sorted(ALLOWED_RELATIONS))
        )

    # --- 7. clamp LIMIT by rewriting the tree --------------------------------
    limit_applied: int | None = None
    if isinstance(tree, exp.Select):
        existing = tree.args.get("limit")
        current: int | None = None
        if existing is not None:
            try:
                current = int(existing.expression.this)
            except (AttributeError, TypeError, ValueError):
                current = None
        if current is None or current > max_rows:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
            limit_applied = max_rows
        else:
            limit_applied = current

    safe_sql = tree.sql(dialect=DIALECT, pretty=False)
    return GuardResult(
        safe=True,
        sql=safe_sql,
        relations=sorted(tables),
        limit_applied=limit_applied,
    )


def is_safe(sql: str) -> bool:
    try:
        validate_sql(sql)
        return True
    except SqlGuardError:
        return False
