"""The DDL half of the Text-to-SQL prompt, introspected live from the database
so it can't drift from schema changes. Hand-written hazard notes live in
schema_card.md; generated SQL still runs through the sqlglot guard as chatbot_ro.
"""

from __future__ import annotations

import re
from functools import lru_cache

from app.config import settings
from app.logging_conf import get_logger
from app.tools.db import _sqlalchemy_url

log = get_logger(__name__)

# Allowlist so introspection doesn't expose tables chatbot_ro can see but not query.
EXPOSED = (
    "v_doctor_directory",
    "v_sessions_expanded",
    "v_package_services",
    "specialties",
    "doctors",
    "channeling_sessions",
    "lab_tests",
    "health_packages",
)

# Strips Postgres's verbose identity-column clause, meaningless for writing a SELECT.
_IDENTITY_NOISE = re.compile(r" GENERATED (?:BY DEFAULT|ALWAYS) AS IDENTITY \([^)]*\)")

_FALLBACK = (
    "(Schema introspection unavailable - rely on the views and columns named "
    "in the notes below.)"
)


@lru_cache(maxsize=1)
def table_ddl() -> str:
    """CREATE TABLE for the eight relations the agent is allowed to query."""
    try:
        from langchain_community.utilities import SQLDatabase

        db = SQLDatabase.from_uri(
            _sqlalchemy_url(settings.database_ro_url),
            view_support=True,
            include_tables=list(EXPOSED),
            # Sample rows would leak the same data the answer is supposed to
            # come from a real query, and cost ~800 tokens a call.
            sample_rows_in_table_info=0,
        )
        ddl = _IDENTITY_NOISE.sub("", db.get_table_info())
        return ddl.strip()
    except Exception as exc:
        # Same call as _specialty_roster in sql_agent: a degraded prompt beats a
        # dead endpoint. The hazard notes still carry the view names.
        log.warning("schema.introspection_failed", error=str(exc)[:200])
        return _FALLBACK
