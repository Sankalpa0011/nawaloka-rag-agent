"""Curated SQL tools - the agent's primary path to structured data.

Design
------
Free-form Text-to-SQL alone fails on this dataset in ways that look confident:
the literal ``'Daily'`` weekday, two doctors with no sessions at all, and a
package-contents column whose item names deliberately do not match
``lab_tests.test_name``. Curated tools alone cannot cover the long tail
("which test has the longest turnaround under LKR 3000?").

So this is a hybrid: these six parameterised tools handle the common cases and
are the default; ``app/tools/text2sql.py`` is the guarded fallback. Crucially
**both go through the same trap-proof views**, so the fallback inherits the
same correctness rather than relying on the model having read a warning.

The ``notes`` field
-------------------
Every result carries deterministic, machine-generated notes - for example
"Dr. Duminda Pathirana's session is recorded as 'Daily' and therefore includes
Tuesday." Telling the answer model a fact is far more reliable than hoping it
infers one from raw rows, and it means the explanation of a trap is generated
by the same code that handles it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.logging_conf import get_logger
from app.tools.db import fetch

log = get_logger(__name__)

WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)

# pg_trgm similarity floor for fuzzy name matching. 0.3 tolerates honorifics,
# spelling drift and partial surnames without matching unrelated doctors.
NAME_SIMILARITY = 0.3


@dataclass
class SqlToolResult:
    tool: str
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    sql_executed: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "row_count": self.row_count,
            "rows": self.rows,
            "notes": self.notes,
            "sql_executed": " ".join(self.sql_executed.split()),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


def _normalise_weekday(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    aliases = {
        "mon": "Monday", "tue": "Tuesday", "tues": "Tuesday", "wed": "Wednesday",
        "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
        "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
        "today": None, "tomorrow": None,
    }
    for day in WEEKDAYS:
        if v == day.lower():
            return day
    return aliases.get(v)


def _run(tool: str, sql: str, params: dict[str, Any]) -> SqlToolResult:
    try:
        rows, ms = fetch(sql, params)
        return SqlToolResult(
            tool=tool, rows=rows, row_count=len(rows),
            sql_executed=sql, params=params, elapsed_ms=ms,
        )
    except Exception as exc:
        log.error("sql_tool.failed", tool=tool, error=str(exc))
        return SqlToolResult(
            tool=tool, sql_executed=sql, params=params,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# 1. Doctors
# ---------------------------------------------------------------------------
def find_doctors(
    specialty: str | None = None,
    name: str | None = None,
    department: str | None = None,
    weekday: str | None = None,
    max_fee: float | Decimal | None = None,
    include_schedule: bool = False,
    limit: int = 20,
) -> SqlToolResult:
    """Find doctors by specialty, name, department, availability or fee.

    Weekday filtering goes through ``v_sessions_expanded`` so a doctor recorded
    as available 'Daily' matches every weekday.
    """
    params: dict[str, Any] = {"limit": min(limit, 50)}
    where: list[str] = []
    day = _normalise_weekday(weekday)

    if specialty:
        where.append(
            "(d.specialty ILIKE :specialty OR similarity(d.specialty, :specialty_raw) > :sim)"
        )
        params["specialty"] = f"%{specialty}%"
        params["specialty_raw"] = specialty
        params["sim"] = NAME_SIMILARITY
    if department:
        where.append("d.department ILIKE :department")
        params["department"] = f"%{department}%"
    if name:
        where.append(
            "(d.name_plain ILIKE :name OR d.doctor_name ILIKE :name "
            "OR similarity(d.name_plain, :name_raw) > :nsim)"
        )
        params["name"] = f"%{name}%"
        params["name_raw"] = name
        params["nsim"] = NAME_SIMILARITY
    if max_fee is not None:
        where.append("d.consultation_fee <= :max_fee")
        params["max_fee"] = float(max_fee)

    if day:
        where.append(
            "EXISTS (SELECT 1 FROM v_sessions_expanded s "
            "WHERE s.doctor_id = d.doctor_id AND s.weekday = :weekday)"
        )
        params["weekday"] = day

    clause = " AND ".join(where) if where else "TRUE"

    schedule_select = ""
    if include_schedule or day:
        schedule_select = """,
            (SELECT json_agg(json_build_object(
                    'day_of_week', s.day_of_week_raw,
                    'weekday', s.weekday,
                    'start_time', s.start_time,
                    'end_time', s.end_time,
                    'room_number', s.room_number,
                    'max_patients', s.max_patients,
                    'is_recurring_daily', s.is_recurring_daily)
                    ORDER BY s.weekday, s.start_time)
             FROM v_sessions_expanded s
             WHERE s.doctor_id = d.doctor_id
               AND (CAST(:weekday_opt AS text) IS NULL OR s.weekday = :weekday_opt)
            ) AS sessions"""
        # NB: CAST(... AS text) rather than `:weekday_opt::text`. SQLAlchemy's
        # text() binds on a leading colon, so `::` in a Postgres cast is parsed
        # as the start of another parameter and the statement fails to compile.
        params["weekday_opt"] = day

    sql = f"""
        SELECT d.doctor_id, d.doctor_name, d.specialty, d.department,
               d.qualifications, d.consultation_fee,
               d.session_count, d.has_published_sessions{schedule_select}
        FROM v_doctor_directory d
        WHERE {clause}
        ORDER BY d.consultation_fee ASC, d.doctor_name
        LIMIT :limit
    """

    result = _run("find_doctors", sql, params)
    if result.error:
        return result

    _annotate_doctors(result, weekday=day)
    if day:
        result.notes.insert(
            0,
            f"Availability for {day} was resolved through v_sessions_expanded, "
            "which expands the literal 'Daily' schedule value into every weekday.",
        )
    if not result.rows:
        result.notes.append(
            "No doctor in the database matches those criteria. Do not suggest one anyway."
        )
    return result


def get_doctor_schedule(doctor: str) -> SqlToolResult:
    """Full channeling schedule for one doctor, matched fuzzily by name."""
    sql = """
        SELECT d.doctor_id, d.doctor_name, d.specialty, d.department,
               d.consultation_fee, d.session_count, d.has_published_sessions,
               (SELECT json_agg(json_build_object(
                       'day_of_week', s.day_of_week_raw,
                       'weekday', s.weekday,
                       'start_time', s.start_time,
                       'end_time', s.end_time,
                       'room_number', s.room_number,
                       'max_patients', s.max_patients,
                       'is_recurring_daily', s.is_recurring_daily)
                       ORDER BY s.weekday, s.start_time)
                FROM v_sessions_expanded s WHERE s.doctor_id = d.doctor_id
               ) AS sessions,
               similarity(d.name_plain, :q) AS match_score
        FROM v_doctor_directory d
        WHERE d.name_plain ILIKE :like
           OR d.doctor_name ILIKE :like
           OR similarity(d.name_plain, :q) > :sim
        ORDER BY similarity(d.name_plain, :q) DESC
        LIMIT 5
    """
    result = _run(
        "get_doctor_schedule",
        sql,
        {"q": doctor, "like": f"%{doctor}%", "sim": NAME_SIMILARITY},
    )
    if result.error:
        return result

    if not result.rows:
        result.notes.append(
            f"No doctor matching '{doctor}' exists in the database. "
            "Do not invent a schedule."
        )
        return result

    _annotate_doctors(result)
    if len(result.rows) > 1:
        result.notes.append(
            f"'{doctor}' matched {len(result.rows)} doctors; they are ordered by "
            "match confidence. Ask the patient which one they mean if it is ambiguous."
        )
    return result


def _annotate_doctors(result: SqlToolResult, weekday: str | None = None) -> None:
    """Attach deterministic notes about the data hazards in doctor results."""
    no_sessions = [r["doctor_name"] for r in result.rows if not r.get("has_published_sessions")]
    if no_sessions:
        result.notes.append(
            f"{len(no_sessions)} doctor(s) have no published channeling sessions: "
            + ", ".join(no_sessions)
            + ". They practise at the hospital but no channeling times are listed, "
            "so do not state or imply any availability for them."
        )

    for row in result.rows:
        sessions = row.get("sessions") or []
        if any(s.get("is_recurring_daily") for s in sessions):
            day_phrase = f"and therefore includes {weekday}" if weekday else "meaning every day of the week"
            result.notes.append(
                f"{row['doctor_name']}'s schedule is recorded as 'Daily' {day_phrase}."
            )


# ---------------------------------------------------------------------------
# 2. Lab tests
# ---------------------------------------------------------------------------
def search_lab_tests(
    query: str | None = None,
    category: str | None = None,
    max_price: float | Decimal | None = None,
    fasting_required: bool | None = None,
    max_report_hours: int | None = None,
    limit: int = 20,
) -> SqlToolResult:
    """Search the priced laboratory test catalogue held in the database."""
    params: dict[str, Any] = {"limit": min(limit, 50)}
    where: list[str] = []

    if query:
        where.append(
            "(test_name ILIKE :q OR test_code ILIKE :q OR category ILIKE :q "
            "OR similarity(test_name, :q_raw) > :sim)"
        )
        params.update({"q": f"%{query}%", "q_raw": query, "sim": NAME_SIMILARITY})
    if category:
        where.append("category ILIKE :category")
        params["category"] = f"%{category}%"
    if max_price is not None:
        where.append("price <= :max_price")
        params["max_price"] = float(max_price)
    if fasting_required is True:
        where.append("fasting_required_hours > 0")
    elif fasting_required is False:
        where.append("fasting_required_hours = 0")
    if max_report_hours is not None:
        where.append("report_delivery_hours <= :max_report_hours")
        params["max_report_hours"] = int(max_report_hours)

    clause = " AND ".join(where) if where else "TRUE"
    sql = f"""
        SELECT id, test_code, test_name, category, price,
               fasting_required_hours, preparation_instructions,
               report_delivery_hours
        FROM lab_tests
        WHERE {clause}
        ORDER BY price ASC, test_name
        LIMIT :limit
    """

    result = _run("search_lab_tests", sql, params)
    if result.error:
        return result

    fasting = [r for r in result.rows if (r.get("fasting_required_hours") or 0) > 0]
    if fasting:
        detail = "; ".join(
            f"{r['test_name']} requires {r['fasting_required_hours']}h fasting"
            for r in fasting
        )
        result.notes.append(
            f"Fasting is required for {len(fasting)} of these tests: {detail}. "
            "Quote preparation_instructions verbatim - never paraphrase them."
        )
    if not result.rows:
        result.notes.append(
            "No priced test in the hospital database matches. The website lists a "
            "wider bookable catalogue without prices, so the test may still be "
            "available - offer to check rather than saying it is not offered."
        )
    return result


# ---------------------------------------------------------------------------
# 3. Health packages
# ---------------------------------------------------------------------------
def search_health_packages(
    query: str | None = None,
    includes_service: str | None = None,
    max_price: float | Decimal | None = None,
    target_audience: str | None = None,
    category: str | None = None,
    limit: int = 20,
) -> SqlToolResult:
    """Search health check packages, including inside their contents text.

    ``includes_service`` searches *within* the free-text
    ``included_tests_and_services`` column. That column is not joinable to
    ``lab_tests``: it says "Lipid Profile" where the test table says "Advanced
    Lipid & ApoB Profile".
    """
    params: dict[str, Any] = {"limit": min(limit, 50)}
    where: list[str] = []
    select_extra = ""

    if query:
        where.append(
            "(package_name ILIKE :q OR category ILIKE :q OR target_audience ILIKE :q "
            "OR included_tests_and_services ILIKE :q)"
        )
        params["q"] = f"%{query}%"
    if includes_service:
        where.append("included_tests_and_services ILIKE :svc")
        params["svc"] = f"%{includes_service}%"
        # Return the matched item verbatim so the answer can quote it.
        select_extra = """,
            (SELECT string_agg(vps.service_item, '; ')
             FROM v_package_services vps
             WHERE vps.package_id = hp.id
               AND vps.service_item ILIKE :svc) AS matched_services"""
    if max_price is not None:
        where.append("price <= :max_price")
        params["max_price"] = float(max_price)
    if target_audience:
        where.append("target_audience ILIKE :audience")
        params["audience"] = f"%{target_audience}%"
    if category:
        where.append("category ILIKE :category")
        params["category"] = f"%{category}%"

    clause = " AND ".join(where) if where else "TRUE"
    sql = f"""
        SELECT hp.id, hp.package_name, hp.category, hp.price,
               hp.target_audience, hp.included_tests_and_services{select_extra}
        FROM health_packages hp
        WHERE {clause}
        ORDER BY hp.price ASC, hp.package_name
        LIMIT :limit
    """

    result = _run("search_health_packages", sql, params)
    if result.error:
        return result

    if includes_service and result.rows:
        result.notes.append(
            f"Matched on the package's own contents text for '{includes_service}'. "
            "Package contents are free text and are NOT linked to the lab_tests "
            "table, so quote the wording shown rather than mapping it to a test name."
        )
    if not result.rows:
        result.notes.append(
            "No package in the database matches. Note the public website "
            "advertises some packages that are not in the database; the database "
            "is the system of record, so do not quote a package from memory."
        )
    return result


# ---------------------------------------------------------------------------
# 4. Specialties
# ---------------------------------------------------------------------------
def list_specialties() -> SqlToolResult:
    """All specialties and departments, with doctor counts and fee ranges."""
    sql = """
        SELECT s.id, s.name AS specialty, s.department,
               count(d.id)::int          AS doctor_count,
               min(d.consultation_fee)   AS min_fee,
               max(d.consultation_fee)   AS max_fee
        FROM specialties s
        LEFT JOIN doctors d ON d.specialty_id = s.id
        GROUP BY s.id, s.name, s.department
        ORDER BY s.name
        LIMIT 50
    """
    result = _run("list_specialties", sql, {})
    if not result.error:
        empty = [r["specialty"] for r in result.rows if not r["doctor_count"]]
        if empty:
            result.notes.append(
                "Specialties with no doctors currently listed: " + ", ".join(empty)
            )
    return result


# ---------------------------------------------------------------------------
# 5. Symptom -> specialty navigation (not diagnosis)
# ---------------------------------------------------------------------------
# Maps a described complaint to the department that normally sees it. This is
# navigational only: it never asserts a cause, and the guardrail layer adds the
# "not a diagnosis" framing.
SYMPTOM_SPECIALTY_HINTS: dict[str, tuple[str, ...]] = {
    "chest pain": ("Cardiology",),
    "heart": ("Cardiology", "Cardiac Anaesthesia & Critical Care"),
    "palpitation": ("Cardiology",),
    "headache": ("Neurosurgery", "General Medicine"),
    "migraine": ("Neurosurgery", "General Medicine"),
    "seizure": ("Neurosurgery",),
    "ear": ("ENT Surgery",),
    "nose": ("ENT Surgery",),
    "throat": ("ENT Surgery",),
    "hearing": ("ENT Surgery",),
    "sinus": ("ENT Surgery",),
    "cough": ("Respiratory & Chest Medicine",),
    "breathing": ("Respiratory & Chest Medicine",),
    "asthma": ("Respiratory & Chest Medicine",),
    "wheeze": ("Respiratory & Chest Medicine",),
    "stomach": ("General Surgery & Gastroenterology",),
    "abdominal": ("General Surgery & Gastroenterology",),
    "digestion": ("General Surgery & Gastroenterology",),
    "gastric": ("General Surgery & Gastroenterology",),
    "child": ("Paediatrics",),
    "baby": ("Paediatrics",),
    "infant": ("Paediatrics",),
    "joint": ("Rheumatology & Sports Medicine",),
    "arthritis": ("Rheumatology & Sports Medicine",),
    "back pain": ("Rheumatology & Sports Medicine", "Neurosurgery"),
    "sports injury": ("Rheumatology & Sports Medicine",),
    "scan": ("Radiology & Imaging",),
    "x-ray": ("Radiology & Imaging",),
    "mri": ("Radiology & Imaging",),
    "fever": ("General Medicine",),
    "diabetes": ("General Medicine",),
    "blood pressure": ("General Medicine",),
}


def suggest_specialty(complaint: str) -> SqlToolResult:
    """Map a described complaint to candidate specialties, then list doctors."""
    text = (complaint or "").lower()
    matched: list[str] = []
    for key, specialties in SYMPTOM_SPECIALTY_HINTS.items():
        if re.search(rf"\b{re.escape(key)}", text):
            matched.extend(s for s in specialties if s not in matched)

    if not matched:
        result = find_doctors(specialty="General Medicine", include_schedule=True)
        result.tool = "suggest_specialty"
        result.notes.insert(
            0,
            "No specific specialty mapped to this complaint, so General Medicine "
            "is offered as the starting point. Present this as navigation, not a "
            "diagnosis.",
        )
        return result

    sql = """
        SELECT d.doctor_id, d.doctor_name, d.specialty, d.department,
               d.qualifications, d.consultation_fee,
               d.session_count, d.has_published_sessions,
               (SELECT json_agg(json_build_object(
                       'day_of_week', s.day_of_week_raw,
                       'weekday', s.weekday,
                       'start_time', s.start_time,
                       'end_time', s.end_time,
                       'room_number', s.room_number,
                       'is_recurring_daily', s.is_recurring_daily)
                       ORDER BY s.weekday, s.start_time)
                FROM v_sessions_expanded s WHERE s.doctor_id = d.doctor_id
               ) AS sessions
        FROM v_doctor_directory d
        WHERE d.specialty = ANY(:specialties)
        ORDER BY d.consultation_fee ASC, d.doctor_name
        LIMIT 20
    """
    result = _run("suggest_specialty", sql, {"specialties": matched})
    if result.error:
        return result

    _annotate_doctors(result)
    result.notes.insert(
        0,
        f"'{complaint}' is normally seen by: {', '.join(matched)}. This is "
        "navigational guidance about which clinic to visit - it is not a "
        "diagnosis and must not be presented as one.",
    )
    return result


TOOLS = {
    "find_doctors": find_doctors,
    "get_doctor_schedule": get_doctor_schedule,
    "search_lab_tests": search_lab_tests,
    "search_health_packages": search_health_packages,
    "list_specialties": list_specialties,
    "suggest_specialty": suggest_specialty,
}
