"""The curated SQL tools - the agent's main path to structured data, with
text2sql.py as fallback for the long tail. Both go through the same views, and
results carry generated `notes` spelling out data hazards (e.g. a 'Daily'
weekday session) rather than leaving the model to infer them from raw rows.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from app.config import settings
from app.logging_conf import get_logger
from app.tools.db import fetch

log = get_logger(__name__)

WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)

NAME_SIMILARITY = settings.sql_name_similarity

SPECIALTY_THRESHOLD = settings.sql_specialty_threshold

# shorter than this and a word isn't worth matching on
_MIN_WORD_LEN = settings.sql_min_word_len


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
    # True when an empty search was re-run without the free-text filter; rows
    # are then the whole catalogue, not a match.
    relaxed: bool = False
    # The model's argument dict (not SQL bind params), for naming criteria in the empty-result note.
    requested_filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Single place the empty-result rule is stated, so every tool gets it.
        notes = list(self.notes)
        if self.row_count == 0 and not self.relaxed and not self.error:
            notes.append(no_match_note(self.requested_filters))
        return {
            "tool": self.tool,
            "row_count": self.row_count,
            "rows": self.rows,
            "notes": notes,
            "sql_executed": " ".join(self.sql_executed.split()),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "relaxed": self.relaxed,
        }


def _first(values) -> str:
    for v in values:
        if v:
            return str(v)
    return ""


# Zero rows means no *record* in this system of record - not that the hospital
# doesn't offer the thing (plenty of units are on the website but unpriced
# here). Must say what the rows are NOT before what they are, or the model
# presents a broadened/absent result as an answer.
def no_match_note(filters: dict[str, Any] | None = None) -> str:
    applied = ", ".join(
        f"{k}={v}"
        for k, v in (filters or {}).items()
        if k != "limit" and v not in (None, "", False)
    )
    return (
        "0 rows found matching these specific filters"
        + (f" ({applied})" if applied else "")
        + ". This database is the system of record for prices, schedules and "
        "listings, and it holds NO record matching this. Say that plainly. Do NOT "
        "invent details, and do NOT offer a different doctor, test, specialty or "
        "package as though it were the one asked for. Note this means there is no "
        "priced or scheduled record - not necessarily that the hospital does not "
        "provide it. If documents describing it are supplied, use them to say what "
        "it is, and attach no price, fee or time to what they say. Otherwise offer "
        "to have the hospital confirm. Do not name the database, the query or the "
        "filter values to the patient."
    )


def _relaxed_note(kind: str, query: str) -> str:
    return (
        f"IMPORTANT: nothing in the database matched '{query}'. The rows below are "
        f"the full {kind} list, NOT an answer to what was asked. Say plainly that "
        f"there is no {kind} by that name first, then offer these as what is "
        "available. Never present them as if they matched."
    )


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


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if len(w) >= settings.sql_min_word_len:
            out.append(w)
    return out


# Returns (name -> its distinctive words, words shared by more than one name).
# Read off the table, so adding a specialty or renaming a department can't leave
# a stale list behind.
@lru_cache(maxsize=2)
def _vocabulary(column: str) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    try:
        rows, _ = fetch(f"SELECT DISTINCT {column} AS name FROM specialties")
    except Exception as exc:
        log.warning("vocabulary.unavailable", column=column, error=str(exc)[:160])
        return {}, frozenset()

    per_name = {}
    for r in rows:
        per_name[str(r["name"])] = set(_words(str(r["name"])))

    counts = Counter()
    for words in per_name.values():
        counts.update(words)
    shared = frozenset(w for w, n in counts.items() if n > 1)

    distinctive = {}
    for name, words in per_name.items():
        # if every word is shared there's nothing better to offer, keep them all
        distinctive[name] = frozenset((words - shared) or words)
    return distinctive, shared


def _specialty_tokens() -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    return _vocabulary("name")


# Prefixes count as near-exact because that's how people shorten these -
# "neuro" for "neurosurgery". Plain edit distance scores that pair 0.59 and
# would reject it. The length check is what stops "ent" claiming "dental".
def _word_match(query_word: str, specialty_word: str) -> float:
    if query_word == specialty_word:
        return 1.0
    if (
        len(query_word) >= settings.sql_prefix_min_len
        and specialty_word.startswith(query_word)
    ):
        return settings.sql_prefix_match_score
    return SequenceMatcher(None, query_word, specialty_word).ratio()


def _resolve_against(text: str, column: str) -> str | None:
    """Which single name in `column` a phrase means, or None to fall through to a literal ILIKE."""
    q = (text or "").strip()
    if not q:
        return None

    tokens, shared = _vocabulary(column)
    if not tokens:
        return None

    # Literal substring needs no guessing. And if several names contain the
    # phrase then the user did mean all of them, so ILIKE is right anyway.
    for name in tokens:
        if q.lower() in name.lower():
            return None

    # Drop words every name shares - those are what made "eye surgery" look like
    # ENT Surgery. Nothing left means the phrase is generic, so give up.
    query_words = []
    for w in _words(q):
        if w not in shared:
            query_words.append(w)
    if not query_words:
        return None

    scored = []
    for name, name_words in tokens.items():
        best = 0.0
        for qw in query_words:
            for nw in name_words:
                score = _word_match(qw, nw)
                if score > best:
                    best = score
        scored.append((best, name))
    scored.sort(reverse=True)

    top_score, top_name = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    # a tie means we genuinely can't tell them apart, so don't guess
    if top_score < settings.sql_specialty_threshold or top_score == runner_up:
        return None
    return top_name


def _resolve_specialty(text: str) -> str | None:
    return _resolve_against(text, "name")


def _resolve_department(text: str) -> str | None:
    return _resolve_against(text, "department")


def reset_specialty_cache() -> None:
    _vocabulary.cache_clear()

def _name_clause(name: str, params: dict[str, Any], *, broad: bool = True,
                 prefix: str = "name") -> str:
    column = "d.name_plain"
    parts = [
        f"({column} ILIKE :{prefix} OR d.doctor_name ILIKE :{prefix} "
        f"OR similarity({column}, :{prefix}_raw) > :{prefix}_sim)"
    ]
    params[prefix] = f"%{name}%"
    params[f"{prefix}_raw"] = name
    params[f"{prefix}_sim"] = settings.sql_name_similarity

    if broad:
        # Honorifics are stripped by name_plain, so they are noise as query
        # words too - they would match every doctor in the table.
        words = [w for w in _words(name) if w not in {"dr", "prof", "doctor"}]
        if words:
            params[f"{prefix}_wsim"] = settings.sql_name_word_similarity
        for i, word in enumerate(words):
            key = f"{prefix}_w{i}"
            parts.append(
                f"({column} ILIKE :{key} OR word_similarity(:{key}_raw, {column}) > :{prefix}_wsim)"
            )
            params[key] = f"%{word}%"
            params[f"{key}_raw"] = word

    return "(" + " OR ".join(parts) + ")"


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


# 1. Doctors
def find_doctors(
    specialty: str | None = None,
    name: str | None = None,
    department: str | None = None,
    weekday: str | None = None,
    max_fee: float | Decimal | None = None,
    include_schedule: bool = False,
    limit: int = settings.sql_default_row_limit,
) -> SqlToolResult:
    """Find doctors by specialty, name, department, availability or fee.
    ``include_schedule`` is accepted and ignored - schedules always come back now."""
    params: dict[str, Any] = {"limit": min(limit, settings.sql_max_row_limit)}
    where: list[str] = []
    day = _normalise_weekday(weekday)

    if specialty:
        resolved = _resolve_specialty(specialty)
        if resolved:
            where.append("d.specialty = :specialty_exact")
            params["specialty_exact"] = resolved
        else:
            # Not confidently one specialty: match literally rather than guess.
            where.append("d.specialty ILIKE :specialty")
            params["specialty"] = f"%{specialty}%"
    if department:
        # Departments carry internal names ("Heart Centre" for Cardiology), so a
        # value here is often actually a specialty filed under the wrong argument.
        resolved_dept = _resolve_department(department)
        resolved_spec = None if resolved_dept else _resolve_specialty(department)
        if resolved_dept:
            where.append("d.department = :department_exact")
            params["department_exact"] = resolved_dept
        elif resolved_spec:
            where.append("d.specialty = :department_as_specialty")
            params["department_as_specialty"] = resolved_spec
        else:
            where.append("(d.department ILIKE :department OR d.specialty ILIKE :department)")
            params["department"] = f"%{department}%"
    if name:
        where.append(_name_clause(name, params))
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

    order_by = (
        "similarity(d.name_plain, :name_raw) DESC, d.doctor_name"
        if name
        else "d.consultation_fee ASC, d.doctor_name"
    )

    schedule_select = """,
        COALESCE((SELECT json_agg(json_build_object(
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
            ), CAST('[]' AS json)) AS sessions"""
    params["weekday_opt"] = day

    sql = f"""
        SELECT d.doctor_id, d.doctor_name, d.specialty, d.department,
               d.qualifications, d.consultation_fee,
               d.session_count, d.has_published_sessions{schedule_select}
        FROM v_doctor_directory d
        WHERE {clause}
        ORDER BY {order_by}
        LIMIT :limit
    """

    result = _run("find_doctors", sql, params)
    if result.error:
        return result

    if not result.rows and name and day:
        without_day = find_doctors(
            specialty=specialty, name=name, department=department, weekday=None,
            max_fee=max_fee, limit=limit,
        )
        if without_day.rows and not without_day.error:
            without_day.relaxed = True
            without_day.notes.insert(
                0,
                f"This doctor IS in the database but has NO {day} session. The rows "
                f"below are their full schedule, not {day} availability. Say plainly "
                f"that they do not channel on {day}, then give the days they do. "
                "Never say there is no record of them.",
            )
            return without_day

    _annotate_doctors(result, weekday=day)
    if day:
        result.notes.insert(
            0,
            f"Availability for {day} was resolved through v_sessions_expanded, "
            "which expands the literal 'Daily' schedule value into every weekday.",
        )
    # Empty results are handled centrally - see no_match_note.
    return result


def get_doctor_schedule(doctor: str) -> SqlToolResult:
    """Full channeling schedule for one doctor, matched fuzzily by name."""
    params: dict[str, Any] = {"limit": settings.sql_doctor_schedule_limit}
    clause = _name_clause(doctor, params)
    sql = f"""
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
               similarity(d.name_plain, :name_raw) AS match_score
        FROM v_doctor_directory d
        WHERE {clause}
        ORDER BY similarity(d.name_plain, :name_raw) DESC
        LIMIT :limit
    """
    result = _run("get_doctor_schedule", sql, params)
    if result.error:
        return result

    if not result.rows:
        # Empty results are handled centrally - see no_match_note.
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


# 2. Lab tests
def search_lab_tests(
    query: str | None = None,
    category: str | None = None,
    max_price: float | Decimal | None = None,
    fasting_required: bool | None = None,
    max_report_hours: int | None = None,
    limit: int = settings.sql_default_row_limit,
) -> SqlToolResult:
    """Search the priced laboratory test catalogue held in the database."""
    params: dict[str, Any] = {"limit": min(limit, settings.sql_max_row_limit)}
    where: list[str] = []

    if query:
        where.append(
            "(test_name ILIKE :q OR test_code ILIKE :q OR category ILIKE :q "
            "OR similarity(test_name, :q_raw) > :sim)"
        )
        params.update({"q": f"%{query}%", "q_raw": query,
                       "sim": settings.sql_name_similarity})
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
    # Same relax as search_health_packages; numeric/boolean filters survive as stated constraints.
    text_filters = (query, category)
    if not result.rows and any(text_filters):
        wider = search_lab_tests(
            max_price=max_price,
            fasting_required=fasting_required,
            max_report_hours=max_report_hours,
            limit=limit,
        )
        if wider.rows:
            wider.relaxed = True
            wider.notes.insert(0, _relaxed_note("test", _first(text_filters)))
            return wider

    # Empty results are handled centrally - see no_match_note.
    return result


# 3. Health packages
def search_health_packages(
    query: str | None = None,
    includes_service: str | None = None,
    max_price: float | Decimal | None = None,
    target_audience: str | None = None,
    category: str | None = None,
    limit: int = settings.sql_default_row_limit,
) -> SqlToolResult:
    """Search health check packages, including inside their contents text.

    ``includes_service`` searches *within* the free-text
    ``included_tests_and_services`` column. That column is not joinable to
    ``lab_tests``: it says "Lipid Profile" where the test table says "Advanced
    Lipid & ApoB Profile".
    """
    params: dict[str, Any] = {"limit": min(limit, settings.sql_max_row_limit)}
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

    # Drop every *text* filter in one step (the model rarely picks just one to
    # defend), but keep max_price - a number is a real stated constraint.
    text_filters = (query, includes_service, target_audience, category)
    if not result.rows and any(text_filters):
        wider = search_health_packages(max_price=max_price, limit=limit)
        if wider.rows:
            wider.relaxed = True
            wider.notes.insert(0, _relaxed_note("package", _first(text_filters)))
            return wider

    # Empty results are handled centrally - see no_match_note.
    return result



# 4. Specialties
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
        LIMIT :limit
    """
    result = _run("list_specialties", sql,
                  {"limit": settings.sql_specialty_list_limit})
    if not result.error:
        empty = [r["specialty"] for r in result.rows if not r["doctor_count"]]
        if empty:
            result.notes.append(
                "Specialties with no doctors currently listed: " + ", ".join(empty)
            )
    return result


# 5. Symptom -> specialty navigation (not diagnosis)
# Mapping lives in data/ontology/symptoms.yaml, matched semantically - see app/retrieval/ontology.py.


def looks_like_complaint(text: str) -> bool:
    """Is this phrase describing a problem, or naming a record? ("fever" is not a doctor's name.)"""
    from app.retrieval.ontology import looks_like_complaint as _looks

    return _looks(text)


# complaint -> candidate specialties -> the doctors in them
def suggest_specialty(complaint: str, matched: list[str] | None = None) -> SqlToolResult:
    """Which clinic normally sees this complaint, and who works in it.
    Returns zero rows (never a guessed fallback) when the ontology recognises nothing."""
    from app.retrieval.ontology import match_specialties

    matched = match_specialties(complaint) if matched is None else matched

    if not matched:
        result = SqlToolResult(tool="suggest_specialty")
        result.requested_filters = {"complaint": complaint}
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
        LIMIT :limit
    """
    result = _run("suggest_specialty", sql,
                  {"specialties": matched,
                   "limit": settings.sql_suggest_specialty_limit})
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
