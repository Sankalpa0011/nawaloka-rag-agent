"""Pydantic argument schemas for the curated SQL tools - one model per tool,
shared by both LangChain (`args_schema`) and LlamaIndex (`fn_schema`).
reject_bool() and reject_complaint() catch two specific model mistakes
structurally; unknown arguments die on `extra="forbid"`.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from app.config import settings


class Weekday(str, Enum):
    monday = "Monday"
    tuesday = "Tuesday"
    wednesday = "Wednesday"
    thursday = "Thursday"
    friday = "Friday"
    saturday = "Saturday"
    sunday = "Sunday"

    # Accept the shorthands people actually type. Replaces the alias dict that
    # used to live in sql_tools._normalise_weekday.
    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        v = value.strip().lower()
        for member in cls:
            if member.value.lower() == v or member.value.lower().startswith(v[:3]) and len(v) >= 3:
                return member
        return None


# Values that match every row, so passing one can only empty a result set.
# "which doctors practise here?" came back as department="Nawaloka Hospitals",
# which is ILIKE '%Nawaloka Hospitals%', zero rows, and a confident "I do not
# have that information" about a directory of eleven doctors.
IGNORED_FILTERS = {
    "nawaloka", "nawaloka hospital", "nawaloka hospitals",
    "nawaloka hospitals plc", "nawaloka hospital plc",
    "hospital", "the hospital", "this hospital",
    "all", "any", "*", "none", "null", "n/a", "na", "nil",
    "unknown", "unspecified", "everything", "everyone",
    "all departments", "any department", "all specialties", "any specialty",
    "all doctors", "all tests", "all packages",
}


def _normalise(value: str) -> str:
    return " ".join(value.lower().split()).strip(" .,'\"")


def drop_if_meaningless(value: str | None) -> str | None:
    """Turn a filter that matches everything into no filter at all."""
    if value is None:
        return None
    if _normalise(value) in IGNORED_FILTERS:
        return None
    return value


class ComplaintInNameError(ValueError):
    """A symptom was supplied where a record's name belongs; the SQL agent recovers from exactly this type."""

    def __init__(self, value: str, tool_hint: str = "suggest_specialty") -> None:
        self.value = value
        self.tool_hint = tool_hint
        super().__init__(
            f"{value!r} is a symptom, not a doctor's name. "
            f"Call {tool_hint} with it as the complaint instead."
        )


def reject_complaint(value: str | None) -> str | None:
    """Refuse a symptom where a person's name belongs (e.g. "fever" is not a doctor's name)."""
    if value is None:
        return None

    from app.tools.sql_tools import looks_like_complaint

    if looks_like_complaint(value):
        raise ComplaintInNameError(value)
    return value


def reject_bool(value):
    """Refuse true/false where a number belongs (bool is an int subclass and would coerce silently)."""
    if isinstance(value, bool):
        raise ValueError("expected a number, not true/false")
    return value


# A filter value, cleaned of the placeholders models reach for.
FilterText = Annotated[str | None, AfterValidator(drop_if_meaningless)]

Money = Annotated[float | None, BeforeValidator(reject_bool), Field(gt=0)]
Hours = Annotated[int | None, BeforeValidator(reject_bool), Field(gt=0)]


# What an argument *is*, declared on the field itself, so recovery logic follows
# the tool instead of naming fields in hardcoded lists.
#   subject     the thing the patient named; worth re-reading/broadening if empty.
#   constraint  a stated limit; an empty result may be entirely correct.
#   pagination  row count; never a reason for an empty result.
#   noop        accepted and ignored; shapes output, not matching.
Subject = {"role": "subject"}
Constraint = {"role": "constraint"}
Pagination = {"role": "pagination"}
NoOp = {"role": "noop"}


class ToolArgs(BaseModel):
    """Shared config. extra="forbid" replaces the unknown-argument branch."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class FindDoctorsArgs(ToolArgs):
    """Doctors by specialty, name, department, availability or fee."""

    specialty: FilterText = Field(None, json_schema_extra=Subject)
    name: Annotated[str | None, AfterValidator(drop_if_meaningless),
                    AfterValidator(reject_complaint)] = Field(
                        None, json_schema_extra=Subject)
    department: FilterText = Field(None, json_schema_extra=Subject)
    weekday: Weekday | None = Field(None, json_schema_extra=Constraint)
    max_fee: Money = Field(None, json_schema_extra=Constraint)
    include_schedule: bool = Field(False, json_schema_extra=NoOp)
    limit: int = Field(
        settings.sql_default_row_limit,
        ge=1,
        le=settings.sql_max_row_limit,
        json_schema_extra=Pagination,
    )


class GetDoctorScheduleArgs(ToolArgs):
    """One named doctor's channeling times."""

    doctor: str = Field(min_length=2, json_schema_extra=Subject)


class SearchLabTestsArgs(ToolArgs):
    """Lab test prices, fasting rules and report turnaround."""

    query: FilterText = Field(None, json_schema_extra=Subject)
    category: FilterText = Field(None, json_schema_extra=Subject)
    max_price: Money = Field(None, json_schema_extra=Constraint)
    fasting_required: bool | None = Field(None, json_schema_extra=Constraint)
    max_report_hours: Hours = Field(None, json_schema_extra=Constraint)
    limit: int = Field(
        settings.sql_default_row_limit,
        ge=1,
        le=settings.sql_max_row_limit,
        json_schema_extra=Pagination,
    )


class SearchHealthPackagesArgs(ToolArgs):
    """Health check packages. includes_service searches inside the contents."""

    query: FilterText = Field(None, json_schema_extra=Subject)
    includes_service: FilterText = Field(None, json_schema_extra=Subject)
    max_price: Money = Field(None, json_schema_extra=Constraint)
    target_audience: FilterText = Field(None, json_schema_extra=Subject)
    category: FilterText = Field(None, json_schema_extra=Subject)
    limit: int = Field(
        settings.sql_default_row_limit,
        ge=1,
        le=settings.sql_max_row_limit,
        json_schema_extra=Pagination,
    )


class ListSpecialtiesArgs(ToolArgs):
    """Every specialty and department. Takes no arguments."""


class SuggestSpecialtyArgs(ToolArgs):
    """Map a described complaint to the clinic that normally sees it."""

    complaint: str = Field(min_length=2, json_schema_extra=Subject)


TOOL_ARGS: dict[str, type[ToolArgs]] = {
    "find_doctors": FindDoctorsArgs,
    "get_doctor_schedule": GetDoctorScheduleArgs,
    "search_lab_tests": SearchLabTestsArgs,
    "search_health_packages": SearchHealthPackagesArgs,
    "list_specialties": ListSpecialtiesArgs,
    "suggest_specialty": SuggestSpecialtyArgs,
}


# Reading the declarations back (callers avoid touching model_fields directly)
def tool_names() -> tuple[str, ...]:
    """Every curated tool, in a stable order."""
    return tuple(TOOL_ARGS)


def _role_of(model: type[ToolArgs], field: str) -> str:
    info = model.model_fields.get(field)
    extra = getattr(info, "json_schema_extra", None) if info else None
    if isinstance(extra, dict):
        return str(extra.get("role", ""))
    return ""


def fields_with_role(tool: str, role: str) -> tuple[str, ...]:
    """Argument names on `tool` declared with this role, in declaration order."""
    model = TOOL_ARGS.get(tool)
    if model is None:
        return ()
    return tuple(f for f in model.model_fields if _role_of(model, f) == role)


def required_fields(tool: str) -> tuple[str, ...]:
    """Arguments without which `tool` cannot be called at all."""
    model = TOOL_ARGS.get(tool)
    if model is None:
        return ()
    return tuple(f for f, info in model.model_fields.items() if info.is_required())


def has_required_args(tool: str) -> bool:
    """Is "call it with no arguments" even a legal instruction for this tool?"""
    return bool(required_fields(tool))


def complaint_value(exc: ValidationError) -> str | None:
    """The value reject_complaint refused, if that is why validation failed.

    Pydantic keeps the raised exception object in ctx["error"] for value errors,
    so this is an isinstance check on our own type. The second branch is a
    fallback for pydantic builds that drop the object: it is deliberately
    narrower than the old substring test, requiring the failure to be on a
    subject-role field rather than trusting the message alone.
    """
    for err in exc.errors():
        cause = (err.get("ctx") or {}).get("error")
        if isinstance(cause, ComplaintInNameError):
            return cause.value
    for err in exc.errors():
        if err.get("type") != "value_error":
            continue
        value = err.get("input")
        if isinstance(value, str) and "symptom" in str(err.get("msg", "")):
            return value
    return None
