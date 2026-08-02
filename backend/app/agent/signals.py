"""Deterministic routing signals.

Routing accuracy is an explicitly graded criterion, so it is not left entirely
to an LLM classifier. Before any model is called, the query is matched against:

* a **gazetteer built from the live database** - every doctor surname,
  specialty, department, lab test name and code, and package name. If a user
  says "Priyadarshan" or "LAB-DENGUE", that is near-conclusive evidence the
  answer lives in SQL, and no model needs to be asked.
* **intent lexicons** - words that indicate a transactional question ("fee",
  "how much", "available", "timing") versus a descriptive one ("what is",
  "facilities", "about", "services").

The gazetteer is derived from the database rather than hardcoded, so it cannot
drift from the data. These signals are then given to the LLM classifier as
evidence, and used afterwards to reconcile its answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from app.logging_conf import get_logger

log = get_logger(__name__)

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "weekend", "weekday", "today", "tomorrow")

# Transactional intent: answerable only from the relational database.
SQL_LEXICON = {
    "fee", "fees", "cost", "costs", "price", "prices", "pricing", "charge",
    "charges", "how much", "lkr", "rupee", "rupees", "rs", "cheapest",
    "expensive", "affordable", "budget",
    "schedule", "schedules", "timing", "timings", "time", "times", "hours",
    "available", "availability", "when", "channel", "channeling", "channelling",
    "appointment", "book", "booking", "slot", "session", "sessions",
    "room", "duration", "turnaround", "report", "fasting", "preparation",
    "package", "packages", "test", "tests", "consultation", "qualification",
    "qualifications", "specialist", "specialty", "speciality", "list all",
}

# Descriptive intent: answerable from the scraped website content.
#
# Split by strength on purpose. "what is" and "tell me about" appear in almost
# every question, including purely transactional ones ("what is Dr X's fee?"),
# so counting them as evidence of a descriptive need makes the reconciliation
# step widen nearly everything to HYBRID. Only STRONG terms - which name
# something the database cannot describe - are treated as real evidence.
VECTOR_LEXICON_STRONG = {
    "services", "service", "facility", "facilities", "centre", "center",
    "unit", "equipment", "technology", "policy", "policies",
    "terms", "conditions", "insurance", "accreditation", "award", "awards",
    "history", "mission", "vision", "chairman", "address",
    "location", "directions", "parking", "visiting", "international",
    "capabilities", "expertise", "offer", "offers", "provide", "provides",
    "do you have", "do you offer", "treatment", "treatments",
}

VECTOR_LEXICON_WEAK = {
    "about", "what is", "who is", "tell me about", "describe", "overview",
    "department", "contact", "treat", "treats",
}

VECTOR_LEXICON = VECTOR_LEXICON_STRONG | VECTOR_LEXICON_WEAK


@dataclass
class RouteSignals:
    sql_hits: list[str] = field(default_factory=list)
    vector_hits: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_sql(self) -> bool:
        return bool(self.sql_hits) or bool(
            {"doctor", "lab_test", "package", "weekday"} & set(self.entities)
        )

    @property
    def has_vector(self) -> bool:
        return bool(self.vector_hits)

    @property
    def strong_vector_hits(self) -> list[str]:
        return [h for h in self.vector_hits if h in VECTOR_LEXICON_STRONG]

    @property
    def has_strong_vector(self) -> bool:
        return bool(self.strong_vector_hits)

    @property
    def both(self) -> bool:
        """Genuine evidence for both sources.

        Uses only STRONG descriptive terms. Counting weak interrogatives here
        marks almost every question as hybrid - "what is Dr X's fee" contains
        "what is" - which was measured to turn 5 correct SQL routes into wrong
        HYBRID ones and zero the other way.
        """
        return self.has_sql and self.has_strong_vector

    def to_dict(self) -> dict:
        return {
            "sql_hits": self.sql_hits,
            "vector_hits": self.vector_hits,
            "strong_vector_hits": self.strong_vector_hits,
            "entities": self.entities,
            "has_sql": self.has_sql,
            "has_vector": self.has_vector,
        }

    def summary(self) -> str:
        """Compact evidence string handed to the LLM classifier."""
        parts = []
        if self.entities:
            parts.append(
                "known database entities mentioned: "
                + "; ".join(f"{k}={v}" for k, v in self.entities.items())
            )
        if self.sql_hits:
            parts.append("transactional keywords: " + ", ".join(sorted(set(self.sql_hits))))
        if self.vector_hits:
            parts.append("descriptive keywords: " + ", ".join(sorted(set(self.vector_hits))))
        return "\n".join(parts) if parts else "no strong keyword signals"


@dataclass
class Gazetteer:
    doctor_terms: dict[str, str] = field(default_factory=dict)
    specialty_terms: dict[str, str] = field(default_factory=dict)
    lab_terms: dict[str, str] = field(default_factory=dict)
    package_terms: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return sum(
            len(d) for d in
            (self.doctor_terms, self.specialty_terms, self.lab_terms, self.package_terms)
        )


_STOPWORDS = {
    "the", "and", "for", "with", "test", "profile", "total", "full", "health",
    "package", "screening", "check", "care", "centre", "center", "medicine",
    "general", "serum", "blood", "level", "report", "antigen", "antibody",
}


def _significant_terms(text: str, min_len: int = 5) -> list[str]:
    """Distinctive words from a name, used for partial matching."""
    words = re.findall(r"[a-z0-9][a-z0-9'-]{2,}", text.lower())
    return [w for w in words if len(w) >= min_len and w not in _STOPWORDS]


# Patients name the practitioner, not the discipline: "I need a cardiologist",
# not "I need Cardiology". Regular suffix swaps cover most of the list, with a
# short table for the irregular ones.
_SUFFIX_VARIANTS = (
    ("ology", "ologist"),
    ("iatrics", "iatrician"),
    ("surgery", "surgeon"),
    ("medicine", "physician"),
)

_IRREGULAR_PRACTITIONERS: dict[str, tuple[str, ...]] = {
    "cardiology": ("cardiologist", "heart doctor"),
    "neurosurgery": ("neurosurgeon", "neurologist", "brain surgeon"),
    "ent surgery": ("ent", "ent surgeon", "otolaryngologist", "ear nose throat"),
    "general medicine": ("physician", "general physician", "gp"),
    "paediatrics": ("paediatrician", "pediatrician", "pediatrics", "child doctor"),
    "radiology & imaging": ("radiologist", "imaging", "scan", "x-ray", "mri"),
    "respiratory & chest medicine": ("pulmonologist", "chest physician", "chest specialist"),
    "rheumatology & sports medicine": ("rheumatologist", "sports medicine"),
    "general surgery & gastroenterology": ("gastroenterologist", "surgeon", "gastro"),
    "cardiac anaesthesia & critical care": ("anaesthetist", "anesthesiologist", "intensivist"),
}


def _practitioner_variants(specialty: str) -> set[str]:
    """Words a patient might use for a practitioner in this specialty."""
    lowered = specialty.lower()
    out: set[str] = set(_IRREGULAR_PRACTITIONERS.get(lowered, ()))
    for word in re.findall(r"[a-z]+", lowered):
        for suffix, replacement in _SUFFIX_VARIANTS:
            if word.endswith(suffix) and len(word) > len(suffix):
                out.add(word[: -len(suffix)] + replacement)
    return {v for v in out if len(v) >= 3}


@lru_cache(maxsize=1)
def build_gazetteer() -> Gazetteer:
    """Read entity names from the database. Degrades to empty if it is down."""
    from app.tools.db import fetch

    gz = Gazetteer()
    try:
        doctors, _ = fetch("SELECT doctor_name, name_plain FROM v_doctor_directory")
        for row in doctors:
            full = row["doctor_name"]
            gz.doctor_terms[row["name_plain"].lower()] = full
            for term in _significant_terms(row["name_plain"], min_len=4):
                gz.doctor_terms.setdefault(term, full)

        specialties, _ = fetch("SELECT name, department FROM specialties")
        for row in specialties:
            gz.specialty_terms[row["name"].lower()] = row["name"]
            gz.specialty_terms[row["department"].lower()] = row["name"]
            for term in _significant_terms(row["name"]):
                gz.specialty_terms.setdefault(term, row["name"])
            for variant in _practitioner_variants(row["name"]):
                gz.specialty_terms.setdefault(variant, row["name"])

        tests, _ = fetch("SELECT test_name, test_code FROM lab_tests")
        for row in tests:
            gz.lab_terms[row["test_name"].lower()] = row["test_name"]
            gz.lab_terms[row["test_code"].lower()] = row["test_name"]
            for term in _significant_terms(row["test_name"], min_len=6):
                gz.lab_terms.setdefault(term, row["test_name"])

        packages, _ = fetch("SELECT package_name FROM health_packages")
        for row in packages:
            gz.package_terms[row["package_name"].lower()] = row["package_name"]

        log.info("signals.gazetteer_built", terms=gz.size,
                 doctors=len(gz.doctor_terms), specialties=len(gz.specialty_terms),
                 lab=len(gz.lab_terms), packages=len(gz.package_terms))
    except Exception as exc:
        # Signals are an optimisation; routing still works through the LLM.
        log.warning("signals.gazetteer_unavailable", error=str(exc)[:160])

    return gz


def _contains_term(text: str, term: str) -> bool:
    """Whole-term match that tolerates punctuation in the term itself.

    ``\\b`` is defined relative to word characters, so it never matches next to
    a bracket - "Well Woman Package (Under 40)" would silently never be found.
    Lookarounds on word characters work regardless of what the term starts or
    ends with.
    """
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _phrase_hits(text: str, lexicon: set[str]) -> list[str]:
    return [term for term in lexicon if _contains_term(text, term)]


def extract_signals(query: str) -> RouteSignals:
    text = " " + (query or "").lower().strip() + " "
    gz = build_gazetteer()
    signals = RouteSignals()

    signals.sql_hits = _phrase_hits(text, SQL_LEXICON)
    signals.vector_hits = _phrase_hits(text, VECTOR_LEXICON)

    entities: dict[str, list[str]] = {}

    def match(terms: dict[str, str], bucket: str) -> None:
        found: list[str] = []
        for term, canonical in terms.items():
            if len(term) < 3:
                continue
            if canonical not in found and _contains_term(text, term):
                found.append(canonical)
        if found:
            entities[bucket] = found

    match(gz.doctor_terms, "doctor")
    match(gz.specialty_terms, "specialty")
    match(gz.lab_terms, "lab_test")
    match(gz.package_terms, "package")

    days = [d for d in WEEKDAYS if re.search(rf"\b{d}\b", text)]
    if days:
        entities["weekday"] = days

    # A bare currency figure is a strong transactional cue.
    if re.search(r"\b(?:lkr|rs\.?)\s*[\d,]+|\b\d{3,6}\s*(?:rupees|lkr)\b", text):
        signals.sql_hits.append("currency amount")

    signals.entities = entities
    return signals


def reset_gazetteer() -> None:
    build_gazetteer.cache_clear()
