"""Keyword + entity signals checked before calling the LLM router.

Matches a DB-built gazetteer (doctors, specialties, tests, packages) and
transactional/descriptive keyword sets; used as classifier evidence and for
post-hoc reconciliation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "weekend", "weekday", "today", "tomorrow")

# stuff only the database can answer
SQL_KEYWORDS = {
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

# Stuff that lives in the scraped website text. Split strong/weak because weak
VECTOR_KEYWORDS_STRONG = {
    "services", "service", "facility", "facilities", "centre", "center",
    "unit", "equipment", "technology", "policy", "policies",
    "terms", "conditions", "insurance", "accreditation", "award", "awards",
    "history", "mission", "vision", "chairman", "address",
    "location", "directions", "parking", "visiting", "international",
    "capabilities", "expertise", "offer", "offers", "provide", "provides",
    "do you have", "do you offer", "treatment", "treatments",
}

VECTOR_KEYWORDS_WEAK = {
    "about", "what is", "who is", "tell me about", "describe", "overview",
    "department", "contact", "treat", "treats",
}

VECTOR_KEYWORDS = VECTOR_KEYWORDS_STRONG | VECTOR_KEYWORDS_WEAK

# Plain nouns for record types the DB holds, catching "list all doctors"-style
RECORD_NOUNS = frozenset({
    "doctor", "doctors", "doctor's", "doctors'",
    "consultant", "consultants", "consultants'",
    "physician", "physicians", "surgeon", "surgeons",
})


@dataclass
class RouteSignals:
    sql_hits: list[str] = field(default_factory=list)
    vector_hits: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    # Record-type nouns the query used without naming a specific record.
    record_nouns: list[str] = field(default_factory=list)
    #Whether the gazetteer had loaded when these were extracted; distinguishes
    gazetteer_ready: bool = False

    @property
    def has_sql(self) -> bool:
        return bool(self.sql_hits) or bool(self.record_nouns) or bool(
            {"doctor", "lab_test", "package", "weekday"} & set(self.entities)
        )

    @property
    def has_vector(self) -> bool:
        return bool(self.vector_hits)

    @property
    def strong_vector_hits(self) -> list[str]:
        return [h for h in self.vector_hits if h in VECTOR_KEYWORDS_STRONG]

    @property
    def has_strong_vector(self) -> bool:
        return bool(self.strong_vector_hits)

    # Strong terms only; weak ones over-trigger hybrid on plain SQL questions.
    @property
    def both(self) -> bool:
        return self.has_sql and self.has_strong_vector

    @property
    def has_structured_record(self) -> bool:
        """Whether the query names a DB record or attribute a SQL lookup could plausibly match."""
        return bool(self.entities) or bool(self.sql_hits) or bool(self.record_nouns)

    def to_dict(self) -> dict:
        return {
            "sql_hits": self.sql_hits,
            "vector_hits": self.vector_hits,
            "strong_vector_hits": self.strong_vector_hits,
            "entities": self.entities,
            "record_nouns": self.record_nouns,
            "has_sql": self.has_sql,
            "has_vector": self.has_vector,
            "has_structured_record": self.has_structured_record,
            "gazetteer_ready": self.gazetteer_ready,
        }

    # evidence string we hand to the LLM classifier
    def summary(self) -> str:
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
        if not parts:
            parts.append("no strong keyword signals")
        if not self.has_structured_record:
            parts.append(
                "names no known database record and no database attribute - "
                "nothing here for a SQL lookup to match"
            )
        return "\n".join(parts)


@dataclass
class Gazetteer:
    doctor_terms: dict[str, str] = field(default_factory=dict)
    specialty_terms: dict[str, str] = field(default_factory=dict)
    lab_terms: dict[str, str] = field(default_factory=dict)
    package_terms: dict[str, str] = field(default_factory=dict)
    #: Words that appear across too many record names to identify one, computed
    #: from the data at build time. Kept on the gazetteer so it is visible and
    #: testable rather than hidden inside the build.
    shared_words: frozenset[str] = frozenset()

    @property
    def size(self) -> int:
        total = 0
        for terms in (self.doctor_terms, self.specialty_terms,
                      self.lab_terms, self.package_terms):
            total += len(terms)
        return total


# Domain-free fallback stoplist for when nothing can be computed from the data.
_STOPWORDS = {"the", "and", "for", "with", "of", "from"}
#: Minimum number of distinct record names a word must appear in to be dropped
_SHARED_WORD_CUTOFF = settings.signals_shared_word_cutoff


def _shared_words(names: list[str], cutoff: int = _SHARED_WORD_CUTOFF) -> frozenset[str]:
    """Words too common across these names to identify any one of them."""
    counts = Counter()
    for name in names:
        for word in set(re.findall(r"[a-z0-9][a-z0-9'-]{2,}", name.lower())):
            counts[word] += 1
    return frozenset(w for w, n in counts.items() if n >= cutoff)


# the words worth matching on, for partial name matches
def _useful_words(text: str, min_len: int | None = None,
                  stop: frozenset[str] | None = None) -> list[str]:
    if min_len is None:
        min_len = settings.signals_word_min_len
    words = re.findall(r"[a-z0-9][a-z0-9'-]{2,}", text.lower())
    out = []
    for w in words:
        if len(w) < min_len or w in _STOPWORDS:
            continue
        if stop and w in stop:
            continue
        out.append(w)
    return out



_SUFFIX_SWAPS = (
    ("ology", "ologist"),
    ("iatrics", "iatrician"),
    ("surgery", "surgeon"),
    ("medicine", "physician"),
)


# what a patient might call a doctor in this specialty
def _practitioner_words(specialty: str) -> set[str]:
    out: set[str] = set()

    try:
        from app.retrieval.ontology import specialty_aliases

        aliases = specialty_aliases()
        out.update(aliases.get(specialty, ()))
        if not out:
            # Tolerate case drift between the YAML and the table. load_ontology
            # already rejects a name that isn't in the table at all, so this
            # only ever catches capitalisation.
            lowered = {k.lower(): v for k, v in aliases.items()}
            out.update(lowered.get(specialty.lower(), ()))
    except Exception as exc:
        # Morphology alone still covers the regular forms. Note this import must
        # stay function-local: ontology imports build_gazetteer from this module.
        log.warning("signals.aliases_unavailable", error=str(exc)[:160])

    for word in re.findall(r"[a-z]+", specialty.lower()):
        for suffix, replacement in _SUFFIX_SWAPS:
            if word.endswith(suffix) and len(word) > len(suffix):
                out.add(word[: -len(suffix)] + replacement)

    keep = set()
    for v in out:
        if len(v) >= settings.signals_alias_min_len:
            keep.add(v)
    return keep


# Pulls entity names out of the DB. Comes back empty if the DB is down, which
@lru_cache(maxsize=1)
def build_gazetteer() -> Gazetteer:
    from app.tools.db import fetch

    gz = Gazetteer()
    try:
        # Pass 1: read everything first so the stoplist spans all tables.
        doctors, _ = fetch("SELECT doctor_name, name_plain FROM v_doctor_directory")
        specialties, _ = fetch("SELECT name, department FROM specialties")
        tests, _ = fetch("SELECT test_name, test_code FROM lab_tests")
        packages, _ = fetch("SELECT package_name FROM health_packages")

        shared = _shared_words(
            [str(r["name_plain"]) for r in doctors]
            + [str(r["name"]) for r in specialties]
            + [str(r["department"]) for r in specialties]
            + [str(r["test_name"]) for r in tests]
            + [str(r["package_name"]) for r in packages]
        )
        gz.shared_words = shared

        # Pass 2: index terms, skipping shared/non-distinctive words; full names always kept.
        for row in doctors:
            full = row["doctor_name"]
            gz.doctor_terms[row["name_plain"].lower()] = full
            for term in _useful_words(row["name_plain"],
                                      min_len=settings.signals_doctor_word_min_len,
                                      stop=shared):
                gz.doctor_terms.setdefault(term, full)

        for row in specialties:
            gz.specialty_terms[row["name"].lower()] = row["name"]
            gz.specialty_terms[row["department"].lower()] = row["name"]
            for term in _useful_words(row["name"], stop=shared):
                gz.specialty_terms.setdefault(term, row["name"])
            for variant in _practitioner_words(row["name"]):
                gz.specialty_terms.setdefault(variant, row["name"])

        for row in tests:
            gz.lab_terms[row["test_name"].lower()] = row["test_name"]
            gz.lab_terms[row["test_code"].lower()] = row["test_name"]
            for term in _useful_words(row["test_name"],
                                      min_len=settings.signals_lab_word_min_len,
                                      stop=shared):
                gz.lab_terms.setdefault(term, row["test_name"])

        for row in packages:
            gz.package_terms[row["package_name"].lower()] = row["package_name"]

        log.info("signals.gazetteer_built", terms=gz.size,
                 doctors=len(gz.doctor_terms), specialties=len(gz.specialty_terms),
                 lab=len(gz.lab_terms), packages=len(gz.package_terms),
                 shared_words=len(shared))
    except Exception as exc:
        log.warning("signals.gazetteer_unavailable", error=str(exc)[:160])

    return gz


# Lookarounds, not \b. \b only matches next to word chars, so a term like
def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _phrase_hits(text: str, keywords: set[str]) -> list[str]:
    hits = []
    for term in keywords:
        if _contains_term(text, term):
            hits.append(term)
    return hits


def extract_signals(query: str) -> RouteSignals:
    # TODO: the space padding is left over from when this used \b - the
    # lookarounds in _contains_term handle string edges fine, so probably drop it
    text = " " + (query or "").lower().strip() + " "
    gz = build_gazetteer()
    signals = RouteSignals(gazetteer_ready=bool(gz.size))

    signals.sql_hits = _phrase_hits(text, SQL_KEYWORDS)
    signals.vector_hits = _phrase_hits(text, VECTOR_KEYWORDS)
    signals.record_nouns = _phrase_hits(text, RECORD_NOUNS)

    entities: dict[str, list[str]] = {}

    def match(terms: dict[str, str], bucket: str) -> None:
        found: list[str] = []
        for term, canonical in terms.items():
            if len(term) < settings.signals_term_min_len:
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
